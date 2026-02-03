# utils/CAMUSPreprocess.py
#
# Goal: Make CAMUS preprocessing outputs match ACDC_ED_Extract_Split_SDF.py structure:
#   out_root/
#     split.json
#     train/patient0001_2CH_ED/
#        img.npy        (H,W) float32 in [-1,1]
#        onehot.npy     (4,H,W) uint8
#        sdf.npy        (4,H,W) float32 in [-1,1]   (clipped to [-clip,clip] then /clip)
#     val/...
#     test/...
#
# Each (patient + view + phase) becomes ONE sample folder.
# Use both 2CH and 4CH; use ED+ES by default.
#
# Usage:
#   python utils/CAMUSPreprocess.py \
#       --base_dir /path/to/CAMUSdataset \
#       --out_root ./dataset/CAMUS/SDF \
#       --size 256 \
#       --p_lo 1 --p_hi 99 \
#       --clip 10 \
#       --seed 42 \
#       --use_views 2CH 4CH \
#       --use_phases ED ES
#
# Notes:
# - Expects CAMUS folders: base_dir/patient0001/...
# - Expects files:
#     patient0001_2CH_ED.nii.gz
#     patient0001_2CH_ED_gt.nii.gz
#     patient0001_2CH_ES.nii.gz
#     patient0001_2CH_ES_gt.nii.gz
#     patient0001_4CH_ED.nii.gz
#     patient0001_4CH_ED_gt.nii.gz
#     patient0001_4CH_ES.nii.gz
#     patient0001_4CH_ES_gt.nii.gz
# - half_sequence files are ignored.

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from typing import Tuple, List, Dict

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


# -------------------------
# Utils (aligned with your ACDC script style)
# -------------------------
def _ensure_dirs(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)


def center_pad_or_crop_hw(x: np.ndarray, out_hw=(256, 256), pad_value=0):
    """
    x: (H,W) or (H,W,1)
    """
    original_ndim = x.ndim
    if original_ndim == 2:
        x = x[:, :, None]
        squeeze_back = True
    else:
        squeeze_back = False

    H, W, C = x.shape
    out_h, out_w = out_hw

    # pad
    pad_h = max(0, out_h - H)
    pad_w = max(0, out_w - W)
    if pad_h > 0 or pad_w > 0:
        pt = pad_h // 2
        pb = pad_h - pt
        pl = pad_w // 2
        pr = pad_w - pl
        x = np.pad(
            x,
            ((pt, pb), (pl, pr), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        H, W, _ = x.shape

    # crop
    if H > out_h:
        top = (H - out_h) // 2
        x = x[top: top + out_h, :, :]
    if W > out_w:
        left = (W - out_w) // 2
        x = x[:, left: left + out_w, :]

    if squeeze_back:
        x = x[:, :, 0]
    return x


def minmax01_single(
    x: np.ndarray, eps=1e-8, clip_percentiles=(1.0, 99.0)
) -> np.ndarray:
    """
    Percentile-clip + min-max normalization to [-1,1].

    x: (H,W) float
    return: (H,W) float32 in [-1,1]
    """
    x = x.astype(np.float32, copy=False)
    p_lo, p_hi = clip_percentiles

    lo, hi = np.percentile(x, p_lo), np.percentile(x, p_hi)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)  # constant image -> 0

    x = np.clip(x, lo, hi)
    x01 = (x - lo) / (hi - lo + eps)          # [0,1]
    return (x01 * 2.0 - 1.0).astype(np.float32, copy=False)  # [-1,1]


def _to_hw_if_needed(arr: np.ndarray) -> np.ndarray:
    """
    CAMUS ED/ES frames are typically 2D. nibabel may return (H,W) or (H,W,1).
    Convert to (H,W).
    """
    if arr.ndim == 2:
        return arr
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed
    raise ValueError(f"Unexpected CAMUS frame shape: {arr.shape}")


def to_onehot_4(gt2d: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    gt2d: (H,W) integer labels in {0,1,2,3}
    return: (C,H,W) uint8 one-hot
    """
    gt2d = gt2d.astype(np.int64, copy=False)
    gt2d = np.clip(gt2d, 0, num_classes - 1)
    onehot = np.eye(num_classes, dtype=np.uint8)[gt2d]  # (H,W,C)
    onehot = np.transpose(onehot, (2, 0, 1))           # (C,H,W)
    return onehot


def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = True,          # keep consistent with your ACDC script usage
    normalize_eps: float = 1e-8
) -> np.ndarray:
    """
    onehot: (C,H,W) uint8/bool (C=4)
    returns sdf: (C,H,W) float32
        inside mask  -> negative distance
        outside mask -> positive distance
        boundary -> 0

    If normalize=True:
        sdf_out = clip(sdf_raw, -clip, clip) / clip  => [-1,1]
    """
    if onehot.ndim != 3:
        raise ValueError(f"Expected onehot shape (C,H,W), got {onehot.shape}")
    if clip is None:
        raise ValueError("clip cannot be None.")
    if normalize and clip <= 0:
        raise ValueError(f"clip must be > 0 when normalize=True, got clip={clip}")

    C, H, W = onehot.shape
    sdf = np.zeros((C, H, W), dtype=np.float32)

    for c in range(C):
        mask = onehot[c].astype(bool)

        if mask.any():
            dist_in = distance_transform_edt(mask)
            dist_out = distance_transform_edt(~mask)
            sdf_c = dist_out - dist_in
        else:
            sdf_c = np.full((H, W), fill_value=clip, dtype=np.float32)

        sdf_c = np.clip(sdf_c, -clip, clip).astype(np.float32)

        if normalize:
            sdf_c = sdf_c / (float(clip) + normalize_eps)

        sdf[c] = sdf_c

    return sdf


# -------------------------
# Split (patient-level, same as ACDC style)
# -------------------------
def list_patients_from_camus(base_dir: Path) -> List[str]:
    """
    CAMUSdataset/patient0001, patient0002, ...
    """
    base_dir = Path(base_dir)
    pats = [p.name for p in sorted(base_dir.glob("patient*")) if p.is_dir()]
    return pats


def make_split(patients: List[str], seed=42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    pats = patients.copy()
    rng.shuffle(pats)

    n = len(pats)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    # remaining -> test

    train = pats[:n_train]
    val = pats[n_train:n_train + n_val]
    test = pats[n_train + n_val:]
    return train, val, test


def save_split_json(out_path: Path, seed: int, train, val, test):
    obj = {
        "seed": seed,
        "counts": {
            "total": len(train) + len(val) + len(test),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# CAMUS discovery helpers
# -------------------------
_ALLOWED_VIEWS = {"2CH", "4CH"}
_ALLOWED_PHASES = {"ED", "ES"}


def _is_valid_patient_dir(p: Path) -> bool:
    return p.is_dir() and re.match(r"^patient\d{4}$", p.name) is not None


def _paths_for_item(patient_dir: Path, patient_name: str, view: str, phase: str) -> Tuple[Path, Path]:
    img_path = patient_dir / f"{patient_name}_{view}_{phase}.nii.gz"
    gt_path = patient_dir / f"{patient_name}_{view}_{phase}_gt.nii.gz"
    return img_path, gt_path


# -------------------------
# Main pipeline
# -------------------------
def process_one_patient(
    base_dir: Path,
    patient_name: str,
    out_split_dir: Path,
    out_hw: Tuple[int, int],
    clip_percentiles: Tuple[float, float],
    sdf_clip: float,
    use_views: List[str],
    use_phases: List[str],
) -> int:
    """
    Each (patient + view + phase) becomes ONE folder:

        out_root/train/patient0001_2CH_ED/
            img.npy
            onehot.npy
            sdf.npy

    Returns number of samples saved for this patient.
    """
    patient_dir = Path(base_dir) / patient_name
    if not patient_dir.exists():
        print(f"[WARN] Missing patient dir: {patient_dir}")
        return 0

    saved = 0

    for view in use_views:
        for phase in use_phases:
            img_path, gt_path = _paths_for_item(patient_dir, patient_name, view, phase)
            if not img_path.exists() or not gt_path.exists():
                continue

            # sample folder name: patient0001_2CH_ED
            sample_name = f"{patient_name}_{view}_{phase}"
            out_sample_dir = out_split_dir / sample_name
            out_sample_dir.mkdir(parents=True, exist_ok=True)

            # load
            img = nib.load(str(img_path)).get_fdata()
            gt = nib.load(str(gt_path)).get_fdata()

            img2d = _to_hw_if_needed(img).astype(np.float32)
            gt2d = _to_hw_if_needed(gt).astype(np.int16)

            # pad/crop to fixed size
            img2d = center_pad_or_crop_hw(img2d, out_hw=out_hw, pad_value=0.0)
            gt2d = center_pad_or_crop_hw(gt2d, out_hw=out_hw, pad_value=0)

            # normalize image to [-1,1]
            img_norm = minmax01_single(img2d, clip_percentiles=clip_percentiles)

            # onehot + sdf
            gt_onehot = to_onehot_4(gt2d, num_classes=4)  # (4,H,W)
            gt_sdf = onehot_to_sdf_per_channel(gt_onehot, clip=sdf_clip, normalize=True)  # (4,H,W) in [-1,1]

            # save (ACDC-compatible filenames)
            np.save(out_sample_dir / "img.npy", img_norm.astype(np.float32))
            np.save(out_sample_dir / "onehot.npy", gt_onehot)
            np.save(out_sample_dir / "sdf.npy", gt_sdf.astype(np.float32))

            saved += 1

    return saved


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    seed: int,
    split_json: Path | None,
    use_views: List[str],
    use_phases: List[str],
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)

    # split
    if split_json is not None and Path(split_json).exists():
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train, val, test = split["train"], split["val"], split["test"]
        seed_used = split.get("seed", seed)
    else:
        patients = list_patients_from_camus(base_dir)
        if not patients:
            raise RuntimeError(f"No patientXXXX folders found under {base_dir}")
        train, val, test = make_split(patients, seed=seed, ratios=(0.8, 0.1, 0.1))
        seed_used = seed
        save_split_json(out_root / "split.json", seed_used, train, val, test)

    out_hw = (size, size)
    clip_percentiles = (p_lo, p_hi)

    total_samples = 0
    total_patients_with_samples = 0
    total_patients_skipped = 0

    for split_name, plist in [("train", train), ("val", val), ("test", test)]:
        out_split_dir = out_root / split_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        for pname in plist:
            n_saved = process_one_patient(
                base_dir=base_dir,
                patient_name=pname,
                out_split_dir=out_split_dir,
                out_hw=out_hw,
                clip_percentiles=clip_percentiles,
                sdf_clip=sdf_clip,
                use_views=use_views,
                use_phases=use_phases,
            )
            if n_saved > 0:
                total_samples += n_saved
                total_patients_with_samples += 1
            else:
                total_patients_skipped += 1

    print(f"Done. Saved {total_samples} samples "
          f"from {total_patients_with_samples} patients. "
          f"Skipped {total_patients_skipped} patients.")
    print(f"Split saved at: {out_root / 'split.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default='./CAMUSdataset',
                    help="CAMUSdataset root containing patientXXXX folders")
    ap.add_argument("--out_root", type=str, default='./dataset/CAMUS/SDF',
                    help="Output root containing train/val/test + split.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip distance (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="Optional existing split.json to reuse")

    ap.add_argument("--use_views", nargs="+", default=["2CH", "4CH"], choices=["2CH", "4CH"],
                    help="Views to use: 2CH 4CH")
    ap.add_argument("--use_phases", nargs="+", default=["ED", "ES"], choices=["ED", "ES"],
                    help="Phases to use: ED ES")

    args = ap.parse_args()

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=args.size,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
        sdf_clip=args.clip,
        seed=args.seed,
        split_json=Path(args.split_json) if args.split_json else None,
        use_views=list(args.use_views),
        use_phases=list(args.use_phases),
    )


if __name__ == "__main__":
    main()