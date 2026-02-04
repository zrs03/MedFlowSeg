# utils/CAMUSPreprocess_PIL.py
#
# DDTI-style preprocessing for CAMUS:
# - Each (patient + view + phase) => one sample folder
# - Resize (reshape) to 256x256 using PIL:
#     image: bilinear
#     mask : nearest
# - Normalize image to [-1,1] using percentile clip
# - GT labels are {0,1,2,3} -> onehot (4,H,W) -> SDF (4,H,W)
# - Split on patients (8:1:1) then expand into samples inside each split
#
# Output:
#   out_root/
#     split.json
#     train/patient0001_2CH_ED/{img.npy, onehot.npy, sdf.npy}
#     ...
#
# Usage:
#   python utils/CAMUSPreprocess_PIL.py \
#       --base_dir ./CAMUSdataset \
#       --out_root ./dataset/CAMUS/SDF \
#       --size 256 \
#       --p_lo 1 --p_hi 99 \
#       --clip 10 \
#       --seed 42 \
#       --use_views 2CH 4CH \
#       --use_phases ED ES

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import nibabel as nib
from PIL import Image
from scipy.ndimage import distance_transform_edt


# ============================================================
# Utils
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_hw_if_needed(arr: np.ndarray) -> np.ndarray:
    """
    CAMUS ED/ES frames are usually 2D, but nibabel may return (H,W) or (H,W,1).
    Convert to (H,W).
    """
    if arr.ndim == 2:
        return arr
    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed
    raise ValueError(f"Unexpected CAMUS frame shape: {arr.shape}")


def resize_hw(x: np.ndarray, out_hw=(256, 256), is_mask: bool = False) -> np.ndarray:
    """
    Resize to (out_h, out_w) using PIL:
      - bilinear for image (float grayscale)
      - nearest for mask/labels (uint8)
    x: (H,W) or (H,W,1)
    return: (out_h,out_w)
    """
    if x.ndim == 3 and x.shape[-1] == 1:
        x2 = x[:, :, 0]
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError(f"Expected (H,W) or (H,W,1), got {x.shape}")

    out_h, out_w = out_hw

    if is_mask:
        # labels must be integer
        im = Image.fromarray(x2.astype(np.uint8, copy=False), mode="L")
        im = im.resize((out_w, out_h), resample=Image.NEAREST)
        y = np.asarray(im, dtype=np.uint8)
        return y
    else:
        # float grayscale
        im = Image.fromarray(x2.astype(np.float32, copy=False), mode="F")
        im = im.resize((out_w, out_h), resample=Image.BILINEAR)
        y = np.asarray(im, dtype=np.float32)
        return y


def minmax01_single(x: np.ndarray, eps=1e-8, clip_percentiles=(1.0, 99.0)) -> np.ndarray:
    """
    Percentile-clip + min-max normalization to [-1,1].
    x: (H,W) float
    """
    x = x.astype(np.float32, copy=False)
    p_lo, p_hi = clip_percentiles
    lo, hi = np.percentile(x, p_lo), np.percentile(x, p_hi)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x01 = (x - lo) / (hi - lo + eps)
    return (x01 * 2.0 - 1.0).astype(np.float32, copy=False)


def to_onehot_4(mask2d: np.ndarray) -> np.ndarray:
    """
    mask2d: (H,W) labels in {0,1,2,3}
    return: (4,H,W) uint8 onehot
      ch0 = label0 (bg)
      ch1 = label1
      ch2 = label2
      ch3 = label3
    """
    m = mask2d.astype(np.int64, copy=False)
    m = np.clip(m, 0, 3)
    onehot = np.eye(4, dtype=np.uint8)[m]      # (H,W,4)
    onehot = np.transpose(onehot, (2, 0, 1))   # (4,H,W)
    return onehot


def onehot_to_sdf_per_channel(onehot: np.ndarray, clip: float = 10.0, normalize: bool = True, eps: float = 1e-8):
    """
    onehot: (C,H,W)
    sdf: (C,H,W) float32, inside negative, outside positive
    clip to [-clip,clip], optionally /clip -> [-1,1]
    """
    if onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {onehot.shape}")
    if normalize and clip <= 0:
        raise ValueError("clip must be >0 when normalize=True")

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
            sdf_c = sdf_c / (float(clip) + eps)

        sdf[c] = sdf_c

    return sdf


# ============================================================
# Split (patient-level)
# ============================================================
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
    ids = patients.copy()
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
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


# ============================================================
# CAMUS item discovery
# ============================================================
_ALLOWED_VIEWS = {"2CH", "4CH"}
_ALLOWED_PHASES = {"ED", "ES"}


def _paths_for_item(patient_dir: Path, patient_name: str, view: str, phase: str) -> Tuple[Path, Path]:
    img_path = patient_dir / f"{patient_name}_{view}_{phase}.nii.gz"
    gt_path = patient_dir / f"{patient_name}_{view}_{phase}_gt.nii.gz"
    return img_path, gt_path


# ============================================================
# Processing: each sample_id has its own folder
# ============================================================
def process_one_sample(
    base_dir: Path,
    sample_id: str,     # "patient0001_2CH_ED"
    split_name: str,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    verbose: bool,
) -> bool:
    """
    Output:
      out_root/split/sample_id/
        img.npy
        onehot.npy
        sdf.npy
    """
    # parse sample_id
    m = re.match(r"^(patient\d{4})_(2CH|4CH)_(ED|ES)$", sample_id)
    if not m:
        if verbose:
            print(f"[Skip] bad sample_id: {sample_id}")
        return False

    patient_name, view, phase = m.group(1), m.group(2), m.group(3)
    patient_dir = base_dir / patient_name
    if not patient_dir.exists():
        if verbose:
            print(f"[Skip] missing patient dir: {patient_name}")
        return False

    img_path, gt_path = _paths_for_item(patient_dir, patient_name, view, phase)
    if not img_path.exists() or not gt_path.exists():
        if verbose:
            print(f"[Skip] missing pair for {sample_id}")
        return False

    # load nii
    img = nib.load(str(img_path)).get_fdata()
    gt = nib.load(str(gt_path)).get_fdata()

    img2d = _to_hw_if_needed(img).astype(np.float32)
    gt2d = _to_hw_if_needed(gt).astype(np.int16)

    # ========================================================
    # Resize to fixed size (same style as your DDTI code)
    # ========================================================
    out_hw = (size, size)
    img2d = resize_hw(img2d, out_hw=out_hw, is_mask=False)  # float bilinear
    gt2d = resize_hw(gt2d, out_hw=out_hw, is_mask=True)     # uint8 nearest (IMPORTANT!)

    # ensure labels are 0..3
    gt2d = np.clip(gt2d.astype(np.uint8, copy=False), 0, 3)

    # normalize image [-1,1]
    img_norm = minmax01_single(img2d, clip_percentiles=(p_lo, p_hi))

    # onehot + sdf
    onehot = to_onehot_4(gt2d)  # (4,H,W)
    sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip, normalize=sdf_normalize)  # (4,H,W)

    # save
    out_dir = out_root / split_name / sample_id
    _ensure_dirs(out_dir)
    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot.astype(np.uint8))
    np.save(out_dir / "sdf.npy", sdf.astype(np.float32))

    if verbose:
        print(f"[OK] {sample_id} -> {out_dir}")

    return True


def build_sample_ids_for_patients(patients: List[str], use_views: List[str], use_phases: List[str]) -> List[str]:
    sample_ids = []
    for p in patients:
        for v in use_views:
            for ph in use_phases:
                sample_ids.append(f"{p}_{v}_{ph}")
    return sample_ids


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    seed: int,
    split_json: Path | None,
    use_views: List[str],
    use_phases: List[str],
    verbose: bool,
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    # split on PATIENTS, then expand to samples (patient_view_phase)
    if split_json is not None and split_json.exists():
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_p, val_p, test_p = split["train"], split["val"], split["test"]
        if verbose:
            print(f"[Info] Reusing split: {split_json}")
    else:
        patients = list_patients_from_camus(base_dir)
        if not patients:
            raise RuntimeError(f"No patientXXXX folders found under {base_dir}")
        train_p, val_p, test_p = make_split(patients, seed=seed, ratios=(0.8, 0.1, 0.1))
        save_split_json(out_root / "split.json", seed, train_p, val_p, test_p)
        if verbose:
            print(f"[Info] Created split.json at: {out_root / 'split.json'}")

    # expand
    train_ids = build_sample_ids_for_patients(train_p, use_views, use_phases)
    val_ids = build_sample_ids_for_patients(val_p, use_views, use_phases)
    test_ids = build_sample_ids_for_patients(test_p, use_views, use_phases)

    total_ok = 0
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for sid in ids:
            ok = process_one_sample(
                base_dir=base_dir,
                sample_id=sid,
                split_name=split_name,
                out_root=out_root,
                size=size,
                p_lo=p_lo,
                p_hi=p_hi,
                sdf_clip=sdf_clip,
                sdf_normalize=sdf_normalize,
                verbose=verbose,
            )
            total_ok += int(ok)

    if verbose:
        print("\n====================")
        print(f"Done. Samples processed: {total_ok}")
        print(f"Split json: {out_root / 'split.json'}")
        print("====================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="./CAMUSdataset", help="包含 patientXXXX/ 的文件夹")
    ap.add_argument("--out_root", type=str, default="./dataset/CAMUS/SDF", help="输出根目录：train/val/test + split.json")
    ap.add_argument("--size", type=int, default=256, help="resize 到固定尺寸 size x size（直接 reshape/resize）")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip 距离")
    ap.add_argument("--sdf_no_norm", action="store_true", help="不把 SDF 归一化到 [-1,1]（只 clip）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="可选：复用已有 split.json（patient-level）")
    ap.add_argument("--use_views", nargs="+", default=["2CH", "4CH"], choices=["2CH", "4CH"])
    ap.add_argument("--use_phases", nargs="+", default=["ED", "ES"], choices=["ED", "ES"])
    ap.add_argument("--quiet", action="store_true", help="不打印日志")
    args = ap.parse_args()

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        sdf_clip=float(args.clip),
        sdf_normalize=(not args.sdf_no_norm),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        use_views=list(args.use_views),
        use_phases=list(args.use_phases),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()