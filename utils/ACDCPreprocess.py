# ACDC_ED_Extract_Split_SDF.py
#
# One script to:
# 1) Extract ACDC ED 2D slice (best z by GT), pad/crop to fixed size
# 2) Normalize image with percentile-clip min-max to [-1, 1]
# 3) ✅ Convert GT label to one-hot WITHOUT background (C=3, labels 1/2/3)
#    and then to per-channel SDF (C=3)
# 4) Patient-level split 8:1:1 (train/val/test) and save split.json
# 5) Save outputs to:
#    out_root/
#      split.json
#      train/patientXXX/
#         img.npy        (H,W) float32 in [-1,1]
#         onehot.npy     (3,H,W) uint8   (labels 1..3 only; no background channel)
#         sdf.npy        (3,H,W) float32 clipped to [-1,1] (normalized by clip)
#      val/...
#      test/...
#
# Usage:
#   python ACDC_ED_Extract_Split_SDF.py \
#       --base_dir ./ACDCdataset \
#       --out_root ./dataset/ACDC/SDF \
#       --size 256 \
#       --p_lo 1 --p_hi 99 \
#       --clip 10 \
#       --seed 42
#
# Notes:
# - Works for both training/ and testing/ splits inside ACDCdataset.
# - ED frame is chosen as the earlier frame among the two GT frames (by frame number).
# - Best z slice is picked by maximizing non-zero GT pixels (avoid empty slices).
# - ✅ Background (label 0) is NOT stored in onehot/sdf.

from pathlib import Path
import argparse
import json
import re
from typing import Tuple, List

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


# -------------------------
# Utils
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


def _guess_axes_hwz(arr: np.ndarray) -> Tuple[int, int, int]:
    """
    Guess 3D array axes as (H,W,Z): assume two largest dims are H/W and smallest is Z.
    """
    if arr.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {arr.shape}")
    dims = list(arr.shape)
    sorted_axes = sorted(range(3), key=lambda i: dims[i], reverse=True)
    h_axis, w_axis = sorted_axes[0], sorted_axes[1]
    z_axis = sorted_axes[2]
    return h_axis, w_axis, z_axis


def _to_hwz(arr: np.ndarray) -> np.ndarray:
    """
    Reorder 3D array to (H,W,Z).
    """
    h, w, z = _guess_axes_hwz(arr)
    return np.moveaxis(arr, [h, w, z], [0, 1, 2])


def to_onehot_no_bg(gt2d: np.ndarray, fg_labels=(1, 2, 3)) -> np.ndarray:
    """
    ✅ No-background onehot.

    gt2d: (H,W) integer labels in {0,1,2,3}
    fg_labels: which labels to keep as channels (default 1,2,3)

    return: (C,H,W) uint8 one-hot, where:
      channel 0 -> label 1
      channel 1 -> label 2
      channel 2 -> label 3
    background(0) is ignored (all-zero across channels).
    """
    gt2d = gt2d.astype(np.int64, copy=False)
    C = len(fg_labels)
    H, W = gt2d.shape
    onehot = np.zeros((C, H, W), dtype=np.uint8)
    for i, lab in enumerate(fg_labels):
        onehot[i] = (gt2d == int(lab)).astype(np.uint8)
    return onehot


def pick_best_z_by_gt(gt_hwz: np.ndarray) -> int:
    """
    Pick the slice index with the most non-zero pixels in GT (avoid empty slices).
    """
    counts = [np.count_nonzero(gt_hwz[:, :, z]) for z in range(gt_hwz.shape[2])]
    return int(np.argmax(counts))


# -------------------------
# SDF
# -------------------------

def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = False,          # whether scale to [-1,1]
    normalize_eps: float = 1e-8
) -> np.ndarray:
    """
    onehot: (C,H,W) uint8/bool
    returns sdf: (C,H,W) float32
        inside mask  -> negative distance
        outside mask -> positive distance
        boundary -> 0

    If normalize=True:
        output is scaled to [-1,1] by dividing by `clip` (after clipping).
        sdf_out = clip(sdf_raw, -clip, clip) / clip
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
            # channel empty -> treat as far outside
            sdf_c = np.full((H, W), fill_value=clip, dtype=np.float32)

        sdf_c = np.clip(sdf_c, -clip, clip).astype(np.float32)

        if normalize:
            sdf_c = sdf_c / (float(clip) + normalize_eps)

        sdf[c] = sdf_c

    return sdf


# -------------------------
# Split
# -------------------------

def list_patients_from_acdc(base_dir: Path) -> List[str]:
    """
    Collect patient dirs from base_dir/training and base_dir/testing.
    We return unique patient names.
    """
    base_dir = Path(base_dir)
    roots = [base_dir / "training", base_dir / "testing"]
    pats = []
    for r in roots:
        if not r.exists():
            continue
        for d in sorted([p for p in r.rglob("patient*") if p.is_dir()]):
            pats.append(d.name)
    # unique preserve order
    seen = set()
    uniq = []
    for p in pats:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


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
    n_test = n - n_train - n_val

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
# ACDC file parsing
# -------------------------

def _get_acdc_ed_files(case_dir: Path):
    """
    Find ED image and ED gt based on *_frameXX_gt.nii.gz naming.
    ACDC typically has two GT frames: ED and ES. Pick the earlier by frame number.
    Returns (ed_img_path, ed_gt_path, ed_frame_num) or (None,None,None)
    """
    files = list(case_dir.iterdir())
    gt_files = [p for p in files if re.match(r".*_frame\d{2}_gt\.nii\.gz$", p.name)]
    if len(gt_files) != 2:
        return None, None, None

    frame_info = []
    for gt_path in gt_files:
        m = re.search(r"(.*_frame\d{2})_gt", gt_path.name)
        if not m:
            continue
        frame_prefix = m.group(1)
        img_path = case_dir / (frame_prefix + ".nii.gz")
        if not img_path.exists():
            continue
        frame_num = int(re.search(r"frame(\d{2})", gt_path.name).group(1))
        frame_info.append((frame_num, img_path, gt_path))

    if len(frame_info) != 2:
        return None, None, None

    frame_info.sort(key=lambda x: x[0])
    ed_frame_num, ed_img_path, ed_gt_path = frame_info[0]
    return ed_img_path, ed_gt_path, ed_frame_num


def _find_case_dir(base_dir: Path, patient_name: str) -> Path | None:
    """
    Locate patient folder under base_dir/training or base_dir/testing.
    """
    base_dir = Path(base_dir)
    for root in [base_dir / "training", base_dir / "testing"]:
        if not root.exists():
            continue
        cand = root / patient_name
        if cand.exists() and cand.is_dir():
            return cand
        hits = list(root.rglob(patient_name))
        hits = [h for h in hits if h.is_dir() and h.name == patient_name]
        if hits:
            return hits[0]
    return None


# -------------------------
# Main pipeline
# -------------------------

def process_one_patient(
    base_dir: Path,
    patient_name: str,
    out_patient_dir: Path,
    out_hw: Tuple[int, int],
    clip_percentiles: Tuple[float, float],
    sdf_clip: float,
) -> bool:
    """
    Extract ED slice, normalize to [-1,1], onehot GT (no bg), SDF (no bg), then save.
    Returns True if saved, False if skipped.
    """
    case_dir = _find_case_dir(base_dir, patient_name)
    if case_dir is None:
        print(f"[WARN] Missing patient dir for {patient_name}")
        return False

    ed_img_path, ed_gt_path, ed_frame_num = _get_acdc_ed_files(case_dir)
    if ed_img_path is None:
        return False

    ed_img = nib.load(str(ed_img_path)).get_fdata()
    ed_gt = nib.load(str(ed_gt_path)).get_fdata()

    ed_img_hwz = _to_hwz(ed_img)
    ed_gt_hwz = _to_hwz(ed_gt)

    best_z = pick_best_z_by_gt(ed_gt_hwz)

    img2d = ed_img_hwz[:, :, best_z].astype(np.float32)
    gt2d = ed_gt_hwz[:, :, best_z].astype(np.int16)

    img2d = center_pad_or_crop_hw(img2d, out_hw=out_hw, pad_value=0.0)
    gt2d = center_pad_or_crop_hw(gt2d, out_hw=out_hw, pad_value=0)

    # image [-1,1]
    img_norm = minmax01_single(img2d, clip_percentiles=clip_percentiles)

    # ✅ onehot (no background) + sdf (no background)
    gt_onehot = to_onehot_no_bg(gt2d, fg_labels=(1, 2, 3))              # (3,H,W) uint8
    gt_sdf = onehot_to_sdf_per_channel(gt_onehot, clip=sdf_clip, normalize=True)  # (3,H,W) float32 in [-1,1]

    _ensure_dirs(out_patient_dir)
    np.save(out_patient_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_patient_dir / "onehot.npy", gt_onehot)
    np.save(out_patient_dir / "sdf.npy", gt_sdf.astype(np.float32))

    return True


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    seed: int,
    split_json: Path | None,
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
        patients = list_patients_from_acdc(base_dir)
        train, val, test = make_split(patients, seed=seed, ratios=(0.8, 0.1, 0.1))
        seed_used = seed
        save_split_json(out_root / "split.json", seed_used, train, val, test)

    out_hw = (size, size)
    clip_percentiles = (p_lo, p_hi)

    total_saved = 0
    total_skipped = 0

    for split_name, plist in [("train", train), ("val", val), ("test", test)]:
        for pname in plist:
            out_pdir = out_root / split_name / pname
            ok = process_one_patient(
                base_dir=base_dir,
                patient_name=pname,
                out_patient_dir=out_pdir,
                out_hw=out_hw,
                clip_percentiles=clip_percentiles,
                sdf_clip=sdf_clip,
            )
            if ok:
                total_saved += 1
            else:
                total_skipped += 1

    print(f"Done. Saved {total_saved} patients. Skipped {total_skipped}.")
    print(f"Split saved at: {out_root / 'split.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="./ACDCdataset", help="ACDC dataset root containing training/ testing/")
    ap.add_argument("--out_root", type=str, default="./dataset/ACDC/SDF", help="Output root containing train/val/test + split.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip distance (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="Optional existing split.json to reuse")
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
    )


if __name__ == "__main__":
    main()