# BTCVPreprocess.py
#
# One script to:
# 1) Extract BTCV 2D slices (best z by GT), pad/crop to fixed size
# 2) Normalize image with percentile-clip min-max to [-1, 1]
# 3) Convert GT label to one-hot (C=14) and then to per-channel SDF (C=14)
# 4) Patient-level split 8:1:1 (train/val/test) and save split.json
# 5) Save outputs to:
#    out_root/
#      split.json
#      train/imgXXXX/
#         img.npy        (H,W) float32 in [-1,1]
#         onehot.npy     (14,H,W) uint8
#         sdf.npy        (14,H,W) float32 clipped to [-clip,clip]
#      val/...
#      test/...

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


def to_onehot_btcv(gt2d: np.ndarray, num_classes: int = 14) -> np.ndarray:
    """
    gt2d: (H,W) integer labels in {0,1,2,...,13}
    return: (C,H,W) uint8 one-hot
    """
    gt2d = gt2d.astype(np.int64, copy=False)
    gt2d = np.clip(gt2d, 0, num_classes - 1)
    onehot = np.eye(num_classes, dtype=np.uint8)[gt2d]  # (H,W,C)
    onehot = np.transpose(onehot, (2, 0, 1))           # (C,H,W)
    return onehot


def pick_top_z_slices_by_gt(gt_hwz: np.ndarray, num_slices: int = 5) -> List[int]:
    """
    Pick the top N slice indices with the most non-zero pixels in GT (avoid empty slices).
    
    Args:
        gt_hwz: (H,W,Z) GT volume
        num_slices: number of top slices to return
        
    Returns:
        List of top slice indices sorted by number of non-zero pixels (descending)
    """
    counts = [np.count_nonzero(gt_hwz[:, :, z]) for z in range(gt_hwz.shape[2])]
    # Get indices sorted by counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    # Return top N slices, but not more than available
    return [int(idx) for idx in sorted_indices[:num_slices]]


# -------------------------
# SDF
# -------------------------

def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = False,          # 新增：是否缩放到 [-1,1]
    normalize_eps: float = 1e-8       # 新增：防止 clip=0
) -> np.ndarray:
    """
    onehot: (C,H,W) uint8/bool (C=14)
    returns sdf: (C,H,W) float32
        inside mask  -> negative distance
        outside mask -> positive distance
        boundary -> 0

    If normalize=True:
        output is scaled to [-1,1] by dividing by `clip` (after clipping).
        i.e., sdf_out = clip(sdf_raw, -clip, clip) / clip
    """
    if onehot.ndim != 3:
        raise ValueError(f"Expected onehot shape (C,H,W), got {onehot.shape}")

    if clip is None:
        raise ValueError("clip cannot be None if normalize=True (and recommended to be set always).")
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

        # clip to [-clip, clip]
        sdf_c = np.clip(sdf_c, -clip, clip).astype(np.float32)

        # normalize to [-1,1] if requested
        if normalize:
            sdf_c = sdf_c / (float(clip) + normalize_eps)

        sdf[c] = sdf_c

    return sdf


# -------------------------
# Split
# -------------------------

def list_images_from_btcv(base_dir: Path) -> List[str]:
    """
    Collect image files from base_dir/imagesTr and base_dir/imagesTs.
    We return unique image names without extension.
    """
    base_dir = Path(base_dir)
    roots = [base_dir / "imagesTr", base_dir / "imagesTs"]
    imgs = []
    for r in roots:
        if not r.exists():
            continue
        for d in sorted([p for p in r.iterdir() if p.is_file() and p.suffix == '.gz']):
            # 处理 .nii.gz 双扩展名
            if d.suffixes == ['.nii', '.gz']:
                # 移除两个扩展名
                img_name = d.name[:-7]  # 移除 .nii.gz
            else:
                img_name = d.stem
            imgs.append(img_name)
    # unique preserve order
    seen = set()
    uniq = []
    for p in imgs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def make_split(images: List[str], seed=42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    imgs = images.copy()
    rng.shuffle(imgs)

    n = len(imgs)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train = imgs[:n_train]
    val = imgs[n_train:n_train + n_val]
    test = imgs[n_train + n_val:]
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
# BTCV file parsing
# -------------------------

def _get_btcv_files(base_dir: Path, image_name: str):
    """
    Find image and gt files based on naming convention.
    Returns (img_path, gt_path) or (None, None)
    """
    img_path = base_dir / "imagesTr" / f"{image_name}.nii.gz"
    if not img_path.exists():
        img_path = base_dir / "imagesTs" / f"{image_name}.nii.gz"
        if not img_path.exists():
            return None, None
    
    gt_path = base_dir / "labelsTr" / f"label{image_name[3:]}.nii.gz"
    if not gt_path.exists():
        return img_path, None
    
    return img_path, gt_path


# -------------------------
# Main pipeline
# -------------------------

def process_one_image(
    base_dir: Path,
    image_name: str,
    out_root: Path,
    split_name: str,
    out_hw: Tuple[int, int],
    clip_percentiles: Tuple[float, float],
    sdf_clip: float,
    num_slices: int = 5,
) -> int:
    """
    Extract multiple slices, normalize to [-1,1], onehot GT, SDF, then save.
    Returns number of slices saved, 0 if skipped.
    """
    img_path, gt_path = _get_btcv_files(base_dir, image_name)
    if img_path is None:
        print(f"[WARN] Missing image file for {image_name}")
        return 0

    # Load image
    img = nib.load(str(img_path)).get_fdata()
    img_hwz = _to_hwz(img)

    # Determine which slices to process
    if gt_path is not None:
        gt = nib.load(str(gt_path)).get_fdata()
        gt_hwz = _to_hwz(gt)
        slice_indices = pick_top_z_slices_by_gt(gt_hwz, num_slices=num_slices)
    else:
        # If no GT, pick middle slice only
        slice_indices = [img_hwz.shape[2] // 2]
        gt_hwz = None

    slices_saved = 0
    for i, slice_idx in enumerate(slice_indices):
        # Create output directory for this slice
        slice_suffix = f"_slice{i:03d}"
        slice_out_dir = out_root / split_name / f"{image_name}{slice_suffix}"
        
        # Process image slice
        img2d = img_hwz[:, :, slice_idx].astype(np.float32)
        img2d = center_pad_or_crop_hw(img2d, out_hw=out_hw, pad_value=0.0)
        img_norm = minmax01_single(img2d, clip_percentiles=clip_percentiles)

        _ensure_dirs(slice_out_dir)
        np.save(slice_out_dir / "img.npy", img_norm.astype(np.float32))

        # Process GT if available
        if gt_path is not None:
            gt2d = gt_hwz[:, :, slice_idx].astype(np.int16)
            gt2d = center_pad_or_crop_hw(gt2d, out_hw=out_hw, pad_value=0)
            
            # onehot + sdf
            gt_onehot = to_onehot_btcv(gt2d, num_classes=14)              # (14,H,W) uint8
            gt_sdf = onehot_to_sdf_per_channel(gt_onehot, clip=sdf_clip, normalize=True)  # (14,H,W) float32
            
            np.save(slice_out_dir / "onehot.npy", gt_onehot)
            np.save(slice_out_dir / "sdf.npy", gt_sdf.astype(np.float32))
        
        slices_saved += 1

    return slices_saved


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    seed: int,
    split_json: Path | None,
    num_slices: int = 5,
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
        images = list_images_from_btcv(base_dir)
        train, val, test = make_split(images, seed=seed, ratios=(0.8, 0.1, 0.1))
        seed_used = seed
        save_split_json(out_root / "split.json", seed_used, train, val, test)

    out_hw = (size, size)
    clip_percentiles = (p_lo, p_hi)

    total_saved = 0
    total_skipped = 0

    for split_name, ilist in [("train", train), ("val", val), ("test", test)]:
        for iname in ilist:
            slices_saved = process_one_image(
                base_dir=base_dir,
                image_name=iname,
                out_root=out_root,
                split_name=split_name,
                out_hw=out_hw,
                clip_percentiles=clip_percentiles,
                sdf_clip=sdf_clip,
                num_slices=num_slices,
            )
            if slices_saved > 0:
                total_saved += slices_saved
            else:
                total_skipped += 1

    print(f"Done. Saved {total_saved} slices from {len(train) + len(val) + len(test)} volumes. Skipped {total_skipped} volumes.")
    print(f"Split saved at: {out_root / 'split.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="./BTCVdataset", help="BTCV dataset root containing imagesTr/ imagesTs/ labelsTr/")
    ap.add_argument("--out_root", type=str, default="./dataset/BTCV/SDF", help="Output root containing train/val/test + split.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip distance (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="Optional existing split.json to reuse")
    ap.add_argument("--num_slices", type=int, default=5, help="Number of top slices to extract per volume (default 5)")
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
        num_slices=args.num_slices,
    )


if __name__ == "__main__":
    main()
