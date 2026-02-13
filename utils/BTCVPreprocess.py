# BTCVPreprocess.py
#
# One script to:
# 1) Extract BTCV 2D slices using "per-organ Top-k union" (avoid missing small organs)
# 2) Center pad/crop to fixed size
# 3) Normalize image with percentile-clip min-max to [-1, 1]
# 4) Convert GT label to one-hot (C=14) and then to per-channel SDF (C=14)
# 5) Patient-level split 8:1:1 (train/val/test) and save split.json
# 6) Save outputs to:
#    out_root/
#      split.json
#      train/imgXXXX_slice000/
#         img.npy        (H,W) float32 in [-1,1]
#         onehot.npy     (14,H,W) uint8
#         sdf.npy        (14,H,W) float32 in [-1,1] (clipped by `clip` then /clip)
#      val/...
#      test/...

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Tuple, List, Optional

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


# -------------------------
# Utils
# -------------------------

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def center_pad_or_crop_hw(x: np.ndarray, out_hw=(256, 256), pad_value=0):
    """Center pad/crop a (H,W) (or (H,W,1)) array to out_hw."""
    if x.ndim == 2:
        x = x[:, :, None]
        squeeze_back = True
    else:
        squeeze_back = False

    H, W, C = x.shape
    out_h, out_w = out_hw

    # pad
    pad_h = max(0, out_h - H)
    pad_w = max(0, out_w - W)
    if pad_h or pad_w:
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
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

    return x[:, :, 0] if squeeze_back else x


def minmax01_single(x: np.ndarray, eps=1e-8, clip_percentiles=(1.0, 99.0)) -> np.ndarray:
    """Percentile-clip + min-max normalize to [-1,1]."""
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


def _guess_axes_hwz(arr: np.ndarray) -> Tuple[int, int, int]:
    """Guess axes mapping to (H,W,Z): two largest dims -> H/W, smallest -> Z."""
    if arr.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {arr.shape}")
    dims = list(arr.shape)
    axes = sorted(range(3), key=lambda i: dims[i], reverse=True)
    return axes[0], axes[1], axes[2]


def _to_hwz(arr: np.ndarray) -> np.ndarray:
    """Reorder 3D array to (H,W,Z)."""
    h, w, z = _guess_axes_hwz(arr)
    return np.moveaxis(arr, [h, w, z], [0, 1, 2])


def to_onehot_btcv(gt2d: np.ndarray, num_classes: int = 14) -> np.ndarray:
    """gt2d: (H,W) labels in [0..C-1] -> (C,H,W) one-hot uint8."""
    gt2d = gt2d.astype(np.int64, copy=False)
    gt2d = np.clip(gt2d, 0, num_classes - 1)
    onehot = np.eye(num_classes, dtype=np.uint8)[gt2d]   # (H,W,C)
    return np.transpose(onehot, (2, 0, 1))               # (C,H,W)


# -------------------------
# Slice selection (方案 A)
# -------------------------

def pick_z_slices_by_class_union(
    gt_hwz: np.ndarray,
    num_classes: int = 14,
    k_per_class: int = 1,
    max_slices: Optional[int] = None,
) -> List[int]:
    """
    方案A：对每个器官(1..C-1)各自选像素最多的 Top-k 切片，再取并集（跳过背景0）。
    - 若某类完全不存在：跳过
    - 若并集为空：回退到“总GT最多”的那一张
    - 若 max_slices 给定且并集过大：按“总GT像素”从大到小截断
    """
    Z = gt_hwz.shape[2]
    selected: set[int] = set()

    for c in range(1, num_classes):
        per_z = np.count_nonzero(gt_hwz == c, axis=(0, 1))  # (Z,)
        if per_z.max() == 0:
            continue
        top = np.argsort(per_z)[::-1][:k_per_class]
        for z in top:
            if per_z[z] > 0:
                selected.add(int(z))

    if not selected:
        total = np.count_nonzero(gt_hwz, axis=(0, 1))
        selected = {int(np.argsort(total)[::-1][0])}

    idx = sorted(selected)

    if max_slices is not None and len(idx) > max_slices:
        total = np.count_nonzero(gt_hwz, axis=(0, 1))
        idx = sorted(idx, key=lambda z: total[z], reverse=True)[:max_slices]
        idx = sorted(idx)

    return idx


# -------------------------
# SDF
# -------------------------

def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = True,
    normalize_eps: float = 1e-8,
) -> np.ndarray:
    """
    onehot: (C,H,W) -> sdf: (C,H,W) float32
    sdf = dist_out - dist_in (mask内为负，外为正，边界≈0)
    clip到[-clip,clip]；若 normalize=True，再除以 clip -> [-1,1]
    """
    if onehot.ndim != 3:
        raise ValueError(f"Expected onehot shape (C,H,W), got {onehot.shape}")
    if clip <= 0:
        raise ValueError(f"clip must be > 0, got {clip}")

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
# Split
# -------------------------

def list_images_from_btcv(base_dir: Path) -> List[str]:
    """Collect unique image names (without .nii.gz) from imagesTr + imagesTs."""
    base_dir = Path(base_dir)
    roots = [base_dir / "imagesTr", base_dir / "imagesTs"]
    imgs: List[str] = []

    for r in roots:
        if not r.exists():
            continue
        for p in sorted(r.glob("*.nii.gz")):
            imgs.append(p.name[:-7])  # remove ".nii.gz"

    seen = set()
    uniq = []
    for name in imgs:
        if name not in seen:
            uniq.append(name)
            seen.add(name)
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

    train = imgs[:n_train]
    val = imgs[n_train:n_train + n_val]
    test = imgs[n_train + n_val:]
    return train, val, test


def save_split_json(out_path: Path, seed: int, train, val, test):
    obj = {
        "seed": seed,
        "counts": {"total": len(train) + len(val) + len(test), "train": len(train), "val": len(val), "test": len(test)},
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
    Find image and GT files based on the original naming convention.
    image: imagesTr/{image_name}.nii.gz (fallback to imagesTs)
    gt: labelsTr/label{image_name[3:]}.nii.gz (may be missing for test)
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
    k_per_class: int,
    max_slices: Optional[int],
) -> int:
    """Extract selected slices, normalize, onehot+SDF, then save. Return #slices saved."""
    img_path, gt_path = _get_btcv_files(base_dir, image_name)
    if img_path is None:
        print(f"[WARN] Missing image file for {image_name}")
        return 0

    img = nib.load(str(img_path)).get_fdata()
    img_hwz = _to_hwz(img).astype(np.float32, copy=False)

    if gt_path is not None:
        gt = nib.load(str(gt_path)).get_fdata()
        gt_hwz = _to_hwz(gt).astype(np.int16, copy=False)
        slice_indices = pick_z_slices_by_class_union(
            gt_hwz, num_classes=14, k_per_class=k_per_class, max_slices=max_slices
        )
    else:
        gt_hwz = None
        slice_indices = [img_hwz.shape[2] // 2]

    saved = 0
    for i, z in enumerate(slice_indices):
        slice_out_dir = out_root / split_name / f"{image_name}_slice{i:03d}"
        _ensure_dirs(slice_out_dir)

        # image
        img2d = img_hwz[:, :, z]
        img2d = center_pad_or_crop_hw(img2d, out_hw=out_hw, pad_value=0.0)
        img_norm = minmax01_single(img2d, clip_percentiles=clip_percentiles)
        np.save(slice_out_dir / "img.npy", img_norm.astype(np.float32))

        # gt + sdf
        if gt_hwz is not None:
            gt2d = gt_hwz[:, :, z]
            gt2d = center_pad_or_crop_hw(gt2d, out_hw=out_hw, pad_value=0)
            onehot = to_onehot_btcv(gt2d, num_classes=14)
            sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip, normalize=True)
            np.save(slice_out_dir / "onehot.npy", onehot)
            np.save(slice_out_dir / "sdf.npy", sdf.astype(np.float32))

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
    split_json: Optional[Path],
    k_per_class: int,
    max_slices: Optional[int],
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
    total_vols = len(train) + len(val) + len(test)

    for split_name, ilist in [("train", train), ("val", val), ("test", test)]:
        for name in ilist:
            n = process_one_image(
                base_dir=base_dir,
                image_name=name,
                out_root=out_root,
                split_name=split_name,
                out_hw=out_hw,
                clip_percentiles=clip_percentiles,
                sdf_clip=sdf_clip,
                k_per_class=k_per_class,
                max_slices=max_slices,
            )
            if n > 0:
                total_saved += n
            else:
                total_skipped += 1

    print(f"Done. Saved {total_saved} slices from {total_vols} volumes. Skipped {total_skipped} volumes.")
    print(f"Split saved at: {out_root / 'split.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="./BTCVdataset",
                    help="BTCV dataset root containing imagesTr/ imagesTs/ labelsTr/")
    ap.add_argument("--out_root", type=str, default="./dataset/BTCV/SDF",
                    help="Output root containing train/val/test + split.json")

    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)

    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip distance (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="Optional existing split.json to reuse")

    # 方案A参数
    ap.add_argument("--k_per_class", type=int, default=1,
                    help="Top-k slices per organ class (excluding background). default=1")
    ap.add_argument("--max_slices", type=int, default=-1,
                    help="Cap total selected slices per volume. -1 means no cap")

    args = ap.parse_args()
    max_slices = None if args.max_slices < 0 else int(args.max_slices)

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=args.size,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
        sdf_clip=args.clip,
        seed=args.seed,
        split_json=Path(args.split_json) if args.split_json else None,
        k_per_class=int(args.k_per_class),
        max_slices=max_slices,
    )


if __name__ == "__main__":
    main()