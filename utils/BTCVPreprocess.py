from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Tuple, List, Optional

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, zoom


# =========================
# Utils
# =========================

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# =========================
# Resize
# =========================

def resize_hw(x: np.ndarray, out_hw=(256, 256), order=1):
    if x.ndim == 2:
        H, W = x.shape
        zoom_factors = (out_hw[0] / H, out_hw[1] / W)
        return zoom(x, zoom_factors, order=order)

    elif x.ndim == 3:
        H, W, C = x.shape
        zoom_factors = (out_hw[0] / H, out_hw[1] / W, 1)
        return zoom(x, zoom_factors, order=order)

    else:
        raise ValueError("Unsupported ndim for resize")


# =========================
# HU Clipping + Normalize
# =========================

def hu_clip(x: np.ndarray, hu_min: float = -125.0, hu_max: float = 275.0):
    """
    Clip CT intensities to a soft-tissue window, e.g. [-125, 275].
    """
    x = x.astype(np.float32, copy=False)
    return np.clip(x, hu_min, hu_max)


def minmax01_single(x: np.ndarray, eps=1e-8):
    """
    Normalize a clipped slice to [-1, 1].
    """
    x = x.astype(np.float32, copy=False)

    lo = float(x.min())
    hi = float(x.max())

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x01 = (x - lo) / (hi - lo + eps)
    return (x01 * 2.0 - 1.0).astype(np.float32, copy=False)


# =========================
# Axis
# =========================

def _guess_axes_hwz(arr: np.ndarray):
    dims = list(arr.shape)
    axes = sorted(range(3), key=lambda i: dims[i], reverse=True)
    return axes[0], axes[1], axes[2]


def _to_hwz(arr: np.ndarray):
    h, w, z = _guess_axes_hwz(arr)
    return np.moveaxis(arr, [h, w, z], [0, 1, 2])


# =========================
# Onehot (NO BACKGROUND)
# BTCV labels: 0=background, 1..13=foreground organs
# output channels correspond to labels 1..13
# =========================

def to_onehot_btcv_no_bg(gt2d: np.ndarray, num_fg_classes: int = 13):
    """
    gt2d: (H,W), labels in [0..13]
    return: (13,H,W), channels map to labels 1..13
    """
    gt2d = gt2d.astype(np.int64, copy=False)
    H, W = gt2d.shape
    onehot = np.zeros((num_fg_classes, H, W), dtype=np.uint8)

    for c in range(1, num_fg_classes + 1):
        onehot[c - 1] = (gt2d == c).astype(np.uint8)

    return onehot


# =========================
# SDF
# =========================

def onehot_to_sdf_per_channel(onehot: np.ndarray, clip: float = 10.0):
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

        sdf_c = np.clip(sdf_c, -clip, clip)
        sdf_c = sdf_c / clip
        sdf[c] = sdf_c

    return sdf


# =========================
# Split
# =========================

def list_images_from_btcv(base_dir: Path):
    roots = [base_dir / "imagesTr", base_dir / "imagesTs"]
    imgs = []

    for r in roots:
        if not r.exists():
            continue
        for p in sorted(r.glob("*.nii*")):
            name = p.name
            if name.endswith(".nii.gz"):
                imgs.append(name[:-7])
            elif name.endswith(".nii"):
                imgs.append(name[:-4])
            else:
                imgs.append(p.stem)

    return list(dict.fromkeys(imgs))


def make_split(images, seed=42, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    imgs = images.copy()
    rng.shuffle(imgs)

    n = len(imgs)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))

    train = imgs[:n_train]
    val = imgs[n_train:n_train + n_val]
    test = imgs[n_train + n_val:]
    return train, val, test


def save_split_json(out_path, seed, train, val, test, hu_min, hu_max):
    obj = {
        "seed": seed,
        "num_fg_classes": 13,
        "channels_map_to_labels": list(range(1, 14)),
        "hu_clip_range": [hu_min, hu_max],
        "train": train,
        "val": val,
        "test": test,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# Main Processing
# =========================

def process_one_image(
    base_dir,
    image_name,
    out_root,
    split_name,
    out_hw,
    hu_min,
    hu_max,
    sdf_clip,
):
    def find_file(folder, name):
        for ext in [".nii", ".nii.gz"]:
            p = folder / f"{name}{ext}"
            if p.exists():
                return p
        return None

    img_path = find_file(base_dir / "imagesTr", image_name)
    if img_path is None:
        img_path = find_file(base_dir / "imagesTs", image_name)

    if img_path is None:
        print("Missing image:", image_name)
        return 0

    gt_path = find_file(base_dir / "labelsTr", "label" + image_name[3:])
    gt_exists = gt_path is not None

    img = nib.load(str(img_path)).get_fdata()
    img_hwz = _to_hwz(img).astype(np.float32)

    if gt_exists:
        gt = nib.load(str(gt_path)).get_fdata()
        gt_hwz = _to_hwz(gt).astype(np.int16)
    else:
        gt_hwz = None

    saved = 0

    for i in range(img_hwz.shape[2]):
        img2d = img_hwz[:, :, i]

        # 1) HU clipping
        img2d = hu_clip(img2d, hu_min=hu_min, hu_max=hu_max)

        # 2) Resize
        img2d = resize_hw(img2d, out_hw, order=1)

        # 3) Normalize to [-1, 1]
        img_norm = minmax01_single(img2d)

        if gt_hwz is None:
            continue

        gt2d = gt_hwz[:, :, i]

        # skip pure-background slices
        if np.max(gt2d) == 0:
            continue

        gt2d = resize_hw(gt2d, out_hw, order=0)
        gt2d = np.rint(gt2d).astype(np.int16)

        # no background channel: output shape (13,H,W)
        onehot = to_onehot_btcv_no_bg(gt2d, num_fg_classes=13)
        sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip)

        slice_out_dir = out_root / split_name / f"{image_name}_slice{i:03d}"
        _ensure_dirs(slice_out_dir)

        np.save(slice_out_dir / "img.npy", img_norm.astype(np.float32))
        np.save(slice_out_dir / "onehot.npy", onehot.astype(np.uint8))
        np.save(slice_out_dir / "sdf.npy", sdf.astype(np.float32))

        saved += 1

    return saved


# =========================
# Run
# =========================

def run(base_dir, out_root, size, hu_min, hu_max, sdf_clip, seed):
    images = list_images_from_btcv(base_dir)

    if len(images) == 0:
        print("No NII files found.")
        return

    train, val, test = make_split(images, seed=seed)
    save_split_json(out_root / "split.json", seed, train, val, test, hu_min, hu_max)

    out_hw = (size, size)
    total_saved = 0

    for split_name, ilist in [("train", train), ("val", val), ("test", test)]:
        for name in ilist:
            total_saved += process_one_image(
                base_dir,
                name,
                out_root,
                split_name,
                out_hw,
                hu_min,
                hu_max,
                sdf_clip,
            )

    print(f"Done. Saved {total_saved} foreground slices.")
    print(f"Applied HU clipping to [{hu_min}, {hu_max}].")
    print("Saved onehot/sdf WITHOUT background channel.")
    print("Channel 0..12 correspond to labels 1..13.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="./BTCVdataset")
    ap.add_argument("--out_root", type=str, default="./dataset/BTCV/SDF")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--hu_min", type=float, default=-125.0)
    ap.add_argument("--hu_max", type=float, default=275.0)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=args.size,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        sdf_clip=args.clip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()