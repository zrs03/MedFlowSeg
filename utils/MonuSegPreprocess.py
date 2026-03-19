# utils/MonuSegPreprocess.py
#
# MonuSeg-style preprocessing for histopathology nuclei segmentation dataset:
# - base_dir/
#     kmms_training/
#         images/
#         masks/
#     kmms_test/
#         images/
#         masks/
#
# Supports image/mask files in png / tif / tiff.
# Each image + mask pair can generate multiple random 128x128 crops.
#
# Processing:
# - collect kmms_training + kmms_test together
# - pair image/mask by normalized filename
# - print all samples whose image/mask spatial sizes are different
# - save mismatch report to out_root/mismatch_report.txt
# - split on whole images first (8:1:1), then crop inside each split
# - FIRST resize both image and mask to 500x500
# - THEN random crop to size x size (default 128x128)
# - image: RGB kept as 3-channel, percentile clip normalize to [-1,1] per channel
# - mask : binarize to {0,1}
# - onehot WITHOUT background: (1,H,W)
#
# Output:
#   out_root/
#     split.json
#     mismatch_report.txt
#     train/<sample_id>/{img.npy, onehot.npy}
#     val/...
#     test/...
#
# Usage example:
#   python utils/MonuSegPreprocess.py \
#       --base_dir ./MonuSegdataset \
#       --out_root ./dataset/MonuSeg/ONEHOT \
#       --resize_to 500 \
#       --size 128 \
#       --crops_per_train 8 \
#       --crops_per_val 2 \
#       --crops_per_test 2 \
#       --p_lo 1 --p_hi 99 \
#       --seed 42

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


_VALID_EXTS = {".png", ".tif", ".tiff"}


# ============================================================
# Utils
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize_name(s: str) -> str:
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    return s


def normalize_pair_key(name: str) -> str:
    """
    Robust key for pairing image/mask names.
    """
    s = name.lower().strip()

    # remove common mask suffixes
    s = re.sub(r"(_mask|-mask| mask)$", "", s)
    s = re.sub(r"(_label|-label| label)$", "", s)
    s = re.sub(r"(_labels|-labels| labels)$", "", s)
    s = re.sub(r"(_annotation|-annotation| annotation)$", "", s)

    # unify symbols
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ============================================================
# Resize / normalize
# ============================================================
def resize_hw_image_rgb(x: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    """
    Resize RGB image to (out_h, out_w) using bilinear.
    x: (H,W,3)
    """
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {x.shape}")

    out_h, out_w = out_hw
    im = Image.fromarray(x.astype(np.uint8, copy=False), mode="RGB")
    im = im.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def resize_hw_mask(x: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    """
    Resize mask to (out_h, out_w) using nearest.
    x: (H,W)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")

    out_h, out_w = out_hw
    im = Image.fromarray(x.astype(np.uint8, copy=False), mode="L")
    im = im.resize((out_w, out_h), resample=Image.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def minmax01_rgb_per_channel(
    x: np.ndarray,
    eps=1e-8,
    clip_percentiles=(1.0, 99.0),
) -> np.ndarray:
    """
    Percentile-clip + min-max normalization to [-1,1], per RGB channel.
    x: (H,W,3)
    return: (3,H,W), float32
    """
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {x.shape}")

    x = x.astype(np.float32, copy=False)
    p_lo, p_hi = clip_percentiles

    out = np.zeros_like(x, dtype=np.float32)
    for c in range(3):
        xc = x[:, :, c]
        lo, hi = np.percentile(xc, p_lo), np.percentile(xc, p_hi)

        if hi <= lo:
            lo, hi = float(xc.min()), float(xc.max())
            if hi <= lo:
                out[:, :, c] = np.zeros_like(xc, dtype=np.float32)
                continue

        xc = np.clip(xc, lo, hi)
        xc01 = (xc - lo) / (hi - lo + eps)
        out[:, :, c] = (xc01 * 2.0 - 1.0).astype(np.float32, copy=False)

    return np.transpose(out, (2, 0, 1)).astype(np.float32, copy=False)


def load_rgb_image(img_path: Path) -> np.ndarray:
    """
    Read image as RGB uint8 array: (H,W,3)
    """
    img = Image.open(img_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Read mask as grayscale, binarize to {0,1}
    """
    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def to_onehot_no_bg_binary(mask2d: np.ndarray) -> np.ndarray:
    """
    Binary mask -> onehot WITHOUT background
    input : (H,W), {0,1}
    output: (1,H,W), uint8
    """
    if mask2d.ndim != 2:
        raise ValueError(f"Expected (H,W), got {mask2d.shape}")
    return (mask2d > 0).astype(np.uint8)[None, ...]


# ============================================================
# Dataset discovery
# ============================================================
def _list_image_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTS])


def build_key_to_path(files: List[Path]) -> Dict[str, Path]:
    """
    If duplicated normalized key exists, keep the first one.
    """
    out: Dict[str, Path] = {}
    for p in files:
        key = normalize_pair_key(p.stem)
        if key in out:
            print(f"[Warn] duplicate normalized key: {key}")
            print(f"       keep: {out[key].name}")
            print(f"       skip: {p.name}")
            continue
        out[key] = p
    return out


def find_pairs_in_subset(subset_dir: Path, verbose: bool = True) -> List[Tuple[Path, Path, str]]:
    """
    subset_dir:
      kmms_training/
        images/
        masks/
      kmms_test/
        images/
        masks/
    """
    images_dir = subset_dir / "images"
    masks_dir = subset_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        if verbose:
            print(f"[Warn] missing images/ or masks/ in {subset_dir}")
        return []

    image_files = _list_image_files(images_dir)
    mask_files = _list_image_files(masks_dir)

    image_map = build_key_to_path(image_files)
    mask_map = build_key_to_path(mask_files)

    common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
    pairs: List[Tuple[Path, Path, str]] = []

    if verbose:
        print(
            f"[Info] subset={subset_dir.name} | "
            f"images={len(image_files)} masks={len(mask_files)} paired={len(common_keys)}"
        )

    for key in common_keys:
        img_path = image_map[key]
        mask_path = mask_map[key]
        sample_name = f"{sanitize_name(subset_dir.name)}_{sanitize_name(key)}"
        pairs.append((img_path, mask_path, sample_name))

    if verbose:
        only_img = sorted(set(image_map.keys()) - set(mask_map.keys()))
        only_mask = sorted(set(mask_map.keys()) - set(image_map.keys()))
        if only_img[:5]:
            print(f"[Info] only in images ({subset_dir.name}) sample keys: {only_img[:5]}")
        if only_mask[:5]:
            print(f"[Info] only in masks ({subset_dir.name}) sample keys: {only_mask[:5]}")

    return pairs


def collect_all_pairs(base_dir: Path, use_subsets: List[str], verbose: bool = True) -> List[Dict]:
    items: List[Dict] = []
    for subset in use_subsets:
        subset_dir = base_dir / subset
        if not subset_dir.exists():
            if verbose:
                print(f"[Warn] subset not found: {subset_dir}")
            continue

        pairs = find_pairs_in_subset(subset_dir, verbose=verbose)
        for img_path, mask_path, sample_name in pairs:
            items.append(
                {
                    "subset_name": subset,
                    "img_path": str(img_path),
                    "mask_path": str(mask_path),
                    "sample_id": sample_name,
                }
            )
    return items


# ============================================================
# Split
# ============================================================
def make_split(items: List[Dict], seed=42):
    """
    Force 8:1:1 split over whole images.
    """
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(items))
    rng.shuffle(idxs)

    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:n_train + n_val]
    test_idxs = idxs[n_train + n_val:]

    train = [items[i] for i in train_idxs]
    val = [items[i] for i in val_idxs]
    test = [items[i] for i in test_idxs]
    return train, val, test


def save_split_json(out_path: Path, seed: int, train, val, test):
    obj = {
        "seed": seed,
        "counts": {
            "total_images": len(train) + len(val) + len(test),
            "train_images": len(train),
            "val_images": len(val),
            "test_images": len(test),
        },
        "train": [x["sample_id"] for x in train],
        "val": [x["sample_id"] for x in val],
        "test": [x["sample_id"] for x in test],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Mismatch scan
# ============================================================
def scan_spatial_mismatches(items: List[Dict], out_root: Path, verbose: bool = True) -> List[Dict]:
    """
    Print all samples whose image/mask spatial sizes differ.
    Save a mismatch report to out_root/mismatch_report.txt
    """
    mismatch_list: List[Dict] = []

    for item in items:
        img = load_rgb_image(Path(item["img_path"]))
        mask = load_mask(Path(item["mask_path"]))

        if img.shape[:2] != mask.shape:
            mismatch_list.append(
                {
                    "sample_id": item["sample_id"],
                    "img_shape": tuple(img.shape[:2]),
                    "mask_shape": tuple(mask.shape),
                    "img_path": item["img_path"],
                    "mask_path": item["mask_path"],
                }
            )

    report_path = out_root / "mismatch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total mismatches: {len(mismatch_list)}\n\n")
        for x in mismatch_list:
            f.write(
                f"{x['sample_id']} | image={x['img_shape']} mask={x['mask_shape']}\n"
                f"  img_path : {x['img_path']}\n"
                f"  mask_path: {x['mask_path']}\n\n"
            )

    if verbose:
        print("\n========== Checking spatial mismatches ==========")
        print(f"[Info] total mismatch samples: {len(mismatch_list)}")
        for x in mismatch_list:
            print(
                f"[Mismatch] {x['sample_id']} | "
                f"image={x['img_shape']} mask={x['mask_shape']}"
            )

        pattern_counter = Counter()
        for x in mismatch_list:
            pattern_counter[f"{x['img_shape']} vs {x['mask_shape']}"] += 1

        if len(pattern_counter) > 0:
            print("\n[Pattern stats]")
            for k, v in pattern_counter.items():
                print(f"{k}: {v}")

        print(f"\n[Info] mismatch report saved to: {report_path}")
        print("===============================================\n")

    return mismatch_list


# ============================================================
# Crop helpers
# ============================================================
def random_crop_pair(
    img: np.ndarray,
    mask: np.ndarray,
    crop_hw: Tuple[int, int],
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    img : (H,W,3)
    mask: (H,W)
    return:
      img_crop : (crop_h,crop_w,3)
      mask_crop: (crop_h,crop_w)
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be (H,W,3), got {img.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W), got {mask.shape}")
    if img.shape[:2] != mask.shape:
        raise ValueError(f"img/mask spatial mismatch: {img.shape[:2]} vs {mask.shape}")

    crop_h, crop_w = crop_hw
    h, w = mask.shape

    if h < crop_h or w < crop_w:
        raise ValueError(
            f"Crop size {crop_hw} is larger than image size {(h, w)} after resize."
        )

    top = rng.randint(0, h - crop_h)
    left = rng.randint(0, w - crop_w)

    img_crop = img[top:top + crop_h, left:left + crop_w, :]
    mask_crop = mask[top:top + crop_h, left:left + crop_w]

    return img_crop, mask_crop


# ============================================================
# Processing
# ============================================================
def process_one_image_to_crops(
    item: Dict,
    split_name: str,
    out_root: Path,
    resize_to: int,
    size: int,
    p_lo: float,
    p_hi: float,
    num_crops: int,
    rng: random.Random,
    verbose: bool,
) -> int:
    """
    For one original image/mask pair, generate multiple random crops.

    New pipeline:
      1. load original image and mask
      2. resize BOTH to (resize_to, resize_to)
      3. random crop (size, size)

    Output:
      out_root/split/sample_id_crop000/
        img.npy      # (3,H,W), float32, RGB
        onehot.npy   # (1,H,W), uint8
    """
    img_path = Path(item["img_path"])
    mask_path = Path(item["mask_path"])
    sample_id = item["sample_id"]

    if not img_path.exists() or not mask_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {sample_id}")
        return 0

    img = load_rgb_image(img_path)    # (H,W,3), uint8
    mask = load_mask(mask_path)       # (H,W), uint8 {0,1}

    # First resize both image and mask to fixed size 500x500 (or user-defined resize_to)
    img = resize_hw_image_rgb(img, out_hw=(resize_to, resize_to))
    mask = resize_hw_mask(mask, out_hw=(resize_to, resize_to))
    mask = (mask > 0).astype(np.uint8)

    total_ok = 0
    crop_hw = (size, size)

    for k in range(num_crops):
        img_crop, mask_crop = random_crop_pair(img, mask, crop_hw=crop_hw, rng=rng)

        img_norm = minmax01_rgb_per_channel(
            img_crop,
            clip_percentiles=(p_lo, p_hi),
        )  # (3,H,W), float32

        onehot = to_onehot_no_bg_binary(mask_crop)  # (1,H,W)

        crop_id = f"{sample_id}_crop{k:03d}"
        out_dir = out_root / split_name / crop_id
        _ensure_dirs(out_dir)

        np.save(out_dir / "img.npy", img_norm.astype(np.float32))
        np.save(out_dir / "onehot.npy", onehot.astype(np.uint8))

        if verbose:
            print(f"[OK] {crop_id} -> {out_dir}")

        total_ok += 1

    return total_ok


def run(
    base_dir: Path,
    out_root: Path,
    resize_to: int,
    size: int,
    p_lo: float,
    p_hi: float,
    seed: int,
    split_json: Path | None,
    use_subsets: List[str],
    crops_per_train: int,
    crops_per_val: int,
    crops_per_test: int,
    verbose: bool,
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    items = collect_all_pairs(base_dir, use_subsets, verbose=verbose)
    if not items:
        raise RuntimeError(f"No valid image-mask pairs found under {base_dir}")

    print(f"[Info] total paired images from all subsets: {len(items)}")

    # Print and save all original size-mismatch samples
    scan_spatial_mismatches(items, out_root=out_root, verbose=verbose)

    if split_json is not None and split_json.exists():
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)

        id_to_item = {x["sample_id"]: x for x in items}
        train_items = [id_to_item[sid] for sid in split["train"] if sid in id_to_item]
        val_items = [id_to_item[sid] for sid in split["val"] if sid in id_to_item]
        test_items = [id_to_item[sid] for sid in split["test"] if sid in id_to_item]

        if verbose:
            print(f"[Info] Reusing split: {split_json}")
    else:
        train_items, val_items, test_items = make_split(items, seed=seed)
        save_split_json(out_root / "split.json", seed, train_items, val_items, test_items)
        if verbose:
            print(f"[Info] Created split.json at: {out_root / 'split.json'}")

    print(
        f"[Info] split counts | train={len(train_items)} val={len(val_items)} test={len(test_items)}"
    )

    rng = random.Random(seed)
    total_ok = 0

    split_to_items = {
        "train": (train_items, crops_per_train),
        "val": (val_items, crops_per_val),
        "test": (test_items, crops_per_test),
    }

    for split_name, (split_items, num_crops) in split_to_items.items():
        for item in split_items:
            total_ok += process_one_image_to_crops(
                item=item,
                split_name=split_name,
                out_root=out_root,
                resize_to=resize_to,
                size=size,
                p_lo=p_lo,
                p_hi=p_hi,
                num_crops=num_crops,
                rng=rng,
                verbose=verbose,
            )

    if verbose:
        print("\n====================")
        print(f"Done. Crop samples processed: {total_ok}")
        print(f"Split json: {out_root / 'split.json'}")
        print(f"Mismatch report: {out_root / 'mismatch_report.txt'}")
        print("====================\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default="./MonuSegdataset")
    ap.add_argument("--out_root", type=str, default="./dataset/MonuSeg/ONEHOT")

    ap.add_argument("--resize_to", type=int, default=500, help="resize image and mask to resize_to x resize_to first")
    ap.add_argument("--size", type=int, default=256, help="random crop size x size after resize")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--split_json", type=str, default=None, help="optional: reuse existing split.json")

    ap.add_argument(
        "--use_subsets",
        nargs="+",
        default=["kmms_training", "kmms_test"],
        choices=["kmms_training", "kmms_test"],
        help="which subset folders under base_dir to use",
    )

    ap.add_argument("--crops_per_train", type=int, default=4)
    ap.add_argument("--crops_per_val", type=int, default=2)
    ap.add_argument("--crops_per_test", type=int, default=2)

    ap.add_argument("--quiet", action="store_true", help="do not print logs")

    args = ap.parse_args()

    if args.size > args.resize_to:
        raise ValueError(
            f"--size ({args.size}) must be <= --resize_to ({args.resize_to})"
        )

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        resize_to=int(args.resize_to),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        use_subsets=list(args.use_subsets),
        crops_per_train=int(args.crops_per_train),
        crops_per_val=int(args.crops_per_val),
        crops_per_test=int(args.crops_per_test),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()