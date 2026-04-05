# utils/GlaSPreprocess.py
#
# Official GlaS dataset preprocessing from BMP images and annotation masks.
#
# Expected structure (based on typical GlaS distribution):
#    base_dir/
#      train_1.bmp
#      train_1_anno.bmp
#      testA_1.bmp
#      testA_1_anno.bmp
#      ...
#
# Processing:
# - Discover all images and pair them with their `_anno` counterparts.
# - If files are named with `train_`, `testA_`, `testB_`, use that to create a default split.
# - Resize both image and mask DIRECTLY to `size` x `size` (default 256x256).
# - image: RGB kept as 3-channel, percentile clip normalize to [-1,1] per channel.
# - mask: The original GlaS annotations are instance masks.
#         This script converts them to binary masks {0, 1} where > 0 is foreground.
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
#   python utils/GlaSPreprocess.py \
#       --base_dir ./dataset/GlaS \
#       --out_root ./dataset/GlaS/ONEHOT \
#       --size 256 \
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

_VALID_IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# Utils
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize_name(s: str) -> str:
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    return s


# ============================================================
# Resize / normalize
# ============================================================
def resize_hw_image_rgb(x: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {x.shape}")

    out_h, out_w = out_hw
    im = Image.fromarray(x.astype(np.uint8, copy=False), mode="RGB")
    im = im.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def resize_hw_mask(x: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    # GlaS masks can have values > 255 if they are 16-bit instance masks.
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")

    out_h, out_w = out_hw
    im = Image.fromarray(x.astype(np.int32, copy=False), mode="I")
    im = im.resize((out_w, out_h), resample=Image.NEAREST)
    return np.asarray(im, dtype=np.int32)


def minmax01_rgb_per_channel(
    x: np.ndarray,
    eps=1e-8,
    clip_percentiles=(1.0, 99.0),
) -> np.ndarray:
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
    img = Image.open(img_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_glas_mask(mask_path: Path) -> np.ndarray:
    """
    Load GlaS mask. Annotations are often indexed (instance IDs).
    We load it and convert it to binary (foreground > 0).
    """
    mask_img = Image.open(mask_path)
    mask = np.asarray(mask_img)
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask


def to_onehot_no_bg_binary(mask2d: np.ndarray) -> np.ndarray:
    if mask2d.ndim != 2:
        raise ValueError(f"Expected (H,W), got {mask2d.shape}")
    return (mask2d > 0).astype(np.uint8)[None, ...]


# ============================================================
# Dataset discovery
# ============================================================
def discover_glas_pairs(base_dir: Path, verbose: bool = True) -> List[Dict]:
    """
    Finds all image and annotation pairs in the GlaS dataset.
    Looks recursively for anything matching `*_anno.bmp`.
    """
    items: List[Dict] = []
    anno_files = list(base_dir.rglob("*_anno.*"))
    
    if verbose:
        print(f"[Info] Found {len(anno_files)} annotation files.")

    paired_count = 0
    for anno_path in anno_files:
        if anno_path.suffix.lower() not in _VALID_IMG_EXTS:
            continue

        img_stem = anno_path.stem.replace("_anno", "")
        img_name = f"{img_stem}{anno_path.suffix}"
        img_path = anno_path.parent / img_name

        if not img_path.exists():
            found = False
            for ext in _VALID_IMG_EXTS:
                alt_path = anno_path.parent / f"{img_stem}{ext}"
                if alt_path.exists():
                    img_path = alt_path
                    found = True
                    break
            if not found:
                if verbose:
                    print(f"[Warn] Missing image for annotation: {anno_path.name}")
                continue

        subset_hint = "train"
        name_lower = img_stem.lower()
        if "testa" in name_lower or "testb" in name_lower or "test" in name_lower:
            subset_hint = "test"
        elif "train" in name_lower:
            subset_hint = "train"
            
        sample_id = sanitize_name(img_stem)

        items.append(
            {
                "subset_hint": subset_hint,
                "img_path": str(img_path),
                "anno_path": str(anno_path),
                "sample_id": sample_id,
            }
        )
        paired_count += 1

    if verbose:
        print(f"[Info] Successfully paired {paired_count} images with annotations.")

    return items


# ============================================================
# Split
# ============================================================
# ============================================================
# Split (Modified: 严格按照 8:1:1 混合随机划分)
# ============================================================
def make_split(items: List[Dict], seed=42):
    """
    忽略所有命名提示，将所有数据混合后随机打乱，
    严格按照 8:1:1 的比例划分为 train, val, test。
    """
    rng = np.random.default_rng(seed)
    
    # 获取所有数据的索引
    idxs = np.arange(len(items))
    
    # 随机打乱索引
    rng.shuffle(idxs)
    
    n = len(items)
    
    # 计算划分数量
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    # 根据打乱后的索引获取对应的 item
    train = [items[i] for i in idxs[:n_train]]
    val = [items[i] for i in idxs[n_train:n_train + n_val]]
    test = [items[i] for i in idxs[n_train + n_val:]]
    
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
    mismatch_list: List[Dict] = []

    for item in items:
        img = load_rgb_image(Path(item["img_path"]))
        mask = load_glas_mask(Path(item["anno_path"]))

        if img.shape[:2] != mask.shape:
            mismatch_list.append(
                {
                    "sample_id": item["sample_id"],
                    "img_shape": tuple(img.shape[:2]),
                    "mask_shape": tuple(mask.shape),
                    "img_path": item["img_path"],
                    "anno_path": item["anno_path"],
                }
            )

    report_path = out_root / "mismatch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total mismatches: {len(mismatch_list)}\n\n")
        for x in mismatch_list:
            f.write(
                f"{x['sample_id']} | image={x['img_shape']} mask={x['mask_shape']}\n"
                f"  img_path : {x['img_path']}\n"
                f"  anno_path : {x['anno_path']}\n\n"
            )

    if verbose:
        print("\n========== Checking spatial mismatches ==========")
        print(f"[Info] total mismatch samples: {len(mismatch_list)}")
        if mismatch_list:
            for x in mismatch_list:
                print(
                    f"[Mismatch] {x['sample_id']} | "
                    f"image={x['img_shape']} mask={x['mask_shape']}"
                )

        print(f"\n[Info] mismatch report saved to: {report_path}")
        print("===============================================\n")

    return mismatch_list


# ============================================================
# Processing
# ============================================================
def process_one_image_to_resize(
    item: Dict,
    split_name: str,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    verbose: bool,
) -> int:
    """
    Resizes image and mask directly to size x size and saves them.
    """
    img_path = Path(item["img_path"])
    anno_path = Path(item["anno_path"])
    sample_id = item["sample_id"]

    if not img_path.exists() or not anno_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {sample_id}")
        return 0

    img = load_rgb_image(img_path)              # (H,W,3)
    mask = load_glas_mask(anno_path)            # (H,W), binary {0,1}

    # DIRECT RESIZE to target size
    img_resized = resize_hw_image_rgb(img, out_hw=(size, size))
    mask_resized = resize_hw_mask(mask, out_hw=(size, size))
    mask_resized = (mask_resized > 0).astype(np.uint8)

    # NORMALIZE
    img_norm = minmax01_rgb_per_channel(
        img_resized,
        clip_percentiles=(p_lo, p_hi),
    )  # (3,H,W)

    # ONEHOT
    onehot = to_onehot_no_bg_binary(mask_resized)  # (1,H,W)

    out_dir = out_root / split_name / sample_id
    _ensure_dirs(out_dir)

    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot.astype(np.uint8))

    if verbose:
        print(f"[OK] {sample_id} -> {out_dir}")

    return 1


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    seed: int,
    split_json: Path | None,
    verbose: bool,
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    items = discover_glas_pairs(base_dir, verbose=verbose)
    if not items:
        raise RuntimeError(f"No valid image-annotation pairs found under {base_dir}")

    print(f"[Info] Total paired images discovered: {len(items)}")

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

    print(f"[Info] Split counts | train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    total_ok = 0
    split_to_items = {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    for split_name, split_items in split_to_items.items():
        for item in split_items:
            total_ok += process_one_image_to_resize(
                item=item,
                split_name=split_name,
                out_root=out_root,
                size=size,
                p_lo=p_lo,
                p_hi=p_hi,
                verbose=verbose,
            )

    if verbose:
        print("\n====================")
        print(f"Done. Resized samples processed: {total_ok}")
        print(f"Split json: {out_root / 'split.json'}")
        print(f"Mismatch report: {out_root / 'mismatch_report.txt'}")
        print("====================\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", type=str, default="./GlaSdataset")
    ap.add_argument("--out_root", type=str, default="./dataset/GlaS/ONEHOT")
    
    # Replaced resize_to and crop arguments with a single target size
    ap.add_argument("--size", type=int, default=256, help="Target H=W size for direct resize")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--split_json", type=str, default=None, help="optional: reuse existing split.json")

    ap.add_argument("--quiet", action="store_true", help="do not print logs")

    args = ap.parse_args()

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()