# utils/Covid19Preprocess.py
#
# COVID19 CT lesion segmentation preprocessing (single-channel version).
#
# Expected structure:
#   base_dir/
#     frames/
#       xxx.png
#       yyy.png
#       ...
#     masks/
#       xxx.png
#       yyy.png
#       ...
#
# Processing:
# - Pair files by identical filename between frames/ and masks/
# - Randomly shuffle all paired samples
# - Strictly split into train/val/test with ratio 8:1:1
# - Resize image and mask directly to size x size
# - image: grayscale, percentile clip normalize to [-1,1]
# - mask: converted to binary mask {0,1}, then saved as onehot WITHOUT background => (1,H,W)
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
#   python utils/Covid19Preprocess.py \
#       --base_dir ./dataset/covid19 \
#       --out_root ./dataset/covid19/ONEHOT \
#       --size 256 \
#       --p_lo 1 --p_hi 99 \
#       --seed 42

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

_VALID_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
def resize_hw_image_gray(x: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")

    out_h, out_w = out_hw
    im = Image.fromarray(x.astype(np.uint8, copy=False), mode="L")
    im = im.resize((out_w, out_h), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def resize_hw_mask(x: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")

    out_h, out_w = out_hw
    im = Image.fromarray(x.astype(np.uint8, copy=False), mode="L")
    im = im.resize((out_w, out_h), resample=Image.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def minmax01_gray(
    x: np.ndarray,
    eps=1e-8,
    clip_percentiles=(1.0, 99.0),
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")

    x = x.astype(np.float32, copy=False)
    p_lo, p_hi = clip_percentiles

    lo, hi = np.percentile(x, p_lo), np.percentile(x, p_hi)

    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros((1, x.shape[0], x.shape[1]), dtype=np.float32)

    x = np.clip(x, lo, hi)
    x01 = (x - lo) / (hi - lo + eps)
    x11 = x01 * 2.0 - 1.0

    return x11[None, ...].astype(np.float32, copy=False)


# ============================================================
# Load image / mask
# ============================================================
def load_gray_image(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def load_covid19_mask(mask_path: Path) -> np.ndarray:
    """
    Load mask and convert to binary foreground mask.
    Assumption:
      - background = 0
      - lesion/infection = non-zero
    """
    mask = Image.open(mask_path)
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = (mask.max(axis=2) > 0).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    return mask


def to_onehot_no_bg_binary(mask2d: np.ndarray) -> np.ndarray:
    if mask2d.ndim != 2:
        raise ValueError(f"Expected (H,W), got {mask2d.shape}")
    return (mask2d > 0).astype(np.uint8)[None, ...]


# ============================================================
# Dataset discovery
# ============================================================
def discover_covid19_pairs(base_dir: Path, verbose: bool = True) -> List[Dict]:
    """
    Discover image-mask pairs from:
      base_dir/frames/
      base_dir/masks/
    Pairing rule: same filename
    """
    frames_dir = base_dir / "frames"
    masks_dir = base_dir / "masks"

    if not frames_dir.exists():
        raise RuntimeError(f"frames directory not found: {frames_dir}")
    if not masks_dir.exists():
        raise RuntimeError(f"masks directory not found: {masks_dir}")

    frame_files = []
    for p in frames_dir.iterdir():
        if p.is_file() and p.suffix.lower() in _VALID_IMG_EXTS:
            frame_files.append(p)

    frame_files = sorted(frame_files)

    if verbose:
        print(f"[Info] Found {len(frame_files)} frame files in {frames_dir}")

    items: List[Dict] = []
    missing_masks = 0

    for img_path in frame_files:
        mask_path = masks_dir / img_path.name

        if not mask_path.exists():
            found = False
            for ext in _VALID_IMG_EXTS:
                alt = masks_dir / f"{img_path.stem}{ext}"
                if alt.exists():
                    mask_path = alt
                    found = True
                    break
            if not found:
                missing_masks += 1
                if verbose:
                    print(f"[Warn] Missing mask for image: {img_path.name}")
                continue

        sample_id = sanitize_name(img_path.stem)

        items.append(
            {
                "img_path": str(img_path),
                "anno_path": str(mask_path),
                "sample_id": sample_id,
            }
        )

    if verbose:
        print(f"[Info] Successfully paired {len(items)} samples.")
        if missing_masks > 0:
            print(f"[Warn] Missing masks for {missing_masks} images.")

    return items


# ============================================================
# Split
# ============================================================
def make_split(items: List[Dict], seed=42):
    """
    Shuffle all items and split with ratio 8:1:1
    """
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(items))
    rng.shuffle(idxs)

    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

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
        img = load_gray_image(Path(item["img_path"]))
        mask = load_covid19_mask(Path(item["anno_path"]))

        if img.shape != mask.shape:
            mismatch_list.append(
                {
                    "sample_id": item["sample_id"],
                    "img_shape": tuple(img.shape),
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
    img_path = Path(item["img_path"])
    anno_path = Path(item["anno_path"])
    sample_id = item["sample_id"]

    if not img_path.exists() or not anno_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {sample_id}")
        return 0

    img = load_gray_image(img_path)       # (H,W)
    mask = load_covid19_mask(anno_path)   # (H,W), binary

    img_resized = resize_hw_image_gray(img, out_hw=(size, size))
    mask_resized = resize_hw_mask(mask, out_hw=(size, size))
    mask_resized = (mask_resized > 0).astype(np.uint8)

    img_norm = minmax01_gray(
        img_resized,
        clip_percentiles=(p_lo, p_hi),
    )  # (1,H,W)

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

    items = discover_covid19_pairs(base_dir, verbose=verbose)
    if not items:
        raise RuntimeError(f"No valid image-mask pairs found under {base_dir}")

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

    ap.add_argument("--base_dir", type=str, default="./covid19")
    ap.add_argument("--out_root", type=str, default="./dataset/covid19/ONEHOT")
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