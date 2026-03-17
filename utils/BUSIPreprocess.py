# utils/DDTIPreprocess_PIL.py
#
# DDTI-style preprocessing for PNG breast ultrasound dataset:
# - base_dir/
#     benign/
#     malignant/
#     normal/
# - Each image + mask pair => one sample folder
# - Resize to 256x256 using PIL:
#     image: bilinear
#     mask : nearest
# - Normalize image to [-1,1] using percentile clip
# - Binary mask -> onehot WITHOUT background: (1,H,W)
# - Split on samples (8:1:1)
#
# Output:
#   out_root/
#     split.json
#     train/benign_0001/{img.npy, onehot.npy}
#     val/...
#     test/...
#
# Usage:
#   python utils/DDTIPreprocess_PIL.py \
#       --base_dir ./Dataset_BUSI_with_GT \
#       --out_root ./dataset/DDTI/ONEHOT \
#       --size 256 \
#       --p_lo 1 --p_hi 99 \
#       --seed 42

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image


# ============================================================
# Utils
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resize_hw(x: np.ndarray, out_hw=(256, 256), is_mask: bool = False) -> np.ndarray:
    """
    Resize to (out_h, out_w) using PIL:
      - bilinear for image
      - nearest for mask
    x: (H,W)
    return: (out_h,out_w)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")

    out_h, out_w = out_hw

    if is_mask:
        im = Image.fromarray(x.astype(np.uint8, copy=False), mode="L")
        im = im.resize((out_w, out_h), resample=Image.NEAREST)
        y = np.asarray(im, dtype=np.uint8)
        return y
    else:
        im = Image.fromarray(x.astype(np.float32, copy=False), mode="F")
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


def load_grayscale_image(img_path: Path) -> np.ndarray:
    """
    Read png/jpg as grayscale (H,W), float32 for image usage
    """
    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Read mask as grayscale (H,W), then binarize to {0,1}
    """
    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    arr = (arr > 0).astype(np.uint8)
    return arr


def to_onehot_no_bg_binary(mask2d: np.ndarray) -> np.ndarray:
    """
    Binary mask -> onehot WITHOUT background
    input : (H,W), {0,1}
    output: (1,H,W), uint8
    """
    if mask2d.ndim != 2:
        raise ValueError(f"Expected (H,W), got {mask2d.shape}")
    onehot = (mask2d > 0).astype(np.uint8)[None, ...]
    return onehot


# ============================================================
# Dataset discovery
# ============================================================
_ALLOWED_CLASSES = {"benign", "malignant", "normal"}


def _is_mask_file(p: Path) -> bool:
    return "_mask" in p.stem


def find_pairs_in_class_dir(class_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Return list of (img_path, mask_path, sample_name)
    sample_name example: malignant_001
    """
    pairs = []

    image_files = sorted([p for p in class_dir.glob("*.png") if not _is_mask_file(p)])
    for img_path in image_files:
        mask_path = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")
        if not mask_path.exists():
            continue

        # sanitize sample id
        raw_name = img_path.stem
        safe_name = re.sub(r"\s+", "_", raw_name)
        sample_name = f"{class_dir.name}_{safe_name}"
        pairs.append((img_path, mask_path, sample_name))

    return pairs


def collect_all_pairs(base_dir: Path, use_classes: List[str]) -> List[Dict]:
    items = []
    for cls in use_classes:
        class_dir = base_dir / cls
        if not class_dir.exists():
            continue

        pairs = find_pairs_in_class_dir(class_dir)
        for img_path, mask_path, sample_name in pairs:
            items.append({
                "class_name": cls,
                "img_path": str(img_path),
                "mask_path": str(mask_path),
                "sample_id": sample_name,
            })
    return items


# ============================================================
# Split
# ============================================================
def make_split(items: List[Dict], seed=42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(items))
    rng.shuffle(idxs)

    n = len(items)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_idxs = idxs[:n_train].tolist()
    val_idxs = idxs[n_train:n_train + n_val].tolist()
    test_idxs = idxs[n_train + n_val:].tolist()

    train = [items[i] for i in train_idxs]
    val = [items[i] for i in val_idxs]
    test = [items[i] for i in test_idxs]
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
        "train": [x["sample_id"] for x in train],
        "val": [x["sample_id"] for x in val],
        "test": [x["sample_id"] for x in test],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Processing
# ============================================================
def process_one_sample(
    item: Dict,
    split_name: str,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    verbose: bool,
) -> bool:
    """
    Output:
      out_root/split/sample_id/
        img.npy
        onehot.npy
    """
    img_path = Path(item["img_path"])
    mask_path = Path(item["mask_path"])
    sample_id = item["sample_id"]

    if not img_path.exists() or not mask_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {sample_id}")
        return False

    # load
    img2d = load_grayscale_image(img_path)   # (H,W), float32
    mask2d = load_mask(mask_path)            # (H,W), uint8 {0,1}

    # resize
    out_hw = (size, size)
    img2d = resize_hw(img2d, out_hw=out_hw, is_mask=False)
    mask2d = resize_hw(mask2d, out_hw=out_hw, is_mask=True)

    # make sure binary again after resize
    mask2d = (mask2d > 0).astype(np.uint8)

    # normalize image to [-1,1]
    img_norm = minmax01_single(img2d, clip_percentiles=(p_lo, p_hi))

    # onehot WITHOUT background
    onehot = to_onehot_no_bg_binary(mask2d)  # (1,H,W)

    # save
    out_dir = out_root / split_name / sample_id
    _ensure_dirs(out_dir)
    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot.astype(np.uint8))

    if verbose:
        print(f"[OK] {sample_id} -> {out_dir}")

    return True


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    seed: int,
    split_json: Path | None,
    use_classes: List[str],
    verbose: bool,
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    items = collect_all_pairs(base_dir, use_classes)
    if not items:
        raise RuntimeError(f"No valid image-mask pairs found under {base_dir}")

    # split
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
        train_items, val_items, test_items = make_split(items, seed=seed, ratios=(0.8, 0.1, 0.1))
        save_split_json(out_root / "split.json", seed, train_items, val_items, test_items)
        if verbose:
            print(f"[Info] Created split.json at: {out_root / 'split.json'}")

    total_ok = 0
    for split_name, split_items in [
        ("train", train_items),
        ("val", val_items),
        ("test", test_items),
    ]:
        for item in split_items:
            ok = process_one_sample(
                item=item,
                split_name=split_name,
                out_root=out_root,
                size=size,
                p_lo=p_lo,
                p_hi=p_hi,
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
    ap.add_argument("--base_dir", type=str, default="./BUSIdataset")
    ap.add_argument("--out_root", type=str, default="./dataset/BUSI/ONEHOT")
    ap.add_argument("--size", type=int, default=256, help="resize 到固定尺寸 size x size")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="可选：复用已有 split.json")
    ap.add_argument("--use_classes", nargs="+", default=["benign", "malignant", "normal"], choices=["benign", "malignant", "normal"])
    ap.add_argument("--quiet", action="store_true", help="不打印日志")
    args = ap.parse_args()

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        use_classes=list(args.use_classes),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()