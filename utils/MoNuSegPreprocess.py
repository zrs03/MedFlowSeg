# utils/MonuSegPreprocess_XML.py
#
# Official MoNuSeg preprocessing from TIFF + XML polygon annotations
#
# Supported structure:
#
# 1) Training:
#    base_dir/
#      MoNuSeg 2018 Training Data/
#        Tissue Images/
#          *.tif
#        Annotations/
#          *.xml
#
# 2) Test:
#    base_dir/
#      MoNuSegTestData/
#        *.tif
#        *.xml
#
# Processing:
# - collect train/test images together
# - pair image and xml by normalized filename
# - rasterize XML polygons into FILLED binary masks
# - split on whole images first (8:1:1)
# - FIRST resize both image and mask to resize_to x resize_to
# - THEN random crop to size x size
# - image: RGB kept as 3-channel, percentile clip normalize to [-1,1] per channel
# - mask : binary {0,1}
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
#   python utils/MonuSegPreprocess_XML.py \
#       --base_dir ./MoNuSeg \
#       --out_root ./dataset/MonuSeg/ONEHOT \
#       --resize_to 500 \
#       --size 256 \
#       --crops_per_train 4 \
#       --crops_per_val 2 \
#       --crops_per_test 2 \
#       --p_lo 1 --p_hi 99 \
#       --seed 42

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


_VALID_IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


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
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
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


def to_onehot_no_bg_binary(mask2d: np.ndarray) -> np.ndarray:
    if mask2d.ndim != 2:
        raise ValueError(f"Expected (H,W), got {mask2d.shape}")
    return (mask2d > 0).astype(np.uint8)[None, ...]


# ============================================================
# XML -> filled mask
# ============================================================
def parse_monuseg_xml_polygons(xml_path: Path) -> List[List[Tuple[float, float]]]:
    """
    Parse MoNuSeg XML and return list of polygons.
    Each polygon is a list of (x, y).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons: List[List[Tuple[float, float]]] = []

    # 官方/常见结构通常是:
    # Annotations -> Annotation -> Regions -> Region -> Vertices -> Vertex(X,Y)
    # 这里用通用遍历，尽量兼容
    for region in root.findall(".//Region"):
        pts: List[Tuple[float, float]] = []
        for vertex in region.findall(".//Vertex"):
            x = vertex.attrib.get("X", None)
            y = vertex.attrib.get("Y", None)
            if x is None or y is None:
                continue
            pts.append((float(x), float(y)))
        if len(pts) >= 3:
            polygons.append(pts)

    return polygons


def rasterize_polygons_to_mask(
    polygons: List[List[Tuple[float, float]]],
    hw: Tuple[int, int],
) -> np.ndarray:
    """
    Rasterize filled polygons to binary mask.
    hw: (H, W)
    """
    h, w = hw
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    for poly in polygons:
        # PIL expects (x, y)
        xy = [(float(x), float(y)) for x, y in poly]
        draw.polygon(xy, outline=1, fill=1)

    mask = np.asarray(canvas, dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def load_mask_from_xml(xml_path: Path, hw: Tuple[int, int]) -> np.ndarray:
    polygons = parse_monuseg_xml_polygons(xml_path)
    mask = rasterize_polygons_to_mask(polygons, hw=hw)
    return mask


# ============================================================
# Dataset discovery
# ============================================================
def _list_image_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _VALID_IMG_EXTS])


def _list_xml_files(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".xml"])


def build_key_to_path(files: List[Path]) -> Dict[str, Path]:
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


def find_pairs_training(training_root: Path, verbose: bool = True) -> List[Tuple[Path, Path, str]]:
    """
    Training:
      MoNuSeg 2018 Training Data/
        Tissue Images/*.tif
        Annotations/*.xml
    """
    images_dir = training_root / "Tissue Images"
    ann_dir = training_root / "Annotations"

    if not images_dir.exists() or not ann_dir.exists():
        if verbose:
            print(f"[Warn] missing Tissue Images or Annotations in {training_root}")
        return []

    image_files = _list_image_files(images_dir)
    xml_files = _list_xml_files(ann_dir)

    image_map = build_key_to_path(image_files)
    xml_map = build_key_to_path(xml_files)

    common_keys = sorted(set(image_map.keys()) & set(xml_map.keys()))
    pairs = []

    if verbose:
        print(
            f"[Info] training | images={len(image_files)} xml={len(xml_files)} paired={len(common_keys)}"
        )

    for key in common_keys:
        img_path = image_map[key]
        xml_path = xml_map[key]
        sample_name = f"train_{sanitize_name(key)}"
        pairs.append((img_path, xml_path, sample_name))

    return pairs


def find_pairs_test(test_root: Path, verbose: bool = True) -> List[Tuple[Path, Path, str]]:
    """
    Test:
      MoNuSegTestData/
        *.tif
        *.xml
    """
    if not test_root.exists():
        if verbose:
            print(f"[Warn] test root not found: {test_root}")
        return []

    image_files = _list_image_files(test_root)
    xml_files = _list_xml_files(test_root)

    image_map = build_key_to_path(image_files)
    xml_map = build_key_to_path(xml_files)

    common_keys = sorted(set(image_map.keys()) & set(xml_map.keys()))
    pairs = []

    if verbose:
        print(
            f"[Info] test | images={len(image_files)} xml={len(xml_files)} paired={len(common_keys)}"
        )

    for key in common_keys:
        img_path = image_map[key]
        xml_path = xml_map[key]
        sample_name = f"test_{sanitize_name(key)}"
        pairs.append((img_path, xml_path, sample_name))

    return pairs


def collect_all_pairs(base_dir: Path, verbose: bool = True) -> List[Dict]:
    items: List[Dict] = []

    training_root = base_dir / "MoNuSeg 2018 Training Data"
    test_root = base_dir / "MoNuSegTestData"

    train_pairs = find_pairs_training(training_root, verbose=verbose)
    test_pairs = find_pairs_test(test_root, verbose=verbose)

    for img_path, xml_path, sample_name in train_pairs:
        items.append(
            {
                "subset_name": "training",
                "img_path": str(img_path),
                "xml_path": str(xml_path),
                "sample_id": sample_name,
            }
        )

    for img_path, xml_path, sample_name in test_pairs:
        items.append(
            {
                "subset_name": "test",
                "img_path": str(img_path),
                "xml_path": str(xml_path),
                "sample_id": sample_name,
            }
        )

    return items


# ============================================================
# Split
# ============================================================
def make_split(items: List[Dict], seed=42):
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
    XML rasterized mask should match image size.
    """
    mismatch_list: List[Dict] = []

    for item in items:
        img = load_rgb_image(Path(item["img_path"]))
        mask = load_mask_from_xml(Path(item["xml_path"]), hw=img.shape[:2])

        if img.shape[:2] != mask.shape:
            mismatch_list.append(
                {
                    "sample_id": item["sample_id"],
                    "img_shape": tuple(img.shape[:2]),
                    "mask_shape": tuple(mask.shape),
                    "img_path": item["img_path"],
                    "xml_path": item["xml_path"],
                }
            )

    report_path = out_root / "mismatch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Total mismatches: {len(mismatch_list)}\n\n")
        for x in mismatch_list:
            f.write(
                f"{x['sample_id']} | image={x['img_shape']} mask={x['mask_shape']}\n"
                f"  img_path : {x['img_path']}\n"
                f"  xml_path : {x['xml_path']}\n\n"
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
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be (H,W,3), got {img.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H,W), got {mask.shape}")
    if img.shape[:2] != mask.shape:
        raise ValueError(f"img/mask spatial mismatch: {img.shape[:2]} vs {mask.shape}")

    crop_h, crop_w = crop_hw
    h, w = mask.shape

    if h < crop_h or w < crop_w:
        raise ValueError(f"Crop size {crop_hw} is larger than image size {(h, w)} after resize.")

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
    img_path = Path(item["img_path"])
    xml_path = Path(item["xml_path"])
    sample_id = item["sample_id"]

    if not img_path.exists() or not xml_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {sample_id}")
        return 0

    img = load_rgb_image(img_path)              # (H,W,3)
    mask = load_mask_from_xml(xml_path, hw=img.shape[:2])  # (H,W), binary filled

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
        )  # (3,H,W)

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
    crops_per_train: int,
    crops_per_val: int,
    crops_per_test: int,
    verbose: bool,
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    items = collect_all_pairs(base_dir, verbose=verbose)
    if not items:
        raise RuntimeError(f"No valid image-xml pairs found under {base_dir}")

    print(f"[Info] total paired images from all subsets: {len(items)}")

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

    print(f"[Info] split counts | train={len(train_items)} val={len(val_items)} test={len(test_items)}")

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

    ap.add_argument("--base_dir", type=str, default="./MoNuSegdataset")
    ap.add_argument("--out_root", type=str, default="./dataset/MoNuSeg/ONEHOT")

    ap.add_argument("--resize_to", type=int, default=500, help="resize image and mask to resize_to x resize_to first")
    ap.add_argument("--size", type=int, default=256, help="random crop size x size after resize")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--split_json", type=str, default=None, help="optional: reuse existing split.json")

    ap.add_argument("--crops_per_train", type=int, default=4)
    ap.add_argument("--crops_per_val", type=int, default=2)
    ap.add_argument("--crops_per_test", type=int, default=2)

    ap.add_argument("--quiet", action="store_true", help="do not print logs")

    args = ap.parse_args()

    if args.size > args.resize_to:
        raise ValueError(f"--size ({args.size}) must be <= --resize_to ({args.resize_to})")

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        resize_to=int(args.resize_to),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        crops_per_train=int(args.crops_per_train),
        crops_per_val=int(args.crops_per_val),
        crops_per_test=int(args.crops_per_test),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()