# ISICPreprocess.py
#
# Preprocess ISIC2018 dataset for segmentation with SDF:
# 1) Load RGB image and binary GT (lesion segmentation) with OpenCV
# 2) Merge all provided train/val/test files into one pool
# 3) Randomly split all files into train/val/test with ratio 8:1:1
# 4) Resize to fixed size with OpenCV
# 5) Normalize image by: x / 255.0 * 2.0 - 1.0
# 6) Convert GT to one-hot (no background by default) and then to per-channel SDF
# 7) Save outputs to:
#    out_root/
#      split.json
#      train/ISIC_xxxx/
#         img.npy        (3,H,W) float32 in [-1,1]
#         onehot.npy     (1,H,W) uint8
#         sdf.npy        (1,H,W) float32 in [-1,1]
#      val/...
#      test/...

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Tuple, List, Optional, Dict

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# -------------------------
# Utils
# -------------------------

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resize_image(
    x: np.ndarray,
    out_hw=(256, 256),
    is_mask: bool = False,
):
    """
    Resize (H,W) or (H,W,C) array to target out_hw using OpenCV.

    Args:
        x: input image or mask
        out_hw: (H, W)
        is_mask:
            - False: use bilinear interpolation
            - True:  use nearest neighbor
    """
    out_h, out_w = out_hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    if x.ndim == 2:
        return cv2.resize(x, (out_w, out_h), interpolation=interp)
    elif x.ndim == 3:
        return cv2.resize(x, (out_w, out_h), interpolation=interp)
    else:
        raise ValueError(f"Unsupported ndim={x.ndim}, expected 2 or 3.")


def normalize_to_minus1_1_from_255(x: np.ndarray) -> np.ndarray:
    """
    Normalize image intensities by:
        x / 255.0 * 2.0 - 1.0
    Assumes input is in [0,255].
    """
    x = x.astype(np.float32, copy=False)
    x = x / 255.0
    return (x * 2.0 - 1.0).astype(np.float32, copy=False)


def to_onehot_isic(gt2d: np.ndarray, num_classes: int = 2, no_bg: bool = True) -> np.ndarray:
    """
    gt2d: (H,W) labels in {0,1} -> (C,H,W) one-hot uint8.

    Args:
        gt2d: 2D ground truth mask
        num_classes: total number of classes (including background)
        no_bg: if True, return only foreground channels (C=num_classes-1)
    """
    gt2d = gt2d.astype(np.int64, copy=False)
    gt2d = np.clip(gt2d, 0, num_classes - 1)

    if no_bg:
        onehot = np.zeros((1, gt2d.shape[0], gt2d.shape[1]), dtype=np.uint8)
        onehot[0] = (gt2d == 1).astype(np.uint8)
    else:
        onehot = np.eye(num_classes, dtype=np.uint8)[gt2d]   # (H,W,C)
        onehot = np.transpose(onehot, (2, 0, 1))             # (C,H,W)

    return onehot


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
    sdf = dist_out - dist_in
      - inside mask: negative
      - outside mask: positive
      - boundary: ~0

    Clip to [-clip, clip]; if normalize=True, divide by clip -> [-1,1].
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
# Split handling
# -------------------------

def list_images_from_dir(input_dir: Path, gt_dir: Path) -> List[str]:
    """Collect all image base names (without extension) that have both .jpg and _segmentation.png."""
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)

    images = []
    for jpg_path in sorted(input_dir.glob("*.jpg")):
        stem = jpg_path.stem
        gt_path = gt_dir / f"{stem}_segmentation.png"
        if gt_path.exists():
            images.append(stem)
        else:
            print(f"[WARN] Missing GT for {stem}, skipping.")
    return images


def build_image_source_map(
    train_input_dir: Path,
    train_gt_dir: Path,
    val_input_dir: Optional[Path] = None,
    val_gt_dir: Optional[Path] = None,
    test_input_dir: Optional[Path] = None,
    test_gt_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Merge all provided directories into one image pool.

    Returns:
        source_map[image_name] = {
            "input_dir": ...,
            "gt_dir": ...,
            "source_split": "train"/"val"/"test"
        }

    If duplicated names appear across directories, later ones are skipped.
    """
    source_map: Dict[str, Dict[str, Path]] = {}

    dir_specs = [
        ("train", train_input_dir, train_gt_dir),
        ("val", val_input_dir, val_gt_dir),
        ("test", test_input_dir, test_gt_dir),
    ]

    for split_name, input_dir, gt_dir in dir_specs:
        if input_dir is None or gt_dir is None:
            continue

        names = list_images_from_dir(input_dir, gt_dir)
        for name in names:
            if name in source_map:
                print(f"[WARN] Duplicate image name {name} found in multiple dirs, keeping first occurrence.")
                continue
            source_map[name] = {
                "input_dir": Path(input_dir),
                "gt_dir": Path(gt_dir),
                "source_split": split_name,
            }

    return source_map


def make_random_split(images: List[str], seed=42, ratios=(0.8, 0.1, 0.1)):
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


def save_split_json(out_path: Path, seed: int, train, val, test, source_map: Dict[str, Dict[str, Path]]):
    def _decorate(names: List[str]):
        out = []
        for n in names:
            src = source_map.get(n, {})
            out.append({
                "name": n,
                "source_split": src.get("source_split", "unknown"),
                "input_dir": str(src.get("input_dir", "")),
                "gt_dir": str(src.get("gt_dir", "")),
            })
        return out

    obj = {
        "seed": seed,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "counts": {
            "total": len(train) + len(val) + len(test),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
        "train_detailed": _decorate(train),
        "val_detailed": _decorate(val),
        "test_detailed": _decorate(test),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# Processing
# -------------------------

def process_one_image(
    input_dir: Path,
    gt_dir: Path,
    image_name: str,
    out_root: Path,
    split_name: str,
    out_hw: Tuple[int, int],
    sdf_clip: float,
) -> int:
    """
    Process a single ISIC image:
      - load image / GT
      - resize
      - image normalize to [-1,1] by x/255*2-1
      - GT -> onehot
      - onehot -> sdf
      - save

    Return 1 if saved.
    """
    img_path = input_dir / f"{image_name}.jpg"
    gt_path = gt_dir / f"{image_name}_segmentation.png"

    if not img_path.exists() or not gt_path.exists():
        print(f"[WARN] Missing file for {image_name}, skip.")
        return 0

    # Load image with OpenCV (BGR -> RGB)
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Failed to read image: {img_path}")
        return 0

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(f"[WARN] Unsupported image shape {img.shape} for {img_path}, skip.")
        return 0

    img = img.astype(np.float32)

    # Load GT with OpenCV
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    if gt is None:
        print(f"[WARN] Failed to read GT: {gt_path}")
        return 0

    if gt.ndim == 3:
        gt = gt[..., 0]
    gt = (gt > 0).astype(np.uint8)

    # Resize
    img = resize_image(img, out_hw=out_hw, is_mask=False)
    gt = resize_image(gt, out_hw=out_hw, is_mask=True)
    gt = (gt > 0.5).astype(np.uint8)

    # Normalize image to [-1,1]
    img_norm = normalize_to_minus1_1_from_255(img)

    # One-hot and SDF (no background)
    onehot = to_onehot_isic(gt, num_classes=2, no_bg=True)
    sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip, normalize=True)

    # Save (channel-first)
    sample_out_dir = out_root / split_name / image_name
    _ensure_dirs(sample_out_dir)

    img_norm = np.transpose(img_norm, (2, 0, 1))  # (H,W,3) -> (3,H,W)

    np.save(sample_out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(sample_out_dir / "onehot.npy", onehot.astype(np.uint8))
    np.save(sample_out_dir / "sdf.npy", sdf.astype(np.float32))

    return 1


def run(
    train_input_dir: Path,
    train_gt_dir: Path,
    out_root: Path,
    size: int,
    sdf_clip: float,
    seed: int,
    split_json: Optional[Path],
    val_input_dir: Optional[Path] = None,
    val_gt_dir: Optional[Path] = None,
    test_input_dir: Optional[Path] = None,
    test_gt_dir: Optional[Path] = None,
):
    out_root = Path(out_root)
    out_hw = (size, size)

    # Merge all provided dirs into one pool, then split 8:1:1
    source_map = build_image_source_map(
        train_input_dir=train_input_dir,
        train_gt_dir=train_gt_dir,
        val_input_dir=val_input_dir,
        val_gt_dir=val_gt_dir,
        test_input_dir=test_input_dir,
        test_gt_dir=test_gt_dir,
    )

    all_images = sorted(source_map.keys())

    if len(all_images) == 0:
        print("No valid images found. Exiting.")
        return

    if split_json is not None and Path(split_json).exists():
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train = split["train"]
        val = split["val"]
        test = split["test"]
        seed_used = split.get("seed", seed)

        valid_names = set(all_images)
        train = [x for x in train if x in valid_names]
        val = [x for x in val if x in valid_names]
        test = [x for x in test if x in valid_names]

        used = set(train) | set(val) | set(test)
        missing = [x for x in all_images if x not in used]
        if len(missing) > 0:
            print(f"[WARN] {len(missing)} images were not present in split_json; they will be ignored.")
    else:
        train, val, test = make_random_split(all_images, seed=seed, ratios=(0.8, 0.1, 0.1))
        seed_used = seed
        save_split_json(out_root / "split.json", seed_used, train, val, test, source_map)

    total_samples = len(train) + len(val) + len(test)

    if not (split_json is not None and Path(split_json).exists()):
        print(f"Merged all provided files, then random split 8:1:1 -> {len(train)} train, {len(val)} val, {len(test)} test")
    else:
        print(f"Loaded split_json -> {len(train)} train, {len(val)} val, {len(test)} test")

    total_saved = 0
    total_skipped = 0

    def process_split(split_name, images):
        nonlocal total_saved, total_skipped
        for name in images:
            src = source_map.get(name, None)
            if src is None:
                print(f"[WARN] Missing source mapping for {name}, skip.")
                total_skipped += 1
                continue

            n = process_one_image(
                input_dir=Path(src["input_dir"]),
                gt_dir=Path(src["gt_dir"]),
                image_name=name,
                out_root=out_root,
                split_name=split_name,
                out_hw=out_hw,
                sdf_clip=sdf_clip,
            )
            if n > 0:
                total_saved += n
            else:
                total_skipped += 1

    process_split("train", train)
    process_split("val", val)
    process_split("test", test)

    if split_json is not None and Path(split_json).exists():
        save_split_json(out_root / "split.json", seed_used, train, val, test, source_map)

    print(f"Done. Saved {total_saved} images from {total_samples} total in split. Skipped {total_skipped} samples.")
    print(f"Split saved at: {out_root / 'split.json'}")
    print("Image normalization: x / 255.0 * 2.0 - 1.0")
    print("Resize backend: OpenCV")
    print("Saved onehot/sdf WITHOUT background channel.")


def main():
    ap = argparse.ArgumentParser()

    # backward compatibility
    ap.add_argument("--input_dir", type=str, default=None,
                    help="[Deprecated] Directory containing input images (.jpg); use --train_input_dir instead.")
    ap.add_argument("--gt_dir", type=str, default=None,
                    help="[Deprecated] Directory containing ground truth segmentations; use --train_gt_dir instead.")

    ap.add_argument("--train_input_dir", type=str, default='./ISICdataset/ISIC2018_Task1-2_Training_Input',
                    help="Directory containing training input images (.jpg)")
    ap.add_argument("--train_gt_dir", type=str, default='./ISICdataset/ISIC2018_Task1_Training_GroundTruth',
                    help="Directory containing training ground truth segmentations (_segmentation.png)")
    ap.add_argument("--val_input_dir", type=str, default='./ISICdataset/ISIC2018_Task1-2_Validation_Input',
                    help="Directory containing validation input images (.jpg)")
    ap.add_argument("--val_gt_dir", type=str, default='./ISICdataset/ISIC2018_Task1_Validation_GroundTruth',
                    help="Directory containing validation ground truth segmentations")
    ap.add_argument("--test_input_dir", type=str, default='./ISICdataset/ISIC2018_Task1-2_Test_Input',
                    help="Directory containing test input images (.jpg)")
    ap.add_argument("--test_gt_dir", type=str, default='./ISICdataset/ISIC2018_Task1_Test_GroundTruth',
                    help="Directory containing test ground truth segmentations")

    ap.add_argument("--out_root", type=str, default="./dataset/ISIC/SDF",
                    help="Output root containing train/val/test + split.json")
    ap.add_argument("--size", type=int, default=256, help="Output image size (square)")
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip distance")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--split_json", type=str, default=None,
                    help="Optional existing split.json to reuse")

    args = ap.parse_args()

    if args.train_input_dir is None and args.input_dir is not None:
        args.train_input_dir = args.input_dir
    if args.train_gt_dir is None and args.gt_dir is not None:
        args.train_gt_dir = args.gt_dir

    if args.train_input_dir is None or args.train_gt_dir is None:
        ap.error("Must provide at least --train_input_dir and --train_gt_dir (or legacy --input_dir/--gt_dir).")

    run(
        train_input_dir=Path(args.train_input_dir),
        train_gt_dir=Path(args.train_gt_dir),
        out_root=Path(args.out_root),
        size=args.size,
        sdf_clip=args.clip,
        seed=args.seed,
        split_json=Path(args.split_json) if args.split_json else None,
        val_input_dir=Path(args.val_input_dir) if args.val_input_dir else None,
        val_gt_dir=Path(args.val_gt_dir) if args.val_gt_dir else None,
        test_input_dir=Path(args.test_input_dir) if args.test_input_dir else None,
        test_gt_dir=Path(args.test_gt_dir) if args.test_gt_dir else None,
    )


if __name__ == "__main__":
    main()