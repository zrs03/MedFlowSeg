# ISICPreprocess.py
#
# Preprocess ISIC2018 dataset for segmentation with SDF:
# 1) Load RGB image and binary GT (lesion segmentation)
# 2) Resize to fixed size
# 3) Normalize image by: x / 255.0 * 2.0 - 1.0
# 4) Convert GT to one-hot (no background by default) and then to per-channel SDF
# 5) Support both pre-defined split (via separate dirs) and random split
# 6) Save outputs to:
#    out_root/
#      split.json
#      train/ISIC_xxxx/
#         img.npy        (3,H,W) float32 in [-1,1]
#         onehot.npy     (1,H,W) uint8   # no_bg=True
#         sdf.npy        (1,H,W) float32 in [-1,1]
#      val/...
#      test/...

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Tuple, List, Optional

import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize


# -------------------------
# Utils
# -------------------------

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resize_image(x: np.ndarray, out_hw=(256, 256), preserve_range=True):
    """Resize (H,W) or (H,W,C) array to target out_hw."""
    orig_ndim = x.ndim
    if orig_ndim == 2:
        x = x[..., None]  # (H,W,1)

    x_resized = resize(
        x,
        (out_hw[0], out_hw[1], x.shape[2] if x.ndim == 3 else 1),
        preserve_range=preserve_range,
        mode="reflect",
        anti_aliasing=True,
    )

    if orig_ndim == 2:
        x_resized = x_resized[..., 0]
    return x_resized


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
        # Only foreground channel: lesion
        onehot = np.zeros((1, gt2d.shape[0], gt2d.shape[1]), dtype=np.uint8)
        onehot[0] = (gt2d == 1).astype(np.uint8)
    else:
        # Include background + lesion
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

    # Load image (RGB)
    img = imageio.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img = img.astype(np.float32)

    # Load GT (binary mask)
    gt = imageio.imread(gt_path)
    if gt.ndim == 3:
        gt = gt[..., 0]
    gt = (gt > 0).astype(np.uint8)

    # Resize
    img = resize_image(img, out_hw=out_hw)
    gt = resize_image(gt, out_hw=out_hw)
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

    # if val/test dirs are given, use predefined split
    use_predefined = val_input_dir is not None or test_input_dir is not None

    if use_predefined:
        if val_input_dir is None or val_gt_dir is None:
            raise ValueError("When using predefined split, both val_input_dir and val_gt_dir must be provided.")
        if test_input_dir is None or test_gt_dir is None:
            raise ValueError("When using predefined split, both test_input_dir and test_gt_dir must be provided.")

        train_images = list_images_from_dir(train_input_dir, train_gt_dir)
        val_images = list_images_from_dir(val_input_dir, val_gt_dir)
        test_images = list_images_from_dir(test_input_dir, test_gt_dir)

        train, val, test = train_images, val_images, test_images
        seed_used = seed
        save_split_json(out_root / "split.json", seed_used, train, val, test)

        total_samples = len(train) + len(val) + len(test)
        print(f"Pre-defined split: {len(train)} train, {len(val)} val, {len(test)} test")
    else:
        if split_json is not None and Path(split_json).exists():
            with open(split_json, "r", encoding="utf-8") as f:
                split = json.load(f)
            train, val, test = split["train"], split["val"], split["test"]
            seed_used = split.get("seed", seed)
        else:
            all_images = list_images_from_dir(train_input_dir, train_gt_dir)
            if not all_images:
                print("No valid images found. Exiting.")
                return
            train, val, test = make_random_split(all_images, seed=seed, ratios=(0.8, 0.1, 0.1))
            seed_used = seed
            save_split_json(out_root / "split.json", seed_used, train, val, test)

        total_samples = len(train) + len(val) + len(test)

    total_saved = 0
    total_skipped = 0

    def process_split(split_name, images, input_dir, gt_dir):
        nonlocal total_saved, total_skipped
        for name in images:
            n = process_one_image(
                input_dir=input_dir,
                gt_dir=gt_dir,
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

    if use_predefined:
        process_split("train", train, train_input_dir, train_gt_dir)
        process_split("val", val, val_input_dir, val_gt_dir)
        process_split("test", test, test_input_dir, test_gt_dir)
    else:
        process_split("train", train, train_input_dir, train_gt_dir)
        process_split("val", val, train_input_dir, train_gt_dir)
        process_split("test", test, train_input_dir, train_gt_dir)

    print(f"Done. Saved {total_saved} images from {total_samples} total in split. Skipped {total_skipped} samples.")
    print(f"Split saved at: {out_root / 'split.json'}")
    print("Image normalization: x / 255.0 * 2.0 - 1.0")
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
                    help="Optional existing split.json to reuse (only in random split mode)")

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