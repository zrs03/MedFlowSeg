# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from PIL import Image


# ============================================================
# Utils (normalize + resize)
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize_rgb_to_minus1_1(x: np.ndarray) -> np.ndarray:
    """
    Normalize RGB image to [-1,1].
    x: (H,W,3) uint8 array
    return: (H,W,3) float32 in [-1,1]
    """
    x = x.astype(np.float32, copy=False)
    return (x / 255.0 * 2.0 - 1.0).astype(np.float32, copy=False)


def resize_to_fixed_hw(
    img: np.ndarray,
    msk: np.ndarray,
    out_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image+mask to fixed (H,W).
    - image: bilinear (RGB uint8)
    - mask : nearest (uint8)
    """
    out_h, out_w = int(out_hw[0]), int(out_hw[1])

    # image (RGB)
    im = Image.fromarray(img.astype(np.uint8), mode="RGB")
    im = im.resize((out_w, out_h), resample=Image.BILINEAR)
    img_rs = np.asarray(im, dtype=np.uint8)

    # mask (uint8)
    mm = Image.fromarray(msk.astype(np.uint8), mode="L")
    mm = mm.resize((out_w, out_h), resample=Image.NEAREST)
    msk_rs = np.asarray(mm, dtype=np.uint8)

    return img_rs, msk_rs


# ============================================================
# Mask -> single channel (disc only, no background)
# ============================================================
def mask_to_onehot_disc(mask2d: np.ndarray, disc_value: int = 128) -> np.ndarray:
    m = mask2d.astype(np.uint8, copy=False)
    disc = (m != np.uint8(255)).astype(np.uint8)
    # 直接扩展维度，返回单通道前景，形状为 (1, H, W)
    return np.expand_dims(disc, axis=0)


# ============================================================
# IO
# ============================================================
def read_image_rgb_uint8(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def read_mask_uint8(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as im:
        im = im.convert("L")
        return np.asarray(im, dtype=np.uint8)


def save_mask_vis_png(mask2d: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask2d.astype(np.uint8)).save(out_path)


# ============================================================
# File resolver
# ============================================================
_IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
_MASK_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_mask_file(mask_dir: Path, stem: str) -> Optional[Path]:
    hits: List[Path] = []
    for ext in _MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            hits.append(p)

    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"Multiple mask files found for {stem}: {[h.name for h in hits]}")
    return hits[0]


def list_all_pairs(dataset_root: Path, verbose: bool = False) -> List[Tuple[str, Path, Path]]:
    """
    汇总 dataset_root 下 train/val/test 全部样本，返回 (sample_id, img_path, mask_path)
    sample_id 默认 stem；若重名则加后缀避免覆盖。
    """
    dataset_root = Path(dataset_root)
    pairs: List[Tuple[str, Path, Path]] = []
    seen_ids: Dict[str, int] = {}

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        img_dir = split_dir / "images"
        msk_dir = split_dir / "mask"
        if not img_dir.exists() or not msk_dir.exists():
            continue

        img_files: List[Path] = []
        for ext in _IMG_EXTS:
            img_files += list(img_dir.glob(f"*{ext}"))
        img_files = sorted(img_files)

        for img_path in img_files:
            stem = img_path.stem
            mask_path = find_mask_file(msk_dir, stem)
            if mask_path is None:
                if verbose:
                    print(f"[Skip] missing mask for {split}/{stem}")
                continue

            sample_id = stem
            if sample_id in seen_ids:
                seen_ids[sample_id] += 1
                sample_id = f"{stem}__{seen_ids[stem]}"
            else:
                seen_ids[sample_id] = 0

            pairs.append((sample_id, img_path, mask_path))

    return pairs


def make_split_ids(
    sample_ids: List[str], seed: int = 42, ratios=(0.8, 0.1, 0.1)
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    ids = sample_ids.copy()
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test


def save_split_json(out_path: Path, seed: int, train: List[str], val: List[str], test: List[str]):
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
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Processing
# ============================================================
def process_one_sample(
    sample_id: str,
    img_path: Path,
    mask_path: Path,
    out_dir: Path,
    size: int,               # final output size, e.g. 256
    save_png_mask: bool,
    verbose: bool,
    disc_value: int = 128,
) -> bool:
    if not img_path.exists() or not mask_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {img_path} / {mask_path}")
        return False

    img = read_image_rgb_uint8(img_path)    # (H,W,3) uint8
    msk = read_mask_uint8(mask_path)        # (H,W) uint8

    # keep only {0,128,255}
    msk = np.where(msk == 0, 0, np.where(msk == disc_value, disc_value, 255)).astype(np.uint8)

    # resize full image+mask to size x size
    img_rs, msk_rs = resize_to_fixed_hw(img, msk, out_hw=(int(size), int(size)))

    # RGB normalize to [-1,1] and transpose to (3,H,W)
    img_norm = normalize_rgb_to_minus1_1(img_rs)     # (H,W,3) float32
    img_norm = np.transpose(img_norm, (2, 0, 1))     # -> (3,H,W)

    # disc-only onehot (1,H,W)
    onehot = mask_to_onehot_disc(msk_rs, disc_value=disc_value).astype(np.uint8)

    _ensure_dirs(out_dir)
    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot)

    if save_png_mask:
        disc_vis = (msk_rs == disc_value).astype(np.uint8) * 255
        save_mask_vis_png(disc_vis, out_dir / "mask_vis.png")

    if verbose:
        print(f"[OK] {out_dir.name}: resized={size}x{size}, img={img_path.name}, mask={mask_path.name}")

    return True


def run(
    dataset_root: Path,
    out_root: Path,
    size: int,
    save_png_mask: bool,
    seed: int,
    split_json: Optional[Path],
    verbose: bool,
):
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    pairs = list_all_pairs(dataset_root, verbose=verbose)
    if len(pairs) == 0:
        raise RuntimeError(f"No (image,mask) pairs found under {dataset_root}/*/images and mask.")

    id_to_pair: Dict[str, Tuple[Path, Path]] = {sid: (ip, mp) for sid, ip, mp in pairs}
    all_ids = list(id_to_pair.keys())

    split_json = Path(split_json) if split_json is not None else (out_root / "split.json")

    if split_json.exists():
        with open(split_json, "r", encoding="utf-8") as f:
            sp = json.load(f)
        train_ids = sp.get("train", [])
        val_ids = sp.get("val", [])
        test_ids = sp.get("test", [])
        if verbose:
            print(f"[Info] Reusing split: {split_json}")
    else:
        train_ids, val_ids, test_ids = make_split_ids(all_ids, seed=seed, ratios=(0.8, 0.1, 0.1))
        save_split_json(split_json, seed, train_ids, val_ids, test_ids)
        if verbose:
            print(f"[Info] Created split.json at: {split_json}")

    total = 0
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out_split_dir = out_root / split_name
        _ensure_dirs(out_split_dir)

        ok = 0
        for sid in ids:
            if sid not in id_to_pair:
                if verbose:
                    print(f"[Skip] split contains unknown id (not found in dataset pool): {sid}")
                continue

            img_path, mask_path = id_to_pair[sid]
            out_dir = out_split_dir / sid
            ok += int(
                process_one_sample(
                    sample_id=sid,
                    img_path=img_path,
                    mask_path=mask_path,
                    out_dir=out_dir,
                    size=size,
                    save_png_mask=save_png_mask,
                    verbose=verbose,
                    disc_value=128,
                )
            )

        total += ok
        if verbose:
            print(f"[Split] {split_name}: processed {ok} samples -> {out_split_dir}")

    if verbose:
        print("\n====================")
        print(f"Done. Total samples processed: {total}")
        print(f"Output root: {out_root}")
        print(f"Split json: {split_json}")
        print("====================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="./REFUGE2dataset")
    ap.add_argument("--out_root", type=str, default="./dataset/REFUGE2_disc/ONEHOT")

    ap.add_argument("--size", type=int, default=256, help="整图 resize 输出尺寸（默认256）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None)

    ap.add_argument("--save_png_mask", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(
        dataset_root=Path(args.dataset_root),
        out_root=Path(args.out_root),
        size=int(args.size),
        save_png_mask=bool(args.save_png_mask),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()