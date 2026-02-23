from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


# ============================================================
# Utils (normalize + bbox + random crop + optional downscale)
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def minmax01_single(
    x: np.ndarray, eps=1e-8, clip_percentiles=(1.0, 99.0)
) -> np.ndarray:
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


def bbox_from_mask(bin_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    bin_mask: bool array
    return (x0,y0,x1,y1) inclusive
    """
    ys, xs = np.where(bin_mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def resize_keep_aspect(
    img2d: np.ndarray,
    msk2d: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    等比例缩放 image 和 mask
    - image: bilinear (float32)
    - mask : nearest (uint8)
    """
    if scale <= 0:
        raise ValueError("scale must be > 0")

    H, W = img2d.shape
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))

    # image
    im = Image.fromarray(img2d.astype(np.float32), mode="F")
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    img_rs = np.asarray(im, dtype=np.float32)

    # mask
    mm = Image.fromarray(msk2d.astype(np.uint8), mode="L")
    mm = mm.resize((new_w, new_h), resample=Image.NEAREST)
    msk_rs = np.asarray(mm, dtype=np.uint8)

    return img_rs, msk_rs


def random_crop_containing_bbox(
    img: np.ndarray,
    msk: np.ndarray,
    crop_size: int,
    bbox_xyxy: Tuple[int, int, int, int],
    rng: np.random.Generator,
    max_tries: int = 50,
    disc_value: int = 128,
    require_disc: bool = False,
    min_disc_ratio: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    在原图上随机裁 crop_size x crop_size，且裁剪窗口必须完整包含 bbox_xyxy。
    可选：要求裁剪内 disc 覆盖率 >= min_disc_ratio。
    """
    H, W = img.shape
    s = int(crop_size)

    if s > H or s > W:
        raise ValueError(f"crop_size {s} larger than image {H}x{W}")

    x0, y0, x1, y1 = bbox_xyxy
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw > s or bh > s:
        raise ValueError(f"bbox larger than crop_size: bbox=({x0},{y0},{x1},{y1}), crop={s}")

    # 合法 left/top 范围（保证窗口覆盖 bbox）
    left_min = max(0, x1 - (s - 1))
    left_max = min(x0, W - s)
    top_min = max(0, y1 - (s - 1))
    top_max = min(y0, H - s)

    if left_min > left_max or top_min > top_max:
        raise RuntimeError("No feasible crop position that contains the bbox. Check inputs.")

    best = None
    best_ratio = -1.0

    for _ in range(max_tries):
        left = int(rng.integers(left_min, left_max + 1))
        top = int(rng.integers(top_min, top_max + 1))

        img_crop = img[top:top + s, left:left + s]
        msk_crop = msk[top:top + s, left:left + s]

        if not require_disc:
            return img_crop, msk_crop, (top, left)

        disc_ratio = float((msk_crop == disc_value).mean())
        if disc_ratio >= min_disc_ratio:
            return img_crop, msk_crop, (top, left)

        if disc_ratio > best_ratio:
            best_ratio = disc_ratio
            best = (img_crop, msk_crop, (top, left))

    if best is not None:
        return best

    raise RuntimeError("Failed to sample crop.")


# ============================================================
# Mask -> onehot (YOUR coding: 0=cup, 128=disc, 255=bg)
# cup-only => [bg, cup]
# ============================================================
def mask_to_onehot_cup(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(np.uint8, copy=False)
    cup = (m == 0).astype(np.uint8)
    bg = (cup == 0).astype(np.uint8)
    return np.stack([bg, cup], axis=0)  # (2,H,W)


# ============================================================
# Onehot -> SDF (per channel)
# ============================================================
def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    if onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {onehot.shape}")
    if normalize and clip <= 0:
        raise ValueError("clip must be > 0 when normalize=True")

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
            sdf_c = sdf_c / (float(clip) + eps)

        sdf[c] = sdf_c

    return sdf


# ============================================================
# IO
# ============================================================
def read_image_gray_float(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("L")
        return np.asarray(im, dtype=np.float32)


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
    size: int,               # crop size (and final output) e.g. 256
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    save_png_mask: bool,
    deterministic: bool,
    seed: int,
    require_disc: bool,
    min_disc_ratio: float,
    max_tries: int,
    bbox_margin: int,
    verbose: bool,
) -> bool:
    if not img_path.exists() or not mask_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {img_path} / {mask_path}")
        return False

    img = read_image_gray_float(img_path)
    msk = read_mask_uint8(mask_path)

    # RNG（可复现：对每个 sample_id 固定随机）
    if deterministic:
        sid = (str(img_path) + "|" + sample_id).encode("utf-8")
        sample_seed = (zlib.crc32(sid) + int(seed)) & 0xFFFFFFFF
        rng = np.random.default_rng(sample_seed)
    else:
        rng = np.random.default_rng()

    # cup bbox（必须存在）
    cup_bb = bbox_from_mask(msk == 0)
    if cup_bb is None:
        if verbose:
            print(f"[Skip] no cup pixels in {mask_path.name}")
        return False

    # 若 bbox 超过 crop_size，则等比例缩小整图/整mask，再裁
    x0, y0, x1, y1 = cup_bb
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    maxb = max(bw, bh)

    if maxb > int(size):
        target = max(8, int(size) - int(bbox_margin))
        scale = target / float(maxb)

        if verbose:
            print(f"[Warn] {sample_id}: cup bbox {bw}x{bh} > {size}, resize whole image by {scale:.5f}")

        img, msk = resize_keep_aspect(img, msk, scale=scale)

        # 重新计算 bbox
        cup_bb = bbox_from_mask(msk == 0)
        if cup_bb is None:
            if verbose:
                print(f"[Skip] no cup after resize in {mask_path.name}")
            return False

    # 随机裁 256x256，必须包含 cup bbox（可选：disc覆盖率约束）
    img2d, msk2d, (top, left) = random_crop_containing_bbox(
        img=img,
        msk=msk,
        crop_size=int(size),
        bbox_xyxy=cup_bb,
        rng=rng,
        max_tries=int(max_tries),
        disc_value=128,
        require_disc=bool(require_disc),
        min_disc_ratio=float(min_disc_ratio),
    )

    # keep only {0,128,255}（保险）
    msk2d = np.where(msk2d == 0, 0, np.where(msk2d == 128, 128, 255)).astype(np.uint8)

    # normalize image [-1,1]
    img_norm = minmax01_single(img2d, clip_percentiles=(p_lo, p_hi))

    # cup-only onehot + sdf
    onehot = mask_to_onehot_cup(msk2d).astype(np.uint8)  # (2,H,W)
    sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip, normalize=sdf_normalize).astype(np.float32)

    _ensure_dirs(out_dir)
    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot)
    np.save(out_dir / "sdf.npy", sdf)

    if save_png_mask:
        cup_vis = (msk2d == 0).astype(np.uint8) * 255
        save_mask_vis_png(cup_vis, out_dir / "mask_vis.png")

    if verbose:
        print(f"[OK] {out_dir.name}: crop=(top={top}, left={left}), img={img_path.name}, mask={mask_path.name}")

    return True


def run(
    dataset_root: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    save_png_mask: bool,
    deterministic: bool,
    seed: int,
    split_json: Optional[Path],
    require_disc: bool,
    min_disc_ratio: float,
    max_tries: int,
    bbox_margin: int,
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
                    p_lo=p_lo,
                    p_hi=p_hi,
                    sdf_clip=sdf_clip,
                    sdf_normalize=sdf_normalize,
                    save_png_mask=save_png_mask,
                    deterministic=deterministic,
                    seed=seed,
                    require_disc=require_disc,
                    min_disc_ratio=min_disc_ratio,
                    max_tries=max_tries,
                    bbox_margin=bbox_margin,
                    verbose=verbose,
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
    ap.add_argument("--out_root", type=str, default="./dataset/REFUGE2_cup/SDF")

    ap.add_argument("--size", type=int, default=256, help="随机裁剪输出尺寸（默认256）")

    ap.add_argument("--p_lo", type=float, default=1.0)
    ap.add_argument("--p_hi", type=float, default=99.0)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--sdf_no_norm", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--split_json", type=str, default=None)

    # optional constraint to keep crop mostly within disc context
    ap.add_argument("--require_disc", action="store_true", help="裁剪要求 disc 覆盖率 >= min_disc_ratio")
    ap.add_argument("--min_disc_ratio", type=float, default=0.6)
    ap.add_argument("--max_tries", type=int, default=50)

    # when bbox too large, downscale whole image so bbox fits; keep some margin
    ap.add_argument("--bbox_margin", type=int, default=4, help="bbox 过大时，缩放到 (size-bbox_margin) 留余量")

    ap.add_argument("--save_png_mask", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(
        dataset_root=Path(args.dataset_root),
        out_root=Path(args.out_root),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        sdf_clip=float(args.clip),
        sdf_normalize=(not args.sdf_no_norm),
        save_png_mask=bool(args.save_png_mask),
        deterministic=bool(args.deterministic),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        require_disc=bool(args.require_disc),
        min_disc_ratio=float(args.min_disc_ratio),
        max_tries=int(args.max_tries),
        bbox_margin=int(args.bbox_margin),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()