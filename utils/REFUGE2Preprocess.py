from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


# ============================================================
# Utils (pad/crop + normalize)
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def center_pad_or_crop_hw(x: np.ndarray, out_hw=(256, 256), pad_value=0):
    """
    x: (H,W) or (H,W,1)
    return: (out_h,out_w) or (out_h,out_w,1)
    """
    if x.ndim == 2:
        x3 = x[:, :, None]
        squeeze_back = True
    elif x.ndim == 3 and x.shape[2] == 1:
        x3 = x
        squeeze_back = False
    else:
        raise ValueError(f"Expected (H,W) or (H,W,1), got {x.shape}")

    H, W, C = x3.shape
    out_h, out_w = out_hw

    # pad
    pad_h = max(0, out_h - H)
    pad_w = max(0, out_w - W)
    if pad_h > 0 or pad_w > 0:
        pt = pad_h // 2
        pb = pad_h - pt
        pl = pad_w // 2
        pr = pad_w - pl
        x3 = np.pad(
            x3,
            ((pt, pb), (pl, pr), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
        H, W, _ = x3.shape

    # crop
    if H > out_h:
        top = (H - out_h) // 2
        x3 = x3[top: top + out_h, :, :]
    if W > out_w:
        left = (W - out_w) // 2
        x3 = x3[:, left: left + out_w, :]

    if squeeze_back:
        return x3[:, :, 0]
    return x3


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


# ============================================================
# Mask -> onehot (3 channels: 0 / 128 / 255)
# ============================================================
def mask_to_onehot_3(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(np.uint8, copy=False)

    c1 = (m == 0).astype(np.uint8)      # 类别1
    c2 = (m == 128).astype(np.uint8)    # 类别2
    bg = (m == 255).astype(np.uint8)    # 背景

    # 如果出现其他值，建议当背景或直接报错
    other = ~((m == 0) | (m == 128) | (m == 255))
    if np.any(other):
        bg = (bg | other.astype(np.uint8))

    return np.stack([bg, c1, c2], axis=0)  # (3,H,W)


# ============================================================
# Onehot -> SDF (per channel)
# ============================================================
def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    onehot: (C,H,W) uint8/bool
    sdf: (C,H,W) float32
      inside mask  -> negative distance
      outside mask -> positive distance
      boundary -> 0
    Then clip to [-clip, clip], and if normalize: divide by clip -> [-1,1]
    """
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
    """Read image as grayscale float32 (H,W)."""
    with Image.open(img_path) as im:
        im = im.convert("L")
        return np.asarray(im, dtype=np.float32)


def read_mask_uint8(mask_path: Path) -> np.ndarray:
    """
    Read mask as uint8 (H,W), keep values like 0/128/255.
    Works for bmp/jpg/png...
    """
    with Image.open(mask_path) as im:
        im = im.convert("L")  # keep 0/128/255 coding
        return np.asarray(im, dtype=np.uint8)


def save_mask_vis_png(mask2d: np.ndarray, out_path: Path):
    """Save mask visualization png with original coding (0/128/255)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask2d.astype(np.uint8)).save(out_path)


# ============================================================
# Mask file resolver (NEW)
# ============================================================
_MASK_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_mask_file(mask_dir: Path, stem: str) -> Optional[Path]:
    """
    In mask_dir, find exactly one file whose name is {stem}{ext}, ext in _MASK_EXTS.
    Return Path if found, else None.
    If multiple found, raise error to avoid wrong pairing.
    """
    hits: List[Path] = []
    for ext in _MASK_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            hits.append(p)

    if len(hits) == 0:
        return None
    if len(hits) > 1:
        # "对应的一定只有一种" -> if violated, stop loudly
        raise RuntimeError(f"Multiple mask files found for {stem}: {[h.name for h in hits]}")
    return hits[0]


# ============================================================
# Processing
# ============================================================
def process_one_sample(
    img_path: Path,
    mask_path: Path,
    out_dir: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    save_png_mask: bool,
    verbose: bool,
) -> bool:
    """
    Output:
      out_dir/
        img.npy      (H,W) float32 in [-1,1]
        onehot.npy   (3,H,W) uint8  [bg,128,255]
        sdf.npy      (3,H,W) float32
        mask_vis.png (optional, values 0/128/255)
    """
    if not img_path.exists() or not mask_path.exists():
        if verbose:
            print(f"[Skip] missing pair: {img_path.name} / {mask_path.name}")
        return False

    img = read_image_gray_float(img_path)   # (H,W)
    msk = read_mask_uint8(mask_path)        # (H,W)

    # pad/crop
    out_hw = (size, size)
    img2d = center_pad_or_crop_hw(img, out_hw=out_hw, pad_value=0.0)
    msk2d = center_pad_or_crop_hw(msk, out_hw=out_hw, pad_value=0).astype(np.uint8)

    # normalize image [-1,1]
    img_norm = minmax01_single(img2d, clip_percentiles=(p_lo, p_hi))

    # onehot: [bg, 128, 255]
    onehot = mask_to_onehot_3(msk2d).astype(np.uint8)

    # sdf per channel
    sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip, normalize=sdf_normalize).astype(np.float32)

    _ensure_dirs(out_dir)
    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot)
    np.save(out_dir / "sdf.npy", sdf)

    if save_png_mask:
        save_mask_vis_png(msk2d, out_dir / "mask_vis.png")

    if verbose:
        uniq = np.unique(msk2d)
        print(f"[OK] {out_dir.name}: img={img_path.name}, mask={mask_path.name}, mask_vals={uniq.tolist()}")

    return True


def process_split(
    split_dir: Path,
    out_split_dir: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    save_png_mask: bool,
    verbose: bool,
) -> int:
    """
    split_dir/
      images/*.(jpg/png/...)
      mask/*.(bmp/jpg/png/...)  (same stem as image)
    """
    img_dir = split_dir / "images"
    msk_dir = split_dir / "mask"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {img_dir}")
    if not msk_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {msk_dir}")

    # images can be multiple exts
    img_files: List[Path] = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        img_files += list(img_dir.glob(f"*{ext}"))
    img_files = sorted(img_files)

    if len(img_files) == 0:
        if verbose:
            print(f"[Warn] no images found in {img_dir}")
        return 0

    ok = 0
    for img_path in img_files:
        stem = img_path.stem  # g0001
        mask_path = find_mask_file(msk_dir, stem)

        if mask_path is None:
            if verbose:
                print(f"[Skip] missing mask for {stem}: looked for {stem}{{{','.join(_MASK_EXTS)}}} in {msk_dir}")
            continue

        out_dir = out_split_dir / stem
        ok += int(
            process_one_sample(
                img_path=img_path,
                mask_path=mask_path,
                out_dir=out_dir,
                size=size,
                p_lo=p_lo,
                p_hi=p_hi,
                sdf_clip=sdf_clip,
                sdf_normalize=sdf_normalize,
                save_png_mask=save_png_mask,
                verbose=verbose,
            )
        )

    return ok


def run(
    dataset_root: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    save_png_mask: bool,
    verbose: bool,
):
    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    total = 0
    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            if verbose:
                print(f"[Skip] split not found: {split_dir}")
            continue

        out_split_dir = out_root / split
        _ensure_dirs(out_split_dir)

        n = process_split(
            split_dir=split_dir,
            out_split_dir=out_split_dir,
            size=size,
            p_lo=p_lo,
            p_hi=p_hi,
            sdf_clip=sdf_clip,
            sdf_normalize=sdf_normalize,
            save_png_mask=save_png_mask,
            verbose=verbose,
        )
        total += n
        if verbose:
            print(f"[Split] {split}: processed {n} samples -> {out_split_dir}")

    if verbose:
        print("\n====================")
        print(f"Done. Total samples processed: {total}")
        print(f"Output root: {out_root}")
        print("====================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        default="./REFUGE2dataset",
        help="数据集根目录，包含 train/ val/ test/ 三个文件夹；每个 split 下有 images/ 与 mask/。",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="./dataset/REFUGE2/SDF",
        help="输出根目录，将生成 out_root/train|val|test/ 下每个样本一个文件夹",
    )
    ap.add_argument("--size", type=int, default=256, help="pad/crop 到固定尺寸 size x size")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip 距离")
    ap.add_argument("--sdf_no_norm", action="store_true", help="不把 SDF 归一化到 [-1,1]（只 clip）")
    ap.add_argument("--save_png_mask", action="store_true", help="额外保存 mask_vis.png（0/128/255）方便检查")
    ap.add_argument("--quiet", action="store_true", help="不打印日志")
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
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()