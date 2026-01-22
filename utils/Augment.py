# -*- coding: utf-8 -*-
"""
ACDCAugment.py
Offline augmentation for extracted ACDC ED samples.

UPDATED (per your request):
- Input pairs are found by searching for:
    img.npy
    sdf.npy
  under --in_root recursively.
- No longer depends on filename prefixes like *_img.npy or *_gt_sdf.npy.

Expected input under --in_root (any nested structure is OK):
  .../img.npy      : (H,W) float32 (ideally in [0,1] or [-1,1], both OK)
  .../sdf.npy      : (C,H,W) float32 (C=4 by default; can be any C)

Output under --out_root:
  000000/
    image.npy
    sdf.npy
    meta.json
  000001/
    ...

Augmentations:
  Geometry (applied to image AND sdf):
    - random hflip / vflip
    - random translation (pixels)
    - random scaling
  Intensity (applied to image only):
    - brightness shift
    - contrast scaling
    - gamma
    - gaussian noise
    - gaussian blur

Usage:
  python ACDCAugment.py --in_root ./dataset/ACDC/SDF/train --out_root ./dataset/ACDC/Aug/train --aug_per_sample 2 --seed 42

NOTE:
- If your SDF was normalized to [-1,1] in preprocessing, you should set:
    --sdf_cval 1.0
  instead of the default 10.0, to keep padding scale consistent.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
from typing import List, Tuple, Dict, Any

from scipy.ndimage import affine_transform, gaussian_filter


# -------------------------
# IO helpers
# -------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_samples(in_root: Path) -> List[Tuple[Path, Path]]:
    """
    Find pairs (img.npy, sdf.npy) under in_root recursively.

    Expect in ANY subdirectory:
      img.npy
      sdf.npy
    """
    in_root = Path(in_root)
    pairs: List[Tuple[Path, Path]] = []

    for img_p in sorted(in_root.rglob("img.npy")):
        sdf_p = img_p.parent / "sdf.npy"
        if sdf_p.exists():
            pairs.append((img_p, sdf_p))

    return pairs


def _load_check(img_path: Path, sdf_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = np.load(img_path)  # (H,W)
    sdf = np.load(sdf_path)  # (C,H,W)
    if img.ndim != 2:
        raise ValueError(f"Image must be (H,W), got {img.shape} from {img_path}")
    if sdf.ndim != 3:
        raise ValueError(f"SDF must be (C,H,W), got {sdf.shape} from {sdf_path}")
    if sdf.shape[1:] != img.shape:
        raise ValueError(f"Spatial mismatch img {img.shape} vs sdf {sdf.shape} for {img_path}")
    return img.astype(np.float32, copy=False), sdf.astype(np.float32, copy=False)


# -------------------------
# Augmentation primitives
# -------------------------

def _random_bool(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


def _hflip(x: np.ndarray) -> np.ndarray:
    # (H,W) or (C,H,W)
    if x.ndim == 2:
        return x[:, ::-1]
    return x[:, :, ::-1]


def _vflip(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[::-1, :]
    return x[:, ::-1, :]


def _build_affine_2d(
    H: int,
    W: int,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build affine transform for scipy.ndimage.affine_transform (NO ROTATION).
    affine_transform maps output coords -> input coords using matrix A and offset b:
        input_coord = A @ output_coord + offset

    Apply scale + translation around image center (no rotation).

    Returns (A_yx, offset_yx).
    """
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    inv_s = 1.0 / max(scale, 1e-8)

    # In (x,y): A = inv_s * I
    A_xy = np.array([[inv_s, 0.0],
                     [0.0, inv_s]], dtype=np.float32)

    # Convert to (y,x) for scipy: (row,col)
    A_yx = np.array([[A_xy[1, 1], A_xy[1, 0]],
                     [A_xy[0, 1], A_xy[0, 0]]], dtype=np.float32)

    # offset in (x,y):
    offset_xy = -A_xy @ np.array([cx + tx, cy + ty], dtype=np.float32) + np.array([cx, cy], dtype=np.float32)

    # to (y,x)
    offset_yx = np.array([offset_xy[1], offset_xy[0]], dtype=np.float32)
    return A_yx, offset_yx


def _apply_affine_to_image(
    img: np.ndarray,
    A_yx: np.ndarray,
    offset_yx: np.ndarray,
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """img: (H,W)"""
    return affine_transform(
        img,
        matrix=A_yx,
        offset=offset_yx,
        output_shape=img.shape,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=(order > 1),
    ).astype(np.float32, copy=False)


def _apply_affine_to_sdf(
    sdf: np.ndarray,
    A_yx: np.ndarray,
    offset_yx: np.ndarray,
    order: int,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """sdf: (C,H,W)"""
    C, H, W = sdf.shape
    out = np.empty_like(sdf, dtype=np.float32)
    for c in range(C):
        out[c] = affine_transform(
            sdf[c],
            matrix=A_yx,
            offset=offset_yx,
            output_shape=(H, W),
            order=order,
            mode=mode,
            cval=cval,
            prefilter=(order > 1),
        ).astype(np.float32, copy=False)
    return out


# -------------------------
# Intensity transforms (image only)
# -------------------------

def _rand_brightness_contrast_gamma(
    img: np.ndarray,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    out = img.astype(np.float32, copy=False)

    if brightness > 0:
        delta = float(rng.uniform(-brightness, brightness))
        out = out + delta

    if contrast > 0:
        c = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
        m = float(out.mean())
        out = (out - m) * c + m

    if gamma > 1.0:
        g = float(rng.uniform(1.0 / gamma, gamma))
        minv = float(out.min())
        if minv < 0:
            out = out - minv
        out = np.power(np.clip(out, 0, None), g)
        if minv < 0:
            out = out + minv

    return out.astype(np.float32, copy=False)


def _rand_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    n = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return (img + n).astype(np.float32, copy=False)


def _rand_blur(img: np.ndarray, rng: np.random.Generator, sigma_max: float, p: float) -> np.ndarray:
    if sigma_max <= 0 or rng.random() >= p:
        return img
    s = float(rng.uniform(0.0, sigma_max))
    if s <= 1e-6:
        return img
    return gaussian_filter(img, sigma=s).astype(np.float32, copy=False)


# -------------------------
# Main augment function
# -------------------------

def augment_one(
    img: np.ndarray,
    sdf: np.ndarray,
    rng: np.random.Generator,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    img: (H,W) float32
    sdf: (C,H,W) float32
    Returns augmented (img, sdf, meta)
    """
    meta: Dict[str, Any] = {"geom": {}, "intensity": {}}

    # ---------- Flips ----------
    do_h = _random_bool(rng, cfg["p_hflip"])
    do_v = _random_bool(rng, cfg["p_vflip"])
    meta["geom"]["hflip"] = do_h
    meta["geom"]["vflip"] = do_v

    if do_h:
        img = _hflip(img)
        sdf = _hflip(sdf)
    if do_v:
        img = _vflip(img)
        sdf = _vflip(sdf)

    # ---------- Affine (scale/translate ONLY, no rotation) ----------
    H, W = img.shape
    scale = float(rng.uniform(1.0 - cfg["scale"], 1.0 + cfg["scale"])) if cfg["scale"] > 0 else 1.0
    tx = float(rng.uniform(-cfg["trans_px"], cfg["trans_px"])) if cfg["trans_px"] > 0 else 0.0
    ty = float(rng.uniform(-cfg["trans_px"], cfg["trans_px"])) if cfg["trans_px"] > 0 else 0.0

    meta["geom"]["scale"] = scale
    meta["geom"]["tx_px"] = tx
    meta["geom"]["ty_px"] = ty

    if (abs(scale - 1.0) > 1e-6) or (abs(tx) > 1e-6) or (abs(ty) > 1e-6):
        A_yx, offset_yx = _build_affine_2d(H, W, scale=scale, tx=tx, ty=ty)

        # image: order=1 (bilinear)
        img = _apply_affine_to_image(img, A_yx, offset_yx, order=1, mode="constant", cval=cfg["img_cval"])

        # sdf: order=1 (bilinear) for continuous maps
        sdf = _apply_affine_to_sdf(sdf, A_yx, offset_yx, order=1, mode="constant", cval=cfg["sdf_cval"])

    # ---------- Intensity (image only) ----------
    img = _rand_brightness_contrast_gamma(
        img,
        rng,
        brightness=cfg["brightness"],
        contrast=cfg["contrast"],
        gamma=cfg["gamma"],
    )
    meta["intensity"]["brightness_max"] = cfg["brightness"]
    meta["intensity"]["contrast_max"] = cfg["contrast"]
    meta["intensity"]["gamma_max"] = cfg["gamma"]

    img = _rand_blur(img, rng, sigma_max=cfg["blur_sigma_max"], p=cfg["p_blur"])
    meta["intensity"]["p_blur"] = cfg["p_blur"]
    meta["intensity"]["blur_sigma_max"] = cfg["blur_sigma_max"]

    img = _rand_noise(img, rng, sigma=cfg["noise_sigma"])
    meta["intensity"]["noise_sigma"] = cfg["noise_sigma"]

    # optional clamp
    if cfg["clamp_min"] is not None or cfg["clamp_max"] is not None:
        lo = -np.inf if cfg["clamp_min"] is None else float(cfg["clamp_min"])
        hi = np.inf if cfg["clamp_max"] is None else float(cfg["clamp_max"])
        img = np.clip(img, lo, hi).astype(np.float32)

    img = np.ascontiguousarray(img.astype(np.float32, copy=False))
    sdf = np.ascontiguousarray(sdf.astype(np.float32, copy=False))
    return img, sdf, meta


def augment_dataset(
    in_root: Path,
    out_root: Path,
    aug_per_sample: int,
    include_original: bool,
    seed: int,
    start_index: int,
    id_width: int,
    cfg: Dict[str, Any],
    overwrite: bool = False,
):
    in_root = Path(in_root)
    out_root = Path(out_root)
    _ensure_dir(out_root)

    pairs = _find_samples(in_root)
    if not pairs:
        raise RuntimeError(f"No paired img.npy + sdf.npy found under {in_root}.")

    rng = np.random.default_rng(seed)
    out_idx = start_index

    def _write_one(img_arr: np.ndarray, sdf_arr: np.ndarray, meta: Dict[str, Any]):
        nonlocal out_idx
        sample_dir = out_root / str(out_idx).zfill(id_width)

        if sample_dir.exists() and not overwrite:
            while (out_root / str(out_idx).zfill(id_width)).exists():
                out_idx += 1
            sample_dir = out_root / str(out_idx).zfill(id_width)

        _ensure_dir(sample_dir)
        np.save(sample_dir / "img.npy", img_arr.astype(np.float32, copy=False))
        np.save(sample_dir / "sdf.npy", sdf_arr.astype(np.float32, copy=False))
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        out_idx += 1

    for img_p, sdf_p in pairs:
        img, sdf = _load_check(img_p, sdf_p)

        if include_original:
            _write_one(
                img,
                sdf,
                meta={
                    "source_image": str(img_p),
                    "source_sdf": str(sdf_p),
                    "augment": "none",
                    "config": cfg,
                },
            )

        for _ in range(aug_per_sample):
            img_aug, sdf_aug, meta_aug = augment_one(img, sdf, rng, cfg)
            _write_one(
                img_aug,
                sdf_aug,
                meta={
                    "source_image": str(img_p),
                    "source_sdf": str(sdf_p),
                    "augment": meta_aug,
                    "config": cfg,
                },
            )

    print(
        f"Done. Found {len(pairs)} base samples. "
        f"Wrote {out_idx - start_index} samples to {out_root} "
        f"(include_original={include_original}, aug_per_sample={aug_per_sample})."
    )


# -------------------------
# Defaults (common-ish parameters)
# -------------------------

def build_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "p_hflip": float(args.p_hflip),
        "p_vflip": float(args.p_vflip),
        "scale": float(args.scale),
        "trans_px": float(args.trans),
        "img_cval": float(args.img_cval),
        "sdf_cval": float(args.sdf_cval),
        "brightness": float(args.brightness),
        "contrast": float(args.contrast),
        "gamma": float(args.gamma),
        "noise_sigma": float(args.noise),
        "p_blur": float(args.p_blur),
        "blur_sigma_max": float(args.blur_sigma_max),
        "clamp_min": float(args.clamp_min),
        "clamp_max": float(args.clamp_max),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="./dataset/DDMI/SDF/train", help="Input root containing img.npy & sdf.npy")
    ap.add_argument("--out_root", type=str, default="./dataset/DDMI/Aug/train", help="Output root; numeric folders 000000/")
    ap.add_argument("--aug_per_sample", type=int, default=2, help="Number of augmented variants per base sample")
    ap.add_argument("--no_original", action="store_true", help="Do not save the original sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--id_width", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output folders if exist (otherwise auto-find next id)")

    # ---- Geometry (NO ROTATION) ----
    ap.add_argument("--p_hflip", type=float, default=0.5)
    ap.add_argument("--p_vflip", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.10, help="Scale range (default 0.10)")
    ap.add_argument("--trans", type=float, default=8.0, help="Translation range in pixels (default 8)")

    # padding constants
    ap.add_argument("--img_cval", type=float, default=0.0)

    # IMPORTANT: if your SDF is normalized to [-1,1], set sdf_cval=1.0
    ap.add_argument("--sdf_cval", type=float, default=1.0)

    # ---- Intensity ----
    ap.add_argument("--brightness", type=float, default=0.08)
    ap.add_argument("--contrast", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=1.20)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--p_blur", type=float, default=0.2)
    ap.add_argument("--blur_sigma_max", type=float, default=0.8)

    # clamp
    ap.add_argument("--clamp_min", type=float, default=-1.0)
    ap.add_argument("--clamp_max", type=float, default=1.0)

    args = ap.parse_args()
    cfg = build_cfg_from_args(args)

    augment_dataset(
        in_root=Path(args.in_root),
        out_root=Path(args.out_root),
        aug_per_sample=args.aug_per_sample,
        include_original=not args.no_original,
        seed=args.seed,
        start_index=args.start_index,
        id_width=args.id_width,
        cfg=cfg,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()