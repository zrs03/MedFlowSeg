# -*- coding: utf-8 -*-
"""
ACDCAugment.py
Offline augmentation for extracted 2D samples.

UPDATED:
- If img.npy is (H,W), auto convert to (1,H,W)
- Also supports img.npy as (C,H,W)
- onehot.npy is (C,H,W)
- Geometry augment: hflip/vflip + translate + scale + rotation
- After augmentation, regenerate sdf.npy FROM onehot.npy
- Supports no-background onehot safely

Expected input under --in_root:
  .../img.npy      : (H,W) or (C,H,W) float32
  .../onehot.npy   : (C,H,W)

Output under --out_root:
  0000/
    img.npy
    onehot.npy
    sdf.npy
    meta.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np
from typing import List, Tuple, Dict, Any

from scipy.ndimage import affine_transform, gaussian_filter, distance_transform_edt


# -------------------------
# IO helpers
# -------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_samples(in_root: Path) -> List[Tuple[Path, Path]]:
    in_root = Path(in_root)
    pairs: List[Tuple[Path, Path]] = []

    for img_p in sorted(in_root.rglob("img.npy")):
        onehot_p = img_p.parent / "onehot.npy"
        if onehot_p.exists():
            pairs.append((img_p, onehot_p))

    return pairs


def _load_check(img_path: Path, onehot_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = np.load(img_path)
    onehot = np.load(onehot_path)

    # img: (H,W) -> (1,H,W), or keep (C,H,W)
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        pass
    else:
        raise ValueError(f"Image must be (H,W) or (C,H,W), got {img.shape} from {img_path}")

    if onehot.ndim != 3:
        raise ValueError(f"onehot must be (C,H,W), got {onehot.shape} from {onehot_path}")

    if onehot.shape[1:] != img.shape[1:]:
        raise ValueError(f"Spatial mismatch img {img.shape} vs onehot {onehot.shape} for {img_path}")

    img = img.astype(np.float32, copy=False)

    if onehot.dtype == np.bool_:
        onehot_f = onehot.astype(np.float32)
    else:
        onehot_f = onehot.astype(np.float32, copy=False)

    return img, onehot_f


# -------------------------
# Augmentation primitives
# -------------------------
def _random_bool(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


def _hflip_img(x: np.ndarray) -> np.ndarray:
    # (C,H,W)
    return x[:, :, ::-1]


def _vflip_img(x: np.ndarray) -> np.ndarray:
    # (C,H,W)
    return x[:, ::-1, :]


def _hflip_chw(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1]


def _vflip_chw(x: np.ndarray) -> np.ndarray:
    return x[:, ::-1, :]


def _build_affine_2d_rot_scale_trans(
    H: int,
    W: int,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    rot_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    s = float(scale)
    s_inv = 1.0 / max(s, 1e-8)

    theta = math.radians(float(rot_deg))
    c = math.cos(theta)
    sn = math.sin(theta)

    R = np.array([[c, -sn],
                  [sn,  c]], dtype=np.float32)

    A_xy = (s_inv * R.T).astype(np.float32)

    t_xy = np.array([tx, ty], dtype=np.float32)
    c_xy = np.array([cx, cy], dtype=np.float32)
    offset_xy = c_xy - A_xy @ (c_xy + t_xy)

    A_yx = np.array([[A_xy[1, 1], A_xy[1, 0]],
                     [A_xy[0, 1], A_xy[0, 0]]], dtype=np.float32)
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
    """
    img: (C,H,W)
    Apply affine channel by channel.
    """
    if img.ndim != 3:
        raise ValueError(f"img must be (C,H,W), got {img.shape}")

    C, H, W = img.shape
    out = np.empty_like(img, dtype=np.float32)

    for c in range(C):
        out[c] = affine_transform(
            img[c],
            matrix=A_yx,
            offset=offset_yx,
            output_shape=(H, W),
            order=order,
            mode=mode,
            cval=cval,
            prefilter=(order > 1),
        ).astype(np.float32, copy=False)

    return out


def _apply_affine_to_onehot(
    onehot: np.ndarray,
    A_yx: np.ndarray,
    offset_yx: np.ndarray,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    onehot: (C,H,W)
    """
    C, H, W = onehot.shape
    out = np.empty_like(onehot, dtype=np.float32)
    for c in range(C):
        out[c] = affine_transform(
            onehot[c],
            matrix=A_yx,
            offset=offset_yx,
            output_shape=(H, W),
            order=order,
            mode=mode,
            cval=cval,
            prefilter=False,
        ).astype(np.float32, copy=False)
    return out


def _reonehot_argmax_no_bg(onehot_f: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if onehot_f.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {onehot_f.shape}")

    C, H, W = onehot_f.shape
    maxv = np.max(onehot_f, axis=0)
    lbl = np.argmax(onehot_f, axis=0)
    fg = (maxv >= float(thr))

    out = np.zeros((C, H, W), dtype=np.uint8)
    for c in range(C):
        out[c] = (fg & (lbl == c)).astype(np.uint8)
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
    """
    img: (C,H,W)
    intensity transforms applied per-channel
    """
    out = img.astype(np.float32, copy=False)

    if brightness > 0:
        delta = rng.uniform(-brightness, brightness, size=(out.shape[0], 1, 1)).astype(np.float32)
        out = out + delta

    if contrast > 0:
        cc = rng.uniform(1.0 - contrast, 1.0 + contrast, size=(out.shape[0], 1, 1)).astype(np.float32)
        m = out.mean(axis=(1, 2), keepdims=True)
        out = (out - m) * cc + m

    if gamma > 1.0:
        g = rng.uniform(1.0 / gamma, gamma, size=(out.shape[0], 1, 1)).astype(np.float32)
        minv = out.min(axis=(1, 2), keepdims=True)
        shift = np.where(minv < 0, -minv, 0.0).astype(np.float32)
        out = out + shift
        out = np.power(np.clip(out, 0, None), g)
        out = out - shift

    return out.astype(np.float32, copy=False)


def _rand_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    n = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return (img + n).astype(np.float32, copy=False)


def _rand_blur(img: np.ndarray, rng: np.random.Generator, sigma_max: float, p: float) -> np.ndarray:
    """
    img: (C,H,W), blur per-channel over spatial dims
    """
    if sigma_max <= 0 or rng.random() >= p:
        return img
    s = float(rng.uniform(0.0, sigma_max))
    if s <= 1e-6:
        return img

    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        out[c] = gaussian_filter(img[c], sigma=s).astype(np.float32, copy=False)
    return out


# -------------------------
# Onehot -> SDF
# -------------------------
def onehot_to_sdf_per_channel(
    onehot: np.ndarray,
    clip: float = 10.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    if onehot.ndim != 3:
        raise ValueError(f"Expected onehot (C,H,W), got {onehot.shape}")
    if clip <= 0:
        raise ValueError("clip must be > 0")

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
            sdf_c = sdf_c / (clip + eps)
        sdf[c] = sdf_c

    return sdf


# -------------------------
# Main augment function
# -------------------------
def augment_one(
    img: np.ndarray,
    onehot: np.ndarray,
    rng: np.random.Generator,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    img: (C,H,W) float32
    onehot: (C,H,W) float32
    """
    meta: Dict[str, Any] = {"geom": {}, "intensity": {}, "sdf": {}}

    do_h = _random_bool(rng, cfg["p_hflip"])
    do_v = _random_bool(rng, cfg["p_vflip"])
    meta["geom"]["hflip"] = do_h
    meta["geom"]["vflip"] = do_v

    if do_h:
        img = _hflip_img(img)
        onehot = _hflip_chw(onehot)
    if do_v:
        img = _vflip_img(img)
        onehot = _vflip_chw(onehot)

    _, H, W = img.shape

    scale = float(rng.uniform(1.0 - cfg["scale"], 1.0 + cfg["scale"])) if cfg["scale"] > 0 else 1.0
    tx = float(rng.uniform(-cfg["trans_px"], cfg["trans_px"])) if cfg["trans_px"] > 0 else 0.0
    ty = float(rng.uniform(-cfg["trans_px"], cfg["trans_px"])) if cfg["trans_px"] > 0 else 0.0
    rot = float(rng.uniform(-cfg["rot_deg"], cfg["rot_deg"])) if cfg["rot_deg"] > 0 else 0.0

    meta["geom"]["scale"] = scale
    meta["geom"]["tx_px"] = tx
    meta["geom"]["ty_px"] = ty
    meta["geom"]["rot_deg"] = rot

    if (abs(scale - 1.0) > 1e-6) or (abs(tx) > 1e-6) or (abs(ty) > 1e-6) or (abs(rot) > 1e-6):
        A_yx, offset_yx = _build_affine_2d_rot_scale_trans(H, W, scale=scale, tx=tx, ty=ty, rot_deg=rot)

        img = _apply_affine_to_image(img, A_yx, offset_yx, order=1, mode="constant", cval=cfg["img_cval"])
        onehot = _apply_affine_to_onehot(onehot, A_yx, offset_yx, order=0, mode="constant", cval=0.0)

    onehot_u8 = _reonehot_argmax_no_bg(onehot, thr=cfg["onehot_thr"])

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

    lo = cfg["clamp_min"]
    hi = cfg["clamp_max"]
    if lo is not None or hi is not None:
        lo2 = -np.inf if lo is None else float(lo)
        hi2 = np.inf if hi is None else float(hi)
        img = np.clip(img, lo2, hi2).astype(np.float32)

    sdf = onehot_to_sdf_per_channel(
        onehot_u8,
        clip=cfg["sdf_clip"],
        normalize=cfg["sdf_normalize"],
        eps=1e-8,
    ).astype(np.float32, copy=False)

    meta["sdf"]["clip"] = float(cfg["sdf_clip"])
    meta["sdf"]["normalize"] = bool(cfg["sdf_normalize"])

    img = np.ascontiguousarray(img.astype(np.float32, copy=False))
    onehot_u8 = np.ascontiguousarray(onehot_u8.astype(np.uint8, copy=False))
    sdf = np.ascontiguousarray(sdf.astype(np.float32, copy=False))
    return img, onehot_u8, sdf, meta


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
        raise RuntimeError(f"No paired img.npy + onehot.npy found under {in_root}.")

    rng = np.random.default_rng(seed)
    out_idx = start_index

    def _write_one(img_arr: np.ndarray, onehot_arr: np.ndarray, sdf_arr: np.ndarray, meta: Dict[str, Any]):
        nonlocal out_idx
        sample_dir = out_root / str(out_idx).zfill(id_width)

        if sample_dir.exists() and not overwrite:
            while (out_root / str(out_idx).zfill(id_width)).exists():
                out_idx += 1
            sample_dir = out_root / str(out_idx).zfill(id_width)

        _ensure_dir(sample_dir)
        np.save(sample_dir / "img.npy", img_arr.astype(np.float32, copy=False))
        np.save(sample_dir / "onehot.npy", onehot_arr.astype(np.uint8, copy=False))
        np.save(sample_dir / "sdf.npy", sdf_arr.astype(np.float32, copy=False))
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        out_idx += 1

    for img_p, onehot_p in pairs:
        img, onehot = _load_check(img_p, onehot_p)

        if include_original:
            onehot_u8 = _reonehot_argmax_no_bg(onehot, thr=cfg["onehot_thr"])
            sdf = onehot_to_sdf_per_channel(
                onehot_u8,
                clip=cfg["sdf_clip"],
                normalize=cfg["sdf_normalize"],
                eps=1e-8,
            ).astype(np.float32, copy=False)

            _write_one(
                img,
                onehot_u8,
                sdf,
                meta={
                    "source_image": str(img_p),
                    "source_onehot": str(onehot_p),
                    "augment": "none",
                    "config": cfg,
                },
            )

        for _ in range(aug_per_sample):
            img_aug, onehot_aug, sdf_aug, meta_aug = augment_one(img, onehot, rng, cfg)
            _write_one(
                img_aug,
                onehot_aug,
                sdf_aug,
                meta={
                    "source_image": str(img_p),
                    "source_onehot": str(onehot_p),
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
# Defaults / args
# -------------------------
def build_cfg_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "p_hflip": float(args.p_hflip),
        "p_vflip": float(args.p_vflip),
        "scale": float(args.scale),
        "trans_px": float(args.trans),
        "rot_deg": float(args.rot),

        "img_cval": float(args.img_cval),

        "brightness": float(args.brightness),
        "contrast": float(args.contrast),
        "gamma": float(args.gamma),
        "noise_sigma": float(args.noise),
        "p_blur": float(args.p_blur),
        "blur_sigma_max": float(args.blur_sigma_max),

        "clamp_min": None if args.clamp_min == "none" else float(args.clamp_min),
        "clamp_max": None if args.clamp_max == "none" else float(args.clamp_max),

        "sdf_clip": float(args.sdf_clip),
        "sdf_normalize": (not bool(args.sdf_no_norm)),

        "onehot_thr": float(args.onehot_thr),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_root", type=str, default="./dataset/ACDC/ONEHOT/train",
                    help="Input root containing img.npy & onehot.npy")
    ap.add_argument("--out_root", type=str, default="./dataset/ACDC/Aug/train",
                    help="Output root; numeric folders 000000/")
    ap.add_argument("--aug_per_sample", type=int, default=4,
                    help="Number of augmented variants per base sample")
    ap.add_argument("--no_original", action="store_true", help="Do not save the original sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--id_width", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output folders if exist (otherwise auto-find next id)")

    # Geometry
    ap.add_argument("--p_hflip", type=float, default=0.5)
    ap.add_argument("--p_vflip", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.10)
    ap.add_argument("--trans", type=float, default=8.0)
    ap.add_argument("--rot", type=float, default=30.0)

    # padding constants
    ap.add_argument("--img_cval", type=float, default=0.0)

    # Intensity
    ap.add_argument("--brightness", type=float, default=0.08)
    ap.add_argument("--contrast", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=1.20)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--p_blur", type=float, default=0.2)
    ap.add_argument("--blur_sigma_max", type=float, default=0.8)

    # clamp
    ap.add_argument("--clamp_min", type=str, default="-1.0")
    ap.add_argument("--clamp_max", type=str, default="1.0")

    # SDF
    ap.add_argument("--sdf_clip", type=float, default=10.0)
    ap.add_argument("--sdf_no_norm", action="store_true")

    # no-bg reonehot threshold
    ap.add_argument("--onehot_thr", type=float, default=0.5)

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