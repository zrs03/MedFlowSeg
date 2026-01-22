from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# --- Try OpenCV first (fast polygon fill) ---
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --- Fallback polygon fill (skimage) ---
try:
    from skimage.draw import polygon as sk_polygon
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# ============================================================
# Utils
# ============================================================
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def center_pad_or_crop_hw(x: np.ndarray, out_hw=(256, 256), pad_value=0):
    """
    x: (H,W) or (H,W,1)
    return: (out_h,out_w)
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


def to_onehot_2(mask2d: np.ndarray) -> np.ndarray:
    """
    mask2d: (H,W) {0,1}
    return: (2,H,W) uint8 onehot: [bg, fg]
    """
    m = (mask2d > 0).astype(np.uint8)
    bg = (m == 0).astype(np.uint8)
    fg = m.astype(np.uint8)
    return np.stack([bg, fg], axis=0)


def onehot_to_sdf_per_channel(onehot: np.ndarray, clip: float = 10.0, normalize: bool = True, eps: float = 1e-8):
    """
    onehot: (C,H,W)
    sdf: (C,H,W) float32, inside negative, outside positive
    clip to [-clip,clip], optionally /clip -> [-1,1]
    """
    if onehot.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {onehot.shape}")
    if normalize and clip <= 0:
        raise ValueError("clip must be >0 when normalize=True")

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
# XML → polygons
# ============================================================
def parse_marks_from_xml(xml_path: Path) -> Dict[int, List[np.ndarray]]:
    """
    Return dict: {image_idx: [poly(N,2), ...]}  poly in (x,y), int32
    """
    root = ET.parse(xml_path).getroot()
    by_img: Dict[int, List[np.ndarray]] = {}

    for mark in root.findall(".//mark"):
        img_node = mark.find("image")
        svg_node = mark.find("svg")
        if img_node is None or svg_node is None:
            continue

        img_text = (img_node.text or "").strip()
        svg_text = (svg_node.text or "").strip()
        if not img_text or not svg_text:
            continue

        try:
            img_idx = int(img_text)
        except Exception:
            continue

        try:
            rois = json.loads(svg_text)
        except Exception:
            continue

        if isinstance(rois, dict):
            rois = [rois]

        polys: List[np.ndarray] = []
        for roi in rois:
            pts = roi.get("points", None)
            if not pts or len(pts) < 3:
                continue
            poly = np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)
            polys.append(poly)

        if polys:
            by_img.setdefault(img_idx, []).extend(polys)

    return by_img


def fill_polys(mask: np.ndarray, polys_xy: List[np.ndarray], value: int = 1) -> np.ndarray:
    if not polys_xy:
        return mask

    if _HAS_CV2:
        for poly in polys_xy:
            if poly.shape[0] < 3:
                continue
            cv2.fillPoly(mask, [poly.reshape((-1, 1, 2))], color=int(value))
        return mask

    if _HAS_SKIMAGE:
        H, W = mask.shape
        for poly in polys_xy:
            if poly.shape[0] < 3:
                continue
            xs = poly[:, 0].astype(np.int64)
            ys = poly[:, 1].astype(np.int64)
            rr, cc = sk_polygon(ys, xs, shape=(H, W))
            mask[rr, cc] = int(value)
        return mask

    raise RuntimeError("Need opencv-python or scikit-image for polygon filling.")


def read_image_gray_float(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("L")
        return np.asarray(im, dtype=np.float32)


def list_case_ids(in_dir: Path) -> List[str]:
    return sorted([p.stem for p in in_dir.glob("*.xml")])


def list_images_for_case(in_dir: Path, case_id: str) -> List[Tuple[int, Path]]:
    """
    Return [(k, path)] for case_id_k.*
    """
    pat = re.compile(rf"^{re.escape(case_id)}_(\d+)\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE)
    items: List[Tuple[int, Path]] = []
    for p in in_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return items


# ============================================================
# Split (NOW on samples: case_k)
# ============================================================
def list_sample_ids(in_dir: Path) -> List[str]:
    """
    Sample id = "case_k" for each image file that has a corresponding xml (case.xml).
    """
    sample_ids: List[str] = []
    for xml_path in sorted(in_dir.glob("*.xml")):
        case_id = xml_path.stem
        imgs = list_images_for_case(in_dir, case_id)
        for k, _ in imgs:
            sample_ids.append(f"{case_id}_{k}")
    return sample_ids


def make_split(sample_ids: List[str], seed=42, ratios=(0.8, 0.1, 0.1)):
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


# ============================================================
# Processing (NOW each sample_id has its own folder)
# ============================================================
def process_one_sample(
    in_dir: Path,
    sample_id: str,          # "100_1"
    split_name: str,
    out_root: Path,
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
      out_root/split/sample_id/
        img.npy
        onehot.npy
        sdf.npy
        mask.png (optional)
    """
    # parse "case_k"
    m = re.match(r"^(.*)_(\d+)$", sample_id)
    if not m:
        if verbose:
            print(f"[Skip] bad sample_id: {sample_id}")
        return False
    case_id = m.group(1)
    k = int(m.group(2))

    xml_path = in_dir / f"{case_id}.xml"
    if not xml_path.exists():
        if verbose:
            print(f"[Skip] missing xml for {sample_id}: {xml_path.name}")
        return False

    # find image
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        cand = in_dir / f"{case_id}_{k}{ext}"
        if cand.exists():
            img_path = cand
            break
    if img_path is None:
        if verbose:
            print(f"[Skip] missing image for {sample_id}")
        return False

    # polygons for this k
    by_img_polys = parse_marks_from_xml(xml_path)
    polys = by_img_polys.get(k, [])

    img = read_image_gray_float(img_path)  # (H,W)
    H, W = img.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    if polys:
        clipped = []
        for poly in polys:
            poly2 = poly.copy()
            poly2[:, 0] = np.clip(poly2[:, 0], 0, W - 1)
            poly2[:, 1] = np.clip(poly2[:, 1], 0, H - 1)
            clipped.append(poly2)
        mask = fill_polys(mask, clipped, value=1)

    # resize by pad/crop
    out_hw = (size, size)
    img2d = center_pad_or_crop_hw(img, out_hw=out_hw, pad_value=0.0)
    mask2d = center_pad_or_crop_hw(mask, out_hw=out_hw, pad_value=0).astype(np.uint8)

    # normalize image [-1,1]
    img_norm = minmax01_single(img2d, clip_percentiles=(p_lo, p_hi))

    # onehot + sdf
    onehot = to_onehot_2(mask2d)  # (2,H,W)
    sdf = onehot_to_sdf_per_channel(onehot, clip=sdf_clip, normalize=sdf_normalize)  # (2,H,W)

    # save
    out_dir = out_root / split_name / sample_id
    _ensure_dirs(out_dir)
    np.save(out_dir / "img.npy", img_norm.astype(np.float32))
    np.save(out_dir / "onehot.npy", onehot.astype(np.uint8))
    np.save(out_dir / "sdf.npy", sdf.astype(np.float32))

    if save_png_mask:
        Image.fromarray((mask2d > 0).astype(np.uint8) * 255).save(out_dir / "mask.png")

    if verbose:
        print(f"[OK] {sample_id} -> {out_dir}")

    return True


def run(
    in_dir: Path,
    out_root: Path,
    size: int,
    p_lo: float,
    p_hi: float,
    sdf_clip: float,
    sdf_normalize: bool,
    seed: int,
    split_json: Path | None,
    save_png_mask: bool,
    verbose: bool,
):
    in_dir = Path(in_dir)
    out_root = Path(out_root)
    _ensure_dirs(out_root)

    # split on sample_ids
    if split_json is not None and split_json.exists():
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train, val, test = split["train"], split["val"], split["test"]
        if verbose:
            print(f"[Info] Reusing split: {split_json}")
    else:
        sample_ids = list_sample_ids(in_dir)
        train, val, test = make_split(sample_ids, seed=seed, ratios=(0.8, 0.1, 0.1))
        save_split_json(out_root / "split.json", seed, train, val, test)
        if verbose:
            print(f"[Info] Created split.json at: {out_root / 'split.json'}")

    total_ok = 0
    for split_name, ids in [("train", train), ("val", val), ("test", test)]:
        for sid in ids:
            ok = process_one_sample(
                in_dir=in_dir,
                sample_id=sid,
                split_name=split_name,
                out_root=out_root,
                size=size,
                p_lo=p_lo,
                p_hi=p_hi,
                sdf_clip=sdf_clip,
                sdf_normalize=sdf_normalize,
                save_png_mask=save_png_mask,
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
    ap.add_argument("--in_dir", type=str, default="./DDMIdataset", help="包含 *.xml 和 case_k.jpg 的文件夹")
    ap.add_argument("--out_root", type=str, default="./dataset/DDMI/SDF", help="输出根目录：train/val/test + split.json")
    ap.add_argument("--size", type=int, default=256, help="pad/crop 到固定尺寸 size x size")
    ap.add_argument("--p_lo", type=float, default=1.0, help="percentile clip low")
    ap.add_argument("--p_hi", type=float, default=99.0, help="percentile clip high")
    ap.add_argument("--clip", type=float, default=10.0, help="SDF clip 距离")
    ap.add_argument("--sdf_no_norm", action="store_true", help="不把 SDF 归一化到 [-1,1]（只 clip）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_json", type=str, default=None, help="可选：复用已有 split.json")
    ap.add_argument("--save_png_mask", action="store_true", help="额外保存 mask.png 方便检查")
    ap.add_argument("--quiet", action="store_true", help="不打印日志")
    args = ap.parse_args()

    if not _HAS_CV2 and not _HAS_SKIMAGE:
        raise RuntimeError("请安装 opencv-python (推荐) 或 scikit-image，用于多边形填充。")

    run(
        in_dir=Path(args.in_dir),
        out_root=Path(args.out_root),
        size=int(args.size),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        sdf_clip=float(args.clip),
        sdf_normalize=(not args.sdf_no_norm),
        seed=int(args.seed),
        split_json=Path(args.split_json) if args.split_json else None,
        save_png_mask=bool(args.save_png_mask),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()