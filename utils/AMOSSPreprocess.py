#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMOSS / AMOS22 CT preprocess (2D slices) — BTCV-style per-class Top-k union.

Key behavior:
- For each class (exclude background=0), pick Top-k slices by pixel count (default k=2).
- Union slices across classes (duplicates removed).
- If union is empty: fallback to the single slice with max total foreground pixels.
- Optional cap with --max_slices:
    -1 : no cap (keep all union slices)
    >0 : keep at most max_slices (ranked by total foreground pixels on that slice)

Outputs (per selected slice) under out_root/{train|val|test}/{case_id}_sliceXXX/ :
- img.npy            (float32, HxW, normalized)
- onehot.npy         (uint8, CxHxW)            [if label available]
- sdf.npy            (float32, CxHxW)          [if label available]
- original.npy       (uint8, HxW label)       [if label available]
Also writes split_info.json at out_root/.

Example:
python AMOSSPreprocess_topk_union_min.py \
  --base_dir /path/to/AMOS22 \
  --json_file dataset_CT.json \
  --out_root ./amos22_ct_processed \
  --max_slices -1
"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


def load_json(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: Path, rel: str) -> Path:
    rel = rel.lstrip("./")
    name = Path(rel).name
    candidates = [
        base_dir / rel,
        base_dir / "data" / rel,
        base_dir / "imagesTr" / name,
        base_dir / "labelsTr" / name,
        base_dir / "imagesTs" / name,
        base_dir / "labelsTs" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return base_dir / rel


def load_nifti(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def center_pad_or_crop(img: np.ndarray, target_hw: Tuple[int, int], pad_value: float = 0) -> np.ndarray:
    """2D center pad/crop to (H,W)."""
    th, tw = target_hw
    h, w = img.shape

    # pad/crop H
    if h < th:
        pad = th - h
        top = pad // 2
        bottom = pad - top
        img = np.pad(img, ((top, bottom), (0, 0)), constant_values=pad_value)
    elif h > th:
        start = (h - th) // 2
        img = img[start:start + th, :]

    # pad/crop W
    h, w = img.shape
    if w < tw:
        pad = tw - w
        left = pad // 2
        right = pad - left
        img = np.pad(img, ((0, 0), (left, right)), constant_values=pad_value)
    elif w > tw:
        start = (w - tw) // 2
        img = img[:, start:start + tw]

    return img


def normalize_ct(img: np.ndarray, low_p: float, high_p: float) -> np.ndarray:
    lo, hi = np.percentile(img, (low_p, high_p))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img.astype(np.float32)


def labels_to_onehot(lbl2d: np.ndarray, num_classes: int) -> np.ndarray:
    lbl = np.clip(lbl2d.astype(np.int32), 0, num_classes - 1)
    oh = np.eye(num_classes, dtype=np.uint8)[lbl]         # HxWxC
    return np.transpose(oh, (2, 0, 1))                    # CxHxW


def compute_sdf(onehot: np.ndarray, clip: float) -> np.ndarray:
    """Signed distance function per class; background(0) stays zeros."""
    c, h, w = onehot.shape
    sdf = np.zeros((c, h, w), dtype=np.float32)
    for k in range(1, c):
        mask = onehot[k].astype(bool)
        if not np.any(mask):
            sdf[k] = 1.0  # consistent with many med-seg codebases
            continue
        pos = distance_transform_edt(mask)
        neg = distance_transform_edt(~mask)
        s = neg - pos
        s = np.clip(s, -clip, clip) / clip
        sdf[k] = s.astype(np.float32)
    return sdf


def select_slices_topk_union(
    label_vol: Optional[np.ndarray],
    depth: int,
    num_classes: int,
    k_per_class: int = 2,
    max_slices: int = -1,
) -> List[int]:
    """
    BTCV-style:
      per class -> top-k slices (by pixel count) -> union
      empty -> fallback to max-foreground slice
      cap with max_slices (>0) by total foreground pixels
    """
    if label_vol is None:
        n = 5 if max_slices < 0 else max_slices
        if depth <= n:
            return list(range(depth))
        return np.linspace(0, depth - 1, n, dtype=int).tolist()

    selected = set()
    for c in range(1, num_classes):
        per_z = np.count_nonzero(label_vol == c, axis=(0, 1))  # (Z,)
        if per_z.max() == 0:
            continue
        top = np.argsort(per_z)[::-1][:k_per_class]
        for z in top:
            if per_z[z] > 0:
                selected.add(int(z))

    if not selected:
        total = np.count_nonzero(label_vol > 0, axis=(0, 1))
        return [int(np.argmax(total))]

    idx = sorted(selected)

    if max_slices > 0 and len(idx) > max_slices:
        total = np.count_nonzero(label_vol > 0, axis=(0, 1))
        idx = sorted(idx, key=lambda z: total[z], reverse=True)[:max_slices]
        idx = sorted(idx)

    return idx


def parse_case_id(img_rel: str) -> str:
    # expects "...amos_XXXX..." ; fallback to filename stem
    s = Path(img_rel).name
    if "amos_" in s:
        try:
            n = int(s.split("amos_")[1].split(".")[0].split("_")[0])
            return f"amos_{n:04d}"
        except Exception:
            pass
    return Path(img_rel).stem


def is_ct_case(img_rel: str) -> bool:
    # AMOS22: IDs < 200 are CT, >=200 are MRI (common convention)
    if "amos_" not in img_rel:
        return True
    try:
        n = int(Path(img_rel).name.split("amos_")[1].split(".")[0].split("_")[0])
        return n < 200
    except Exception:
        return True


def extract_splits(ds: Dict) -> Dict[str, List[Dict]]:
    splits = {"train": [], "val": [], "test": []}
    for k_json, k_out in [("training", "train"), ("validation", "val"), ("test", "test")]:
        for item in ds.get(k_json, []):
            img = item.get("image")
            if not img or not is_ct_case(img):
                continue
            rec = {"image": img, "label": item.get("label"), "case_id": parse_case_id(img)}
            splits[k_out].append(rec)
    return splits


def process_case(
    base_dir: Path,
    out_root: Path,
    split: str,
    case: Dict,
    target_hw: Tuple[int, int],
    num_classes: int,
    low_p: float,
    high_p: float,
    sdf_clip: float,
    max_slices: int,
    k_per_class: int = 2,
) -> int:
    img_path = resolve_path(base_dir, case["image"])
    lbl_path = resolve_path(base_dir, case["label"]) if case.get("label") else None

    img_vol = load_nifti(img_path)
    lbl_vol = load_nifti(lbl_path) if (lbl_path and lbl_path.exists()) else None

    depth = int(img_vol.shape[2])
    zs = select_slices_topk_union(lbl_vol, depth, num_classes, k_per_class=k_per_class, max_slices=max_slices)

    saved = 0
    for i, z in enumerate(zs):
        if z < 0 or z >= depth:
            continue

        img2d = img_vol[:, :, z]
        img2d = center_pad_or_crop(img2d, target_hw, pad_value=0)
        img2d = normalize_ct(img2d, low_p, high_p)

        d = out_root / split / f"{case['case_id']}_slice{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "img.npy", img2d)

        if lbl_vol is not None and z < lbl_vol.shape[2]:
            lbl2d = lbl_vol[:, :, z]
            lbl2d = center_pad_or_crop(lbl2d, target_hw, pad_value=0).astype(np.int32)
            oh = labels_to_onehot(lbl2d, num_classes)
            sdf = compute_sdf(oh, sdf_clip)
            np.save(d / "onehot.npy", oh)
            np.save(d / "sdf.npy", sdf)
            np.save(d / "original.npy", lbl2d.astype(np.uint8))

        saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="AMOS22 dataset root")
    ap.add_argument("--json_file", type=str, default="dataset_CT.json", help="dataset json (CT)")
    ap.add_argument("--out_root", type=str, default="./amos22_ct_processed", help="output root")
    ap.add_argument("--target_size", type=int, nargs=2, default=[256, 256], help="H W")
    ap.add_argument("--ct_low", type=float, default=0.5, help="CT clip low percentile")
    ap.add_argument("--ct_high", type=float, default=99.5, help="CT clip high percentile")
    ap.add_argument("--sdf_clip", type=float, default=10.0, help="SDF clip distance")
    ap.add_argument("--max_slices", type=int, default=-1,
                    help="Cap total selected slices per volume. -1 means no cap")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train").mkdir(exist_ok=True)
    (out_root / "val").mkdir(exist_ok=True)
    (out_root / "test").mkdir(exist_ok=True)

    js = Path(args.json_file)
    if not js.exists():
        # try under base_dir
        for cand in [base_dir / args.json_file, base_dir / "dataset_CT.json", base_dir / "dataset.json"]:
            if cand.exists():
                js = cand
                break
    ds = load_json(js)

    # num classes from labels map; fallback to 16 (AMOS CT typical 0..15)
    labels_map = ds.get("labels", {})
    num_classes = len(labels_map) if labels_map else 16

    splits = extract_splits(ds)

    total = 0
    for split in ["train", "val", "test"]:
        cases = splits[split]
        print(f"[{split}] cases: {len(cases)}")
        ssum = 0
        for idx, case in enumerate(cases):
            if idx % 10 == 0:
                print(f"  {idx}/{len(cases)} ...")
            ssum += process_case(
                base_dir=base_dir,
                out_root=out_root,
                split=split,
                case=case,
                target_hw=tuple(args.target_size),
                num_classes=num_classes,
                low_p=args.ct_low,
                high_p=args.ct_high,
                sdf_clip=args.sdf_clip,
                max_slices=-1,
                k_per_class=2,   # fixed: each class keep 2 slices
            )
        print(f"[{split}] saved slices: {ssum}")
        total += ssum

    split_info = {
        "train": [c["case_id"] for c in splits["train"]],
        "val": [c["case_id"] for c in splits["val"]],
        "test": [c["case_id"] for c in splits["test"]],
        "labels": labels_map,
        "num_classes": num_classes,
        "k_per_class": 2,
        "max_slices": -1,
    }
    with (out_root / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"Done. Total saved slices: {total}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
