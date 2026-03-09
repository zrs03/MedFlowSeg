# BRATSPreprocessMaxSliceWindow.py
#
# BRATS数据集预处理脚本（按最大肿瘤横截面切片取窗口）
# 将BRATS数据预处理为2D切片格式，并按以下规则筛选切片：
#
# 处理逻辑：
# 1) 提取BRATS 2D切片
# 2) 先找到肿瘤像素数最大的切片（最大肿瘤横截面）
# 3) 以该切片为中心，上下各取window_radius个切片
# 4) 在这个窗口内，仅保留肿瘤像素数 > min_tumor_pixels 的切片
# 5) Center crop到固定大小(size,size)
# 6) 归一化：对每个模态做 z-score（仅在非零区域），再 clip 到[-5,5] 并缩放到[-1,1]
# 7) 标签合并：将所有肿瘤类别合并为二值标签(0/1)
# 8) 患者级split 8:1:1 (train/val/test)，保存split.json

from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Tuple, List, Optional, Dict

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt


# -------------------------
# Utils
# -------------------------

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def center_crop_hw(x: np.ndarray, out_hw=(224, 224)):
    """
    中心裁剪到out_hw；若尺寸不足则补零。
    支持形状: (C, H, W)
    """
    C, H, W = x.shape
    out_h, out_w = out_hw

    if H > out_h:
        top = (H - out_h) // 2
        x = x[:, top:top + out_h, :]
    else:
        pad_h = out_h - H
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        x = np.pad(x, ((0, 0), (pad_top, pad_bottom), (0, 0)),
                   mode='constant', constant_values=0)

    if W > out_w:
        left = (W - out_w) // 2
        x = x[:, :, left:left + out_w]
    else:
        pad_w = out_w - W
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)),
                   mode='constant', constant_values=0)

    return x  # (C,H,W)


def zscore_norm_keep_zero_4d(
    x: np.ndarray,
    clip_z: float = 5.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    x: (C,H,W,D) float32
    For each channel:
      - compute mean/std on non-zero voxels only
      - z-score normalize (keep background zeros)
      - clip to [-clip_z, clip_z] and scale to [-1,1] by /clip_z
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (C,H,W,D), got {x.shape}")
    C = x.shape[0]
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(C):
        v = x[c].astype(np.float32, copy=False)
        m = (v != 0)
        if m.any():
            mu = float(v[m].mean())
            sd = float(v[m].std())
            if sd < eps:
                sd = 1.0
            vv = (v - mu) / (sd + eps)
            vv = np.clip(vv, -clip_z, clip_z) / float(clip_z)
            out[c] = vv.astype(np.float32, copy=False)
        else:
            out[c] = 0.0
    return out


def merge_tumor_labels(label: np.ndarray) -> np.ndarray:
    """
    将所有肿瘤类别(>0)合并为1，背景保持为0
    BRATS标签：0=背景, 1=坏死, 2=水肿, 4=增强肿瘤
    """
    return (label > 0).astype(np.float32)


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
    sdf = dist_out - dist_in (mask内为负，外为正，边界≈0)
    clip到[-clip,clip]；若 normalize=True，再除以 clip -> [-1,1]
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
# BRATS文件解析
# -------------------------

def scan_brats_directory(directory: Path) -> List[Dict[str, Path]]:
    """
    扫描BRATS目录，构建数据索引
    返回列表，每个元素是一个字典，包含t1,t1ce,t2,flair,seg的文件路径
    """
    directory = Path(directory)
    seqtypes = ['flair', 't1', 't1ce', 't2', 'seg']
    seqtypes_set = set(seqtypes)
    database = []

    patient_dirs = [d for d in directory.iterdir() if d.is_dir()]

    for patient_dir in patient_dirs:
        datapoint = {}
        for file_path in patient_dir.glob("*.nii"):
            filename = file_path.name.lower()

            if 'flair' in filename:
                seqtype = 'flair'
            elif 't1ce' in filename or 't1c' in filename:
                seqtype = 't1ce'
            elif 't1' in filename and 'ce' not in filename:
                seqtype = 't1'
            elif 't2' in filename:
                seqtype = 't2'
            elif 'seg' in filename:
                seqtype = 'seg'
            else:
                continue

            datapoint[seqtype] = file_path

        if set(datapoint.keys()) == seqtypes_set:
            database.append(datapoint)
        else:
            missing = seqtypes_set - set(datapoint.keys())
            print(f"警告: 患者 {patient_dir.name} 缺少序列: {missing}")

    return database


def extract_patient_id_from_path(filepath: Path) -> str:
    """
    从文件路径中提取患者ID
    例如: BraTS20_Training_001_flair.nii -> BraTS20_Training_001
    """
    filename = filepath.name
    parts = filename.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[:3])
    return filepath.parent.name


# -------------------------
# Split
# -------------------------

def get_unique_patients(database: List[Dict[str, Path]]) -> List[str]:
    patients = set()
    for datapoint in database:
        first_key = next(iter(datapoint.keys()))
        patient_id = extract_patient_id_from_path(datapoint[first_key])
        patients.add(patient_id)
    return sorted(list(patients))


def make_split(patients: List[str], seed=42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    patients_copy = patients.copy()
    rng.shuffle(patients_copy)

    n = len(patients_copy)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))

    train = patients_copy[:n_train]
    val = patients_copy[n_train:n_train + n_val]
    test = patients_copy[n_train + n_val:]
    return train, val, test


def save_split_json(out_path: Path, seed: int, train, val, test):
    obj = {
        "seed": seed,
        "counts": {
            "total": len(train) + len(val) + len(test),
            "train": len(train),
            "val": len(val),
            "test": len(test)
        },
        "train": train,
        "val": val,
        "test": test,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# 切片选择
# -------------------------

def count_tumor_pixels_per_slice(label_vol: np.ndarray) -> np.ndarray:
    """
    label_vol: (1, H, W, D)
    return: (D,) 每个切片的肿瘤像素数
    """
    if label_vol.ndim != 4 or label_vol.shape[0] != 1:
        raise ValueError(f"Expected label_vol shape (1,H,W,D), got {label_vol.shape}")
    return np.sum(label_vol[0] > 0, axis=(0, 1)).astype(np.int64)


def get_max_cross_section_window_slices(
    label_vol: np.ndarray,
    min_tumor_pixels: int = 100,
    window_radius: int = 5,
) -> List[int]:
    """
    选择规则：
    1) 先找到肿瘤横截面（2D切片）像素数最大的切片作为中心切片
    2) 以该中心切片为中心，上下各取window_radius个切片
    3) 在该范围内，仅保留肿瘤像素数 > min_tumor_pixels 的切片

    返回升序切片索引列表。
    """
    tumor_counts = count_tumor_pixels_per_slice(label_vol)
    max_pixels = int(tumor_counts.max(initial=0))

    if max_pixels <= min_tumor_pixels:
        return []

    center_slice = int(np.argmax(tumor_counts))
    depth = int(tumor_counts.shape[0])

    start_idx = max(0, center_slice - window_radius)
    end_idx = min(depth - 1, center_slice + window_radius)

    selected = [
        idx for idx in range(start_idx, end_idx + 1)
        if int(tumor_counts[idx]) > min_tumor_pixels
    ]
    return selected


# -------------------------
# 主处理流程
# -------------------------

def process_one_patient(
    datapoint: Dict[str, Path],
    out_root: Path,
    split_name: str,
    out_hw: Tuple[int, int],
    min_tumor_pixels: int = 100,
    window_radius: int = 5,
) -> int:
    """
    处理一个患者的所有切片
    返回保存的切片数量
    """
    modalities = []
    for seqtype in ['t1', 't1ce', 't2', 'flair']:
        if seqtype not in datapoint:
            print(f"  错误: 缺少序列 {seqtype}")
            return 0
        nib_img = nib.load(str(datapoint[seqtype]))
        img_data = nib_img.get_fdata().astype(np.float32)
        modalities.append(img_data)

    image_vol = np.stack(modalities, axis=0)
    image_vol = zscore_norm_keep_zero_4d(image_vol, clip_z=5.0)

    seg_nib = nib.load(str(datapoint['seg']))
    label_vol = seg_nib.get_fdata().astype(np.float32)
    label_vol = label_vol[None, ...]  # (1, H, W, D)

    patient_id = extract_patient_id_from_path(datapoint['t1'])

    filtered_slices = get_max_cross_section_window_slices(
        label_vol,
        min_tumor_pixels=min_tumor_pixels,
        window_radius=window_radius,
    )
    num_selected = len(filtered_slices)
    saved = 0

    for slice_idx in filtered_slices:
        img_slice = image_vol[..., slice_idx]   # (4, H, W)
        label_slice = label_vol[..., slice_idx] # (1, H, W)

        img_slice = center_crop_hw(img_slice, out_hw=out_hw)
        label_slice = center_crop_hw(label_slice, out_hw=out_hw)

        label_slice = merge_tumor_labels(label_slice)

        lab = (label_slice[0] > 0.5)
        onehot = lab[None, ...].astype(np.uint8)
        sdf = onehot_to_sdf_per_channel(onehot, clip=10.0, normalize=True)

        slice_out_dir = out_root / split_name / f"{patient_id}_slice{slice_idx:03d}"
        _ensure_dirs(slice_out_dir)

        np.save(slice_out_dir / "img.npy", img_slice.astype(np.float32))
        np.save(slice_out_dir / "label.npy", label_slice.astype(np.float32))
        np.save(slice_out_dir / "onehot.npy", onehot)
        np.save(slice_out_dir / "sdf.npy", sdf.astype(np.float32))

        saved += 1
        if saved % 50 == 0:
            print(f"    已处理 {saved}/{num_selected} 切片")

    return saved


def run(
    base_dir: Path,
    out_root: Path,
    size: int,
    seed: int,
    split_json: Optional[Path],
    min_tumor_pixels: int = 100,
    window_radius: int = 5,
):
    base_dir = Path(base_dir)
    out_root = Path(out_root)

    print(f"正在扫描数据目录: {base_dir}")
    database = scan_brats_directory(base_dir)
    print(f"找到 {len(database)} 个完整的数据点")

    if len(database) == 0:
        print("错误: 未找到任何完整的数据点")
        print("请检查目录结构是否正确")
        return

    patients = get_unique_patients(database)
    print(f"找到 {len(patients)} 个唯一的患者")

    if split_json is not None and Path(split_json).exists():
        print(f"从{split_json}加载现有split...")
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        train_patients, val_patients, test_patients = split["train"], split["val"], split["test"]
        seed_used = split.get("seed", seed)
    else:
        print(f"创建新的split (seed={seed})...")
        train_patients, val_patients, test_patients = make_split(
            patients, seed=seed, ratios=(0.8, 0.1, 0.1)
        )
        seed_used = seed
        save_split_json(out_root / "split.json", seed_used, train_patients, val_patients, test_patients)
        print(f"Split已保存到: {out_root / 'split.json'}")

    print(f"Train: {len(train_patients)} patients")
    print(f"Val: {len(val_patients)} patients")
    print(f"Test: {len(test_patients)} patients")

    patient_to_datapoint = {}
    for dp in database:
        first_key = next(iter(dp.keys()))
        pid = extract_patient_id_from_path(dp[first_key])
        patient_to_datapoint[pid] = dp

    out_hw = (size, size)
    total_saved = 0
    total_patients_processed = 0

    for split_name, patient_list in [
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients)
    ]:
        print(f"\n处理{split_name}集...")
        for patient_id in patient_list:
            if patient_id in patient_to_datapoint:
                print(f"  处理患者 {patient_id}...")
                n = process_one_patient(
                    datapoint=patient_to_datapoint[patient_id],
                    out_root=out_root,
                    split_name=split_name,
                    out_hw=out_hw,
                    min_tumor_pixels=min_tumor_pixels,
                    window_radius=window_radius,
                )
                total_saved += n
                total_patients_processed += 1
                print(f"  患者 {patient_id}: 保存了 {n} 个切片")
            else:
                print(f"  警告: 找不到患者 {patient_id} 的数据")

    print(f"\n处理完成!")
    print(f"处理了 {total_patients_processed} 个患者")
    print(f"保存了 {total_saved} 个2D切片")
    print(f"输出目录: {out_root}")


def main():
    ap = argparse.ArgumentParser(description="BRATS数据集预处理脚本（按最大肿瘤横截面切片取窗口）")
    ap.add_argument(
        "--base_dir",
        type=str,
        default="/lustre/home/2110306221/data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        help="BRATS数据集根目录，包含各患者子文件夹",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="./dataset/BRATS/SDF",
        help="输出根目录，包含train/val/test + split.json",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=256,
        help="输出图像尺寸 (size x size)，默认256",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于数据划分",
    )
    ap.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="可选的现有split.json文件路径，用于复用划分",
    )
    ap.add_argument(
        "--min_tumor_pixels",
        type=int,
        default=100,
        help="窗口内最小肿瘤像素数阈值；仅保留 > 该阈值的切片",
    )
    ap.add_argument(
        "--window_radius",
        type=int,
        default=5,
        help="以最大肿瘤横截面为中心，上下各取的切片数；默认5",
    )

    args = ap.parse_args()

    run(
        base_dir=Path(args.base_dir),
        out_root=Path(args.out_root),
        size=args.size,
        seed=args.seed,
        split_json=Path(args.split_json) if args.split_json else None,
        min_tumor_pixels=args.min_tumor_pixels,
        window_radius=args.window_radius,
    )


if __name__ == "__main__":
    main()
