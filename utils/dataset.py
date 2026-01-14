from __future__ import annotations
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    Generic dataset for folder-per-sample format:

      root/
        sample_1/
          img.npy        or image.npy
          sdf.npy
        sample_2/
          ...

    Returns:
      image:  (1,H,W) float32 torch tensor
      target: (C,H,W) float32 torch tensor
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_file: str = "img.npy",
        target_file: str = "sdf.npy",
        add_channel_dim: bool = True,   # image -> (1,H,W) if needed
        sort_numeric: bool = True,      # sort 000123 numerically if possible
    ):
        self.root = Path(root)
        self.image_file = image_file
        self.target_file = target_file
        self.add_channel_dim = add_channel_dim

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # collect candidate sample directories
        dirs = [p for p in self.root.iterdir() if p.is_dir()]

        if sort_numeric:
            def _key(p: Path):
                try:
                    return int(p.name)
                except Exception:
                    return p.name
            dirs = sorted(dirs, key=_key)
        else:
            dirs = sorted(dirs)

        # keep only valid samples
        self.sample_dirs: List[Path] = []
        for d in dirs:
            if (d / self.image_file).exists() and (d / self.target_file).exists():
                self.sample_dirs.append(d)

        if len(self.sample_dirs) == 0:
            raise RuntimeError(
                f"No valid samples found in {self.root} "
                f"(need {self.image_file} and {self.target_file})."
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        d = self.sample_dirs[idx]

        # -------- load --------
        img = np.load(d / self.image_file).astype(np.float32)   # (H,W) or (1,H,W) or (H,W,1)
        tgt = np.load(d / self.target_file).astype(np.float32)  # (C,H,W)

        # -------- normalize image shape --------
        if self.add_channel_dim:
            if img.ndim == 2:
                img = img[None, ...]               # (1,H,W)
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = np.transpose(img, (2, 0, 1)) # (1,H,W)

        img_t = torch.from_numpy(np.ascontiguousarray(img))
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt))

        return img_t, tgt_t