# viz_npy.py
# Usage:
#   python viz_npy.py --path ./dataset/ACDC/SDF/train/patient001/sdf.npy
#   python viz_npy.py --path ./dataset/ACDC/SDF/train/patient001/img.npy

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default='/Birmingham/MedFlowSeg/dataset/ACDC/Aug/train/0353/onehot.npy', help="path to .npy file")
    args = ap.parse_args()

    x = np.load(args.path)
    print("Loaded:", args.path)
    print("shape:", x.shape, "dtype:", x.dtype, "min/max:", float(x.min()), float(x.max()))

    if x.ndim == 2:
        plt.figure(figsize=(5, 5))
        plt.imshow(x, cmap="gray")
        plt.title(f"{args.path}\n{tuple(x.shape)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    elif x.ndim == 3:
        C, H, W = x.shape
        cols = min(4, C)
        rows = (C + cols - 1) // cols
        plt.figure(figsize=(4 * cols, 4 * rows))
        for c in range(C):
            ax = plt.subplot(rows, cols, c + 1)
            ax.imshow(x[c], cmap="coolwarm")  # SDF/heatmap style
            ax.set_title(f"ch {c}  min={x[c].min():.3f} max={x[c].max():.3f}")
            ax.axis("off")
        plt.suptitle(f"{args.path}\n{tuple(x.shape)}", y=1.02)
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError(f"Unsupported ndim={x.ndim}, expected 2 or 3.")


if __name__ == "__main__":
    main()