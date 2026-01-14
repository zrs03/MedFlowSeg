import os
import numpy as np
import matplotlib.pyplot as plt

# -------- 修改为你的路径 --------
img_path = "/Birmingham/MedFlowSeg/dataset/ACDC/Aug/train/0001/img.npy"
gt_path  = "/Birmingham/MedFlowSeg/dataset/ACDC/Aug/train/0001/sdf.npy"

# -------- 输出文件 --------
out_path = "./viz_patient001_ED_frame01.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# -------- Load --------
img = np.load(img_path)   # (H,W)
gt  = np.load(gt_path)    # (C,H,W)  (这里你是SDF)

print("img:", img.shape, img.dtype, float(img.min()), float(img.max()))
print("gt :", gt.shape, gt.dtype, float(gt.min()), float(gt.max()))

# -------- Plot & Save --------
labels = ["Background", "RV", "Myo", "LV"]
C = gt.shape[0]

fig, axes = plt.subplots(1, 1 + C, figsize=(3.6 * (1 + C), 4))

# image
axes[0].imshow(img, cmap="gray")
axes[0].set_title("ED Image")
axes[0].axis("off")

# sdf channels (use a diverging colormap to show +/-)
for i in range(C):
    im = axes[i + 1].imshow(gt[i], cmap="seismic")
    axes[i + 1].set_title(f"SDF: {labels[i] if i < len(labels) else i}")
    axes[i + 1].axis("off")
    fig.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved visualization to: {out_path}")