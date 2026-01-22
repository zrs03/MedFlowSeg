from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取 mask
mask_path = "/Birmingham/MedFlowSeg/REFUGE2dataset/train/mask/g0001.bmp"
mask = Image.open(mask_path).convert("L")
mask_np = np.array(mask)

print(f"Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
values, counts = np.unique(mask_np, return_counts=True)
print("Unique values and counts:")
for v, c in zip(values, counts):
    print(f"  value = {v}, count = {c}")

# 要可视化的值
target_values = [0, 128, 255]

plt.figure(figsize=(12, 4))

for i, v in enumerate(target_values):
    vis = (mask_np == v).astype(np.uint8) * 255  # 二值图

    plt.subplot(1, 3, i + 1)
    plt.imshow(vis, cmap="gray")
    plt.title(f"value == {v}")
    plt.axis("off")

plt.tight_layout()
plt.show()