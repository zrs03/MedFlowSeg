import matplotlib.pyplot as plt
import numpy as np

x_labels = ['1 Sample', '5 Samples', '25 Samples', '50 Samples', '75 Samples', '100 Samples']
x_positions = np.arange(len(x_labels))

y_ensem_diff = [71.2, 77.5, 83.8, 92.76, 92.76, 92.76]
y_seg_diff = [70.0, 75.3, 84.7, 91.80, 91.80, 91.95]
y_medseg_diff = [81.1, 84.3, 87.7, 91.89, 91.89, 92.00]
y_medseg_diff_v2 = [85.5, 86.7, 88.9, 92.68, 92.68, 92.72]
y_flowsdf = [78.0, 82.1, 86.9, 91.43, 91.43, 91.78]
y_medflowseg = [90.5, 91.7, 92.9, 93.57, 93.87, 93.87]

colors = {
    "EnsemDiff": "#F6CDCE",
    "SegDiff": "#D25562",
    "MedSegDiff": "#8D322B",
    "MedSegDiff-V2": "#5D9CF4",
    "FlowSDF": "#56F358",
    "MedFlowSeg": "#8A53E0",
}

data = {
    "EnsemDiff": y_ensem_diff,
    "SegDiff": y_seg_diff,
    "MedSegDiff": y_medseg_diff,
    "MedSegDiff-V2": y_medseg_diff_v2,
    "FlowSDF": y_flowsdf,
    "MedFlowSeg": y_medflowseg,
}

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

ax.set_facecolor("#F4F7FB")
fig.patch.set_facecolor("white")

lw = 2
ms = 6

for name, y in data.items():
    ax.plot(
        x_positions,
        y,
        marker="o",
        color=colors[name],
        linewidth=lw,
        markersize=ms,
        label=name,
    )

ax.set_title("Ensemble Performance of Generative Segmentation Methods", fontsize=16, pad=15)
ax.set_xlabel("Number of Sampling Steps", fontsize=14, labelpad=15)
ax.set_ylabel("Average Dice Score on ACDC (%)", fontsize=14, labelpad=15)

ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_yticks(np.arange(70, 96, 5))
ax.set_ylim(68.5, 95.0)
ax.tick_params(axis="y", labelsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#E0E6ED")
ax.spines["bottom"].set_color("#E0E6ED")

ax.grid(color="#E5E9F0", linestyle="--", linewidth=1, alpha=0.9)
ax.set_axisbelow(True)

ax.legend(
    loc="lower right",
    fontsize=12,
    framealpha=1.0,
    edgecolor="#E0E6ED",
    borderpad=0.8,
    facecolor="white",
)

plt.tight_layout()
plt.savefig(
    "/Birmingham/MedFlowSeg/assets/ensemble_performance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()