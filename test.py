# ======================================================
# test_medflowseg_staple_multiclass_nobg_withbgmask.py
# Flow-Matching Medical Segmentation - Test / Inference
# ✅ Multi-sample inference + One-vs-Rest STAPLE fusion (multi-class)
# ✅ NO-BG-SDF-CHANNEL, BUT WITH EXPLICIT BG IN MASK (paint black)
#
# Assumption:
# - sdf_gt:   (B,C,H,W)  C = foreground classes only (no explicit bg channel)
# - sdf_pred: (B,C,H,W)  same
#
# Key fix:
# - Previously: pixels where all fg_bin==0 got forced into class0 (a real FG class).
# - Now: we create an explicit background label 0, and shift FG labels to 1..C.
#   - pred decode: if no FG active -> bg(0), else argmax FG -> label (idx+1)
#   - gt decode: if no FG inside -> bg(0), else argmax -> label (idx+1)
#
# STAPLE:
# - Run binary STAPLE only for FG classes (1..C) (one-vs-rest).
# - Fuse by argmax over FG probabilities; if max prob < bg_thresh => background(0)
#
# ✅ EMA inference supported (ema_shadow preferred)
# ✅ SDF visualization (GT + Pred SDF panels)
#   - Visualize FG SDF channels only; titles use class_id = 1..C
# ======================================================

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.MedFlowSeg import build_model
from utils.dataset import FolderDataset


# ======================================================
# Decode + Metrics + Vis  (NO BG SDF CHANNEL, BUT BG IN MASK)
# ======================================================
@torch.no_grad()
def sdf_to_mask_pred_with_bg(
    sdf: torch.Tensor,
    thresh: float,
    fg_factor: float = 3.0,
) -> torch.Tensor:
    """
    Pred decoding with explicit background.

    sdf: (B,C,H,W)  (NO bg channel; C foreground classes)
    returns: (B,H,W) int64 in [0..C]
      0     = background
      1..C  = foreground classes (shifted by +1)
    """
    # fg active per class
    fg_bin = (sdf <= (fg_factor * thresh)).to(torch.int8)  # (B,C,H,W) in {0,1}

    # for pixels with any FG active -> pick argmax among fg_bin (ties go to first)
    fg_any = (fg_bin.sum(dim=1) > 0)  # (B,H,W) bool
    fg_idx = torch.argmax(fg_bin, dim=1).to(torch.int64)  # (B,H,W) in [0..C-1]

    out = torch.zeros_like(fg_idx)  # bg = 0
    out[fg_any] = fg_idx[fg_any] + 1  # shift FG to 1..C
    return out


@torch.no_grad()
def sdf_to_mask_gt_with_bg(sdf_gt: torch.Tensor) -> torch.Tensor:
    """
    GT decoding with explicit background.

    sdf_gt: (B,C,H,W)  (NO bg channel; C foreground classes)
    returns: (B,H,W) int64 in [0..C]
      0     = background
      1..C  = foreground classes (shifted by +1)
    """
    fg_bin = (sdf_gt <= 0.0).to(torch.int8)  # (B,C,H,W)
    fg_any = (fg_bin.sum(dim=1) > 0)         # (B,H,W)
    fg_idx = torch.argmax(fg_bin, dim=1).to(torch.int64)  # (B,H,W) in [0..C-1]

    out = torch.zeros_like(fg_idx)           # bg = 0
    out[fg_any] = fg_idx[fg_any] + 1         # shift to 1..C
    return out


@torch.no_grad()
def per_class_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes_total: int) -> torch.Tensor:
    """
    Dice for each class INCLUDING background.
    pred, gt: (B,H,W) int in [0..num_classes_total-1]
    """
    dices = []
    for c in range(num_classes_total):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())
    return torch.stack(dices)  # (Ctot,)


@torch.no_grad()
def per_class_iou(pred: torch.Tensor, gt: torch.Tensor, num_classes_total: int) -> torch.Tensor:
    """
    IoU for each class INCLUDING background.
    pred, gt: (B,H,W) int in [0..num_classes_total-1]
    """
    ious = []
    for c in range(num_classes_total):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        union = (p + g - p * g).sum(dim=(1, 2)) + 1e-6
        ious.append((inter / union).mean())
    return torch.stack(ious)  # (Ctot,)


@torch.no_grad()
def mean_dice_foreground_only(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Mean Dice over FG classes only: 1..num_classes (bg=0 excluded).
    pred, gt: (B,H,W) in [0..num_classes]
    """
    dices = []
    for c in range(1, num_classes + 1):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())
    return torch.stack(dices).mean()


@torch.no_grad()
def mean_iou_foreground_only(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Mean IoU over FG classes only: 1..num_classes (bg=0 excluded).
    pred, gt: (B,H,W) in [0..num_classes]
    """
    ious = []
    for c in range(1, num_classes + 1):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        union = (p + g - p * g).sum(dim=(1, 2)) + 1e-6
        ious.append((inter / union).mean())
    return torch.stack(ious).mean()


def colorize_multiclass_with_bg(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Mask visualization with explicit BG painted black.
    mask: (H,W) in [0..num_classes]
      0 = bg -> black
      1..num_classes = foreground -> colored
    """
    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)  # bg already black

    if num_classes <= 0:
        return out

    # color fg classes
    for cid in range(1, num_classes + 1):
        hue = cid / (num_classes + 1)  # spread nicely
        color = (np.array(plt.cm.hsv(hue)[:3]) * 255).astype(np.uint8)
        out[mask == cid] = color
    return out


def sdf_to_vis_rgb(sdf_hw: np.ndarray, clip: float = 0.1) -> np.ndarray:
    """
    sdf_hw: (H,W) float, signed distance
    visualize range [-clip, +clip]
    """
    sdf_hw = np.clip(sdf_hw, -clip, clip)
    norm = (sdf_hw + clip) / (2 * clip)  # -> [0,1]
    cmap = plt.get_cmap("seismic")       # blue<0, red>0
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def plot_sdf_panels_fg_only(
    axs,
    sdf_chw: np.ndarray,
    title_prefix: str,
    max_fg_classes: int = 3,
    clip: float = 0.1,
):
    """
    sdf_chw: (C,H,W) where C = num_classes (NO bg channel)
    Visualize fg channels with class_id = 1..C
    """
    C = sdf_chw.shape[0]
    for j in range(max_fg_classes):
        fg_idx = j  # channel index in [0..C-1]
        class_id = j + 1
        ax = axs[j]
        if fg_idx < C:
            ax.imshow(sdf_to_vis_rgb(sdf_chw[fg_idx], clip=clip))
            ax.set_title(f"{title_prefix} SDF class={class_id}")
        else:
            ax.imshow(np.zeros((sdf_chw.shape[1], sdf_chw.shape[2], 3), dtype=np.uint8))
            ax.set_title(f"{title_prefix} SDF class={class_id} (N/A)")
        ax.axis("off")


# ======================================================
# STAPLE (Binary) + OVR Multi-class Fusion
# (FG only; BG decided by threshold)
# ======================================================
@torch.no_grad()
def staple_binary(
    masks: torch.Tensor,
    iters: int = 15,
    eps: float = 1e-6,
    init_s: float = 0.90,
    init_t: float = 0.90,
) -> torch.Tensor:
    """
    Binary STAPLE via EM (simplified, stable for K>=3).

    masks: (K,H,W) in {0,1}
    returns: p_fg (H,W) in [0,1]
    """
    if masks.dim() != 3:
        raise ValueError("masks must be (K,H,W)")
    K, H, W = masks.shape
    m = masks.reshape(K, -1).float()  # (K,N)
    N = m.shape[1]

    p = m.mean(dim=0).clamp(eps, 1 - eps)  # posterior init
    s = torch.full((K,), float(init_s), device=m.device)
    t = torch.full((K,), float(init_t), device=m.device)

    for _ in range(iters):
        log_p1 = torch.zeros(N, device=m.device)
        log_p0 = torch.zeros(N, device=m.device)

        for i in range(K):
            mi = m[i]
            si = s[i].clamp(eps, 1 - eps)
            ti = t[i].clamp(eps, 1 - eps)
            log_p1 += mi * torch.log(si) + (1 - mi) * torch.log(1 - si)
            log_p0 += mi * torch.log(1 - ti) + (1 - mi) * torch.log(ti)

        log_prior1 = torch.log(p.clamp(eps, 1 - eps))
        log_prior0 = torch.log((1 - p).clamp(eps, 1 - eps))

        a = log_prior1 + log_p1
        b = log_prior0 + log_p0
        p_new = torch.sigmoid(a - b).clamp(eps, 1 - eps)

        denom1 = p_new.sum().clamp_min(eps)
        denom0 = (1 - p_new).sum().clamp_min(eps)

        for i in range(K):
            mi = m[i]
            s[i] = ((p_new * mi).sum() / denom1).clamp(eps, 1 - eps)
            t[i] = (((1 - p_new) * (1 - mi)).sum() / denom0).clamp(eps, 1 - eps)

        p = p_new

    return p.reshape(H, W)


@torch.no_grad()
def ovr_staple_fuse_fg_with_bg(
    masks_k: torch.Tensor,
    num_classes: int,
    staple_iters: int = 15,
    bg_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    OVR-STAPLE for FG classes only; BG is decided if max FG prob < bg_thresh.

    masks_k: (K,B,H,W) int in [0..num_classes]
      0=bg, 1..num_classes=fg classes

    returns:
      final_mask: (B,H,W) int in [0..num_classes]
      probs_fg:   (B,num_classes,H,W) float  # probs for fg classes 1..num_classes
    """
    if masks_k.dim() != 4:
        raise ValueError("masks_k must be (K,B,H,W)")
    K, B, H, W = masks_k.shape
    device = masks_k.device

    probs_fg = torch.zeros((B, num_classes, H, W), device=device, dtype=torch.float32)

    # STAPLE per foreground class cid in 1..num_classes
    for cid in range(1, num_classes + 1):
        bin_k = (masks_k == cid).to(torch.float32)  # (K,B,H,W)
        for b in range(B):
            probs_fg[b, cid - 1] = staple_binary(bin_k[:, b], iters=staple_iters)

    maxp, arg = probs_fg.max(dim=1)          # (B,H,W), arg in [0..num_classes-1]
    final = (arg + 1).to(torch.int64)        # -> [1..num_classes]
    final[maxp < bg_thresh] = 0              # -> background

    return final, probs_fg


# ======================================================
# Model wrapper (same network + sampling as train)
# ======================================================
class MedFlowSegTester(nn.Module):
    def __init__(self, num_classes: int = 3, sample_steps: int = 10, freq_attn_depth: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.sample_steps = sample_steps
        # IMPORTANT: model outputs fg SDF channels only
        self.net = build_model(xt_in_ch=num_classes, freq_attn_depth=freq_attn_depth)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Euler integrate dx/dt = v_theta(x, image, t) from x(0)~N(0,I) to t=1.
        returns: (B,num_classes,H,W) predicted SDF (fg only)
        """
        B, _, H, W = image.shape

        if seed is not None:
            g = torch.Generator(device=image.device)
            g.manual_seed(int(seed))
            x = torch.randn(B, self.num_classes, H, W, device=image.device, generator=g)
        else:
            x = torch.randn(B, self.num_classes, H, W, device=image.device)

        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((B,), i * dt, device=image.device)
            v = self.net(x, image, t)
            x = x + v * dt
        return x


def _strip_prefixes(state: dict) -> dict:
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v
        elif k.startswith("net."):
            new_state[k[len("net."):]] = v
        else:
            new_state[k] = v
    return new_state


def load_lightning_ckpt_into_model(model: MedFlowSegTester, ckpt_path: str, use_ema: bool = True) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if use_ema and isinstance(ckpt.get("ema_shadow", None), dict):
        state = _strip_prefixes(ckpt["ema_shadow"])
        missing, unexpected = model.net.load_state_dict(state, strict=False)
        print("[INFO] Loaded EMA weights from ckpt (ema_shadow).")
    else:
        state = _strip_prefixes(ckpt.get("state_dict", ckpt))
        missing, unexpected = model.net.load_state_dict(state, strict=False)
        if use_ema:
            print("[WARN] EMA requested but ckpt has no ema_shadow. Loaded normal weights instead.")
        else:
            print("[INFO] Loaded normal (non-EMA) weights from ckpt.")

    if len(missing) > 0:
        print("[WARN] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if len(unexpected) > 0:
        print("[WARN] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")


# ======================================================
# Test loop
# ======================================================
@torch.no_grad()
def run_test(
    ckpt: str,
    data_root: str,
    out_dir: str,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    sample_steps: int,
    thresh: float,
    fg_factor: float,
    max_vis: int,
    use_ema: bool,
    staple_k: int,
    staple_iters: int,
    bg_thresh: float,
    base_seed: int,
    freq_attn_depth: int,
    sdf_vis_clip: float = 0.1,
    sdf_vis_fg_max: int = 3,
    pred_sdf_mode: str = "k0",  # "k0" or "mean"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        FolderDataset(data_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = MedFlowSegTester(
        num_classes=num_classes,
        sample_steps=sample_steps,
        freq_attn_depth=freq_attn_depth,
    ).to(device)

    load_lightning_ckpt_into_model(model, ckpt_path=ckpt, use_ema=use_ema)
    model.eval()

    dices_fg: List[float] = []
    ious_fg: List[float] = []
    per_class_dices_all = []
    per_class_ious_all = []
    vis_count = 0

    num_classes_total = num_classes + 1  # include bg(0) + fg(1..num_classes)

    for bidx, batch in enumerate(loader):
        image, sdf_gt = batch
        image = image.to(device, non_blocking=True)
        sdf_gt = sdf_gt.to(device, non_blocking=True)  # (B,num_classes,H,W) no-bg-sdf

        # ---------- K samples ----------
        masks_k = []
        sdf_pred_vis = None
        sdf_sum = None

        for k in range(staple_k):
            seed = base_seed + (bidx * 1000) + k
            sdf_pred_k = model.sample(image, seed=seed)  # (B,num_classes,H,W)

            if pred_sdf_mode == "k0":
                if k == 0:
                    sdf_pred_vis = sdf_pred_k
            elif pred_sdf_mode == "mean":
                sdf_sum = sdf_pred_k if sdf_sum is None else (sdf_sum + sdf_pred_k)
            else:
                raise ValueError("pred_sdf_mode must be 'k0' or 'mean'")

            mask_k = sdf_to_mask_pred_with_bg(sdf_pred_k, thresh=thresh, fg_factor=fg_factor)  # (B,H,W) in [0..num_classes]
            masks_k.append(mask_k)

        if pred_sdf_mode == "mean":
            sdf_pred_vis = sdf_sum / float(staple_k)

        masks_k = torch.stack(masks_k, dim=0)  # (K,B,H,W)

        # ---------- OVR-STAPLE over FG + BG fallback ----------
        pred_mask, _probs_fg = ovr_staple_fuse_fg_with_bg(
            masks_k=masks_k,
            num_classes=num_classes,
            staple_iters=staple_iters,
            bg_thresh=bg_thresh,
        )  # (B,H,W) in [0..num_classes]

        # GT mask with explicit bg
        gt_mask = sdf_to_mask_gt_with_bg(sdf_gt)  # (B,H,W) in [0..num_classes]

        # Metrics (foreground-only mean)
        dice_fg = mean_dice_foreground_only(pred_mask, gt_mask, num_classes=num_classes)
        iou_fg = mean_iou_foreground_only(pred_mask, gt_mask, num_classes=num_classes)
        dices_fg.append(float(dice_fg.item()))
        ious_fg.append(float(iou_fg.item()))

        # Per-class including bg
        per_class_dices_all.append(per_class_dice(pred_mask, gt_mask, num_classes_total).cpu())
        per_class_ious_all.append(per_class_iou(pred_mask, gt_mask, num_classes_total).cpu())

        # ---------- vis ----------
        if vis_count < max_vis:
            img_np = image.detach().cpu().numpy()[:, 0]        # (B,H,W)
            pred_np = pred_mask.detach().cpu().numpy()          # (B,H,W) in [0..num_classes]
            gt_np = gt_mask.detach().cpu().numpy()              # (B,H,W) in [0..num_classes]
            sdf_gt_np = sdf_gt.detach().cpu().numpy()           # (B,num_classes,H,W)
            sdf_pred_np = sdf_pred_vis.detach().cpu().numpy()   # (B,num_classes,H,W)

            for i in range(min(image.shape[0], max_vis - vis_count)):
                cols = max(3, sdf_vis_fg_max)
                fig, axs = plt.subplots(3, cols, figsize=(4 * cols, 12))

                # row0: image | GT mask | Pred mask
                for j in range(cols):
                    axs[0, j].axis("off")

                axs[0, 0].imshow(img_np[i], cmap="gray")
                axs[0, 0].set_title("Image")
                axs[0, 0].axis("off")

                axs[0, 1].imshow(colorize_multiclass_with_bg(gt_np[i], num_classes=num_classes))
                axs[0, 1].set_title("GT Mask (bg=black)")
                axs[0, 1].axis("off")

                axs[0, 2].imshow(colorize_multiclass_with_bg(pred_np[i], num_classes=num_classes))
                axs[0, 2].set_title(f"Pred Mask (OVR-STAPLE, bg=black, K={staple_k})")
                axs[0, 2].axis("off")

                # row1 & row2: SDF panels (FG only)
                plot_sdf_panels_fg_only(
                    axs=axs[1, :sdf_vis_fg_max],
                    sdf_chw=sdf_gt_np[i],
                    title_prefix="GT",
                    max_fg_classes=sdf_vis_fg_max,
                    clip=sdf_vis_clip,
                )
                plot_sdf_panels_fg_only(
                    axs=axs[2, :sdf_vis_fg_max],
                    sdf_chw=sdf_pred_np[i],
                    title_prefix=f"Pred({pred_sdf_mode})",
                    max_fg_classes=sdf_vis_fg_max,
                    clip=sdf_vis_clip,
                )

                # fill remaining columns
                for j in range(sdf_vis_fg_max, cols):
                    axs[1, j].imshow(np.zeros((img_np[i].shape[0], img_np[i].shape[1], 3), dtype=np.uint8))
                    axs[1, j].set_title("N/A")
                    axs[1, j].axis("off")
                    axs[2, j].imshow(np.zeros((img_np[i].shape[0], img_np[i].shape[1], 3), dtype=np.uint8))
                    axs[2, j].set_title("N/A")
                    axs[2, j].axis("off")

                plt.tight_layout()
                plt.savefig(out_dir / "vis" / f"sample_{vis_count:04d}.png", dpi=200)
                plt.close()
                vis_count += 1
                if vis_count >= max_vis:
                    break

        if (bidx + 1) % 10 == 0:
            print(
                f"[{bidx+1}/{len(loader)}] "
                f"mean Dice(FG) so far: {np.mean(dices_fg):.4f} | mean IoU(FG) so far: {np.mean(ious_fg):.4f}"
            )

    print("\n===== TEST DONE (NO BG SDF, BUT BG MASK) =====")

    per_class_dices_all = torch.stack(per_class_dices_all)  # (N,Ctot)
    per_class_ious_all = torch.stack(per_class_ious_all)    # (N,Ctot)

    mean_dice_c = per_class_dices_all.mean(dim=0)
    std_dice_c = per_class_dices_all.std(dim=0)

    mean_iou_c = per_class_ious_all.mean(dim=0)
    std_iou_c = per_class_ious_all.std(dim=0)

    print("\n===== Per-class Results (including BG=0) =====")
    for c in range(num_classes_total):
        name = "BG" if c == 0 else f"FG{c}"
        print(
            f"Class {c:02d} ({name}) | "
            f"Dice: {mean_dice_c[c]:.4f} ± {std_dice_c[c]:.4f} | "
            f"IoU:  {mean_iou_c[c]:.4f} ± {std_iou_c[c]:.4f}"
        )

    print(f"\nBatches: {len(dices_fg)}")
    print(f"Mean Dice (FG only 1..{num_classes}): {np.mean(dices_fg):.4f} ± {np.std(dices_fg):.4f}")
    print(f"Mean IoU  (FG only 1..{num_classes}): {np.mean(ious_fg):.4f} ± {np.std(ious_fg):.4f}")
    print(f"Saved vis to: {out_dir / 'vis'}")


def main():
    parser = argparse.ArgumentParser("MedFlowSeg Test (Multi-sample + OVR-STAPLE) - NO BG SDF, BUT BG MASK")

    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning .ckpt")
    parser.add_argument("--data_root", type=str, default="./dataset/ACDC/SDF/test")
    parser.add_argument("--out_dir", type=str, default="./Outputs/MedFlowSeg_test_ACDC")

    parser.add_argument("--num_classes", type=int, default=3, help="number of FOREGROUND classes (SDF channels), bg is implicit")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--freq_attn_depth", type=int, default=0)

    parser.add_argument("--thresh", type=float, default=0.01)
    parser.add_argument("--fg_factor", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=28)

    parser.add_argument("--max_vis", type=int, default=16)

    # EMA
    parser.add_argument("--no_ema", action="store_true", help="disable EMA inference (default: EMA on)")

    # STAPLE
    parser.add_argument("--staple_k", type=int, default=10, help="number of stochastic samples per image (>=3 recommended)")
    parser.add_argument("--staple_iters", type=int, default=15, help="STAPLE EM iterations")
    parser.add_argument("--bg_thresh", type=float, default=0.5, help="if max FG prob < bg_thresh => background(0)")
    parser.add_argument("--base_seed", type=int, default=42, help="base seed for deterministic multi-sampling")

    # SDF visualization
    parser.add_argument("--sdf_vis_clip", type=float, default=0.1, help="clip SDF to [-clip, +clip] for visualization")
    parser.add_argument("--sdf_vis_fg_max", type=int, default=3, help="visualize at most N FG SDF channels")
    parser.add_argument(
        "--pred_sdf_mode",
        type=str,
        default="k0",
        choices=["k0", "mean"],
        help="which predicted SDF to visualize: k0 (first sample) or mean (average over K samples)",
    )

    args = parser.parse_args()
    pl.seed_everything(42)

    if args.staple_k < 2:
        raise ValueError("--staple_k must be >= 2 (>=3 recommended for STAPLE stability)")

    run_test(
        ckpt=args.ckpt,
        data_root=args.data_root,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        sample_steps=args.sample_steps,
        thresh=args.thresh,
        fg_factor=args.fg_factor,
        max_vis=args.max_vis,
        use_ema=(not args.no_ema),
        staple_k=args.staple_k,
        staple_iters=args.staple_iters,
        bg_thresh=args.bg_thresh,
        base_seed=args.base_seed,
        freq_attn_depth=args.freq_attn_depth,
        sdf_vis_clip=args.sdf_vis_clip,
        sdf_vis_fg_max=args.sdf_vis_fg_max,
        pred_sdf_mode=args.pred_sdf_mode,
    )


if __name__ == "__main__":
    main()