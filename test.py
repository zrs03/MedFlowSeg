# ======================================================
# test_medflowseg_staple_multiclass.py
# Flow-Matching Medical Segmentation - Test / Inference
# ✅ Multi-sample inference + One-vs-Rest STAPLE fusion (multi-class)
#
# Notes:
# - K times sampling per image (different noise seeds)
# - Decode each sample to hard mask (your thresh+argmax)
# - For each foreground class c: run binary STAPLE on (mask==c)
# - Fuse by argmax over class-probabilities; set to background if max prob < bg_thresh
#
# ✅ EMA inference supported (ema_shadow preferred)
# ✅ NEW: SDF visualization (GT + Pred SDF panels)
#   - Pred SDF is visualized from k=0 sample (since STAPLE fuses hard masks only)
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
# Decode + Metrics + Vis
# ======================================================
@torch.no_grad()
def sdf_to_mask_thresh_argmax_pred(sdf: torch.Tensor, thresh: float, fg_factor: float = 3.0) -> torch.Tensor:
    """
    Pred decoding: threshold + argmax (SegDiff-style).
    sdf: (B,C,H,W)
    returns: (B,H,W) int64
    """
    sdf_fg = sdf[:, 1:]  # (B,C-1,H,W)
    fg_bin = (sdf_fg <= (fg_factor * thresh)).to(torch.int8)
    bg = torch.zeros_like(fg_bin[:, :1])
    m = torch.cat([bg, fg_bin], dim=1)  # (B,C,H,W) in {0,1}
    return torch.argmax(m, dim=1)       # (B,H,W)


@torch.no_grad()
def sdf_to_mask_thresh_argmax_gt(sdf_gt: torch.Tensor) -> torch.Tensor:
    """
    GT decoding: signed distance => inside<=0.
    sdf_gt: (B,C,H,W)
    returns: (B,H,W) int64
    """
    sdf_fg = sdf_gt[:, 1:]
    fg_bin = (sdf_fg <= 0.0).to(torch.int8)
    bg = torch.zeros_like(fg_bin[:, :1])
    m = torch.cat([bg, fg_bin], dim=1)
    return torch.argmax(m, dim=1)


@torch.no_grad()
def mean_dice_foreground(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Mean Dice over foreground classes only (1..num_classes-1).
    pred, gt: (B,H,W) int
    """
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())
    return torch.stack(dices).mean()


@torch.no_grad()
def mean_iou_foreground(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Mean IoU over foreground classes only (1..num_classes-1).
    pred, gt: (B,H,W) int
    """
    ious = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        union = (p + g - p * g).sum(dim=(1, 2)) + 1e-6
        ious.append((inter / union).mean())
    return torch.stack(ious).mean()


def colorize_three_fg_else_black(mask: np.ndarray) -> np.ndarray:
    """
    Only color class 1/2/3; everything else (including background 0) is pure black.
    """
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = [255, 0, 0]
    out[mask == 2] = [0, 255, 0]
    out[mask == 3] = [0, 0, 255]
    return out


def sdf_to_vis_rgb(sdf_hw: np.ndarray, clip: float = 0.1) -> np.ndarray:
    """
    sdf_hw: (H,W) float, signed distance
    clip: visualize range [-clip, +clip] (outside will saturate)
    returns: (H,W,3) uint8 using a diverging colormap
    """
    sdf_hw = np.clip(sdf_hw, -clip, clip)
    norm = (sdf_hw + clip) / (2 * clip)  # -> [0,1]
    cmap = plt.get_cmap("seismic")       # blue<0, red>0
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def plot_sdf_panels(
    axs,
    sdf_chw: np.ndarray,
    title_prefix: str,
    max_fg_classes: int = 3,
    clip: float = 0.1,
):
    """
    axs: list/np-array of axes length=max_fg_classes
    sdf_chw: (C,H,W) float (channel0 bg, 1.. fg)
    """
    C = sdf_chw.shape[0]
    for j in range(max_fg_classes):
        c = j + 1
        ax = axs[j]
        if c < C:
            ax.imshow(sdf_to_vis_rgb(sdf_chw[c], clip=clip))
            ax.set_title(f"{title_prefix} SDF c={c}")
        else:
            ax.imshow(np.zeros((sdf_chw.shape[1], sdf_chw.shape[2], 3), dtype=np.uint8))
            ax.set_title(f"{title_prefix} SDF c={c} (N/A)")
        ax.axis("off")


# ======================================================
# STAPLE (Binary) + One-vs-Rest Multi-class Fusion
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

    masks: (K,H,W) float/bool/int in {0,1}
    returns: p_fg (H,W) float in [0,1]
    """
    if masks.dim() != 3:
        raise ValueError("masks must be (K,H,W)")
    K, H, W = masks.shape
    m = masks.reshape(K, -1).float()  # (K,N)
    N = m.shape[1]

    # init posterior with mean vote (better than 0.5 for imbalance)
    p = m.mean(dim=0).clamp(eps, 1 - eps)  # (N,)
    s = torch.full((K,), float(init_s), device=m.device)
    t = torch.full((K,), float(init_t), device=m.device)

    for _ in range(iters):
        # E-step: compute p(z=1|m) per voxel
        log_p1 = torch.zeros(N, device=m.device)
        log_p0 = torch.zeros(N, device=m.device)

        for i in range(K):
            mi = m[i]
            si = s[i].clamp(eps, 1 - eps)
            ti = t[i].clamp(eps, 1 - eps)
            # P(m|z=1): mi~Bernoulli(si)
            log_p1 += mi * torch.log(si) + (1 - mi) * torch.log(1 - si)
            # P(m|z=0): mi~Bernoulli(1-ti) (specificity ti = P(m=0|z=0))
            log_p0 += mi * torch.log(1 - ti) + (1 - mi) * torch.log(ti)

        log_prior1 = torch.log(p.clamp(eps, 1 - eps))
        log_prior0 = torch.log((1 - p).clamp(eps, 1 - eps))

        a = log_prior1 + log_p1
        b = log_prior0 + log_p0
        p_new = torch.sigmoid(a - b).clamp(eps, 1 - eps)

        # M-step: update s_i, t_i
        denom1 = p_new.sum().clamp_min(eps)
        denom0 = (1 - p_new).sum().clamp_min(eps)

        for i in range(K):
            mi = m[i]
            s[i] = ((p_new * mi).sum() / denom1).clamp(eps, 1 - eps)
            t[i] = (((1 - p_new) * (1 - mi)).sum() / denom0).clamp(eps, 1 - eps)

        p = p_new

    return p.reshape(H, W)


@torch.no_grad()
def ovrr_staple_fuse_multiclass(
    masks_k: torch.Tensor,
    num_classes: int,
    staple_iters: int = 15,
    bg_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-vs-Rest STAPLE fusion for multi-class masks.

    masks_k: (K,B,H,W) int in [0..C-1]
    returns:
      final_mask: (B,H,W) int
      probs_fg: (B,C-1,H,W) float  # per-class (1..C-1) foreground probability
    """
    if masks_k.dim() != 4:
        raise ValueError("masks_k must be (K,B,H,W)")
    K, B, H, W = masks_k.shape
    device = masks_k.device

    probs = torch.zeros((B, num_classes - 1, H, W), device=device, dtype=torch.float32)

    # for each fg class c: run binary STAPLE on K masks of (mask==c)
    for c in range(1, num_classes):
        bin_k = (masks_k == c).to(torch.float32)  # (K,B,H,W)
        for b in range(B):
            p_c = staple_binary(bin_k[:, b], iters=staple_iters)  # (H,W)
            probs[b, c - 1] = p_c

    maxp, arg = probs.max(dim=1)      # (B,H,W), (B,H,W) in [0..C-2]
    final = (arg + 1).to(torch.int64) # -> class id 1..C-1
    final[maxp < bg_thresh] = 0       # bg if no class confident enough

    return final, probs


# ======================================================
# Model wrapper (same network + sampling as train)
# ======================================================
class MedFlowSegTester(nn.Module):
    def __init__(self, num_classes: int = 4, sample_steps: int = 10, freq_attn_depth: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.sample_steps = sample_steps
        self.net = build_model(xt_in_ch=num_classes, freq_attn_depth=freq_attn_depth)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """
        Euler integrate dx/dt = v_theta(x, image, t) from x(0)~N(0,I) to t=1.
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
        ema_state = ckpt["ema_shadow"]
        state = _strip_prefixes(ema_state)
        missing, unexpected = model.net.load_state_dict(state, strict=False)
        print("[INFO] Loaded EMA weights from ckpt (ema_shadow).")
    else:
        state_raw = ckpt.get("state_dict", ckpt)
        state = _strip_prefixes(state_raw)
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
    # STAPLE
    staple_k: int,
    staple_iters: int,
    bg_thresh: float,
    base_seed: int,
    freq_attn_depth: int,
    # NEW: SDF visualization params
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

    dices: List[float] = []
    ious: List[float] = []
    vis_count = 0

    for bidx, batch in enumerate(loader):
        image, sdf_gt = batch
        image = image.to(device, non_blocking=True)
        sdf_gt = sdf_gt.to(device, non_blocking=True)

        # ---------- K samples ----------
        masks_k = []
        sdf_pred_vis = None  # (B,C,H,W) for visualization only

        sdf_sum = None
        for k in range(staple_k):
            seed = base_seed + (bidx * 1000) + k
            sdf_pred_k = model.sample(image, seed=seed)  # (B,C,H,W)

            if pred_sdf_mode == "k0":
                if k == 0:
                    sdf_pred_vis = sdf_pred_k
            elif pred_sdf_mode == "mean":
                sdf_sum = sdf_pred_k if sdf_sum is None else (sdf_sum + sdf_pred_k)
            else:
                raise ValueError("pred_sdf_mode must be 'k0' or 'mean'")

            mask_k = sdf_to_mask_thresh_argmax_pred(sdf_pred_k, thresh=thresh, fg_factor=fg_factor)
            masks_k.append(mask_k)

        if pred_sdf_mode == "mean":
            sdf_pred_vis = sdf_sum / float(staple_k)

        masks_k = torch.stack(masks_k, dim=0)  # (K,B,H,W)

        # ---------- One-vs-Rest STAPLE fusion ----------
        pred_mask, _probs = ovrr_staple_fuse_multiclass(
            masks_k=masks_k,
            num_classes=num_classes,
            staple_iters=staple_iters,
            bg_thresh=bg_thresh,
        )

        # GT
        gt_mask = sdf_to_mask_thresh_argmax_gt(sdf_gt)

        dice = mean_dice_foreground(pred_mask, gt_mask, num_classes=num_classes)
        iou = mean_iou_foreground(pred_mask, gt_mask, num_classes=num_classes)
        dices.append(float(dice.item()))
        ious.append(float(iou.item()))

        # ---------- vis ----------
        if vis_count < max_vis:
            img_np = image.detach().cpu().numpy()[:, 0]  # (B,H,W)
            pred_np = pred_mask.detach().cpu().numpy()   # (B,H,W)
            gt_np = gt_mask.detach().cpu().numpy()       # (B,H,W)

            sdf_gt_np = sdf_gt.detach().cpu().numpy()         # (B,C,H,W)
            sdf_pred_np = sdf_pred_vis.detach().cpu().numpy() # (B,C,H,W)

            for i in range(min(image.shape[0], max_vis - vis_count)):
                # Layout:
                # row0: image | GT mask | Pred mask
                # row1: GT  SDF c=1..N
                # row2: Pred SDF c=1..N
                cols = max(3, sdf_vis_fg_max)
                fig, axs = plt.subplots(3, cols, figsize=(4 * cols, 12))

                # If cols > 3, fill extra slots with black on row0
                for j in range(cols):
                    axs[0, j].axis("off")

                axs[0, 0].imshow(img_np[i], cmap="gray")
                axs[0, 0].set_title("Image")
                axs[0, 0].axis("off")

                axs[0, 1].imshow(colorize_three_fg_else_black(gt_np[i]))
                axs[0, 1].set_title("GT Mask")
                axs[0, 1].axis("off")

                axs[0, 2].imshow(colorize_three_fg_else_black(pred_np[i]))
                axs[0, 2].set_title(f"Pred Mask (OVR-STAPLE K={staple_k})")
                axs[0, 2].axis("off")

                # row1 & row2: SDF panels
                plot_sdf_panels(
                    axs=axs[1, :sdf_vis_fg_max],
                    sdf_chw=sdf_gt_np[i],
                    title_prefix="GT",
                    max_fg_classes=sdf_vis_fg_max,
                    clip=sdf_vis_clip,
                )
                plot_sdf_panels(
                    axs=axs[2, :sdf_vis_fg_max],
                    sdf_chw=sdf_pred_np[i],
                    title_prefix=f"Pred({pred_sdf_mode})",
                    max_fg_classes=sdf_vis_fg_max,
                    clip=sdf_vis_clip,
                )

                # If cols > sdf_vis_fg_max, blank the rest in rows 1&2
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
                f"mean dice so far: {np.mean(dices):.4f} | mean IoU so far: {np.mean(ious):.4f}"
            )

    print("\n===== TEST DONE =====")
    print(f"Batches: {len(dices)}")
    print(f"Mean Dice (foreground 1..{num_classes-1}): {np.mean(dices):.4f}  ±  {np.std(dices):.4f}")
    print(f"Mean IoU  (foreground 1..{num_classes-1}): {np.mean(ious):.4f}  ±  {np.std(ious):.4f}")
    print(f"Saved vis to: {out_dir / 'vis'}")


def main():
    parser = argparse.ArgumentParser("MedFlowSeg Test (Multi-sample + One-vs-Rest STAPLE)")

    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning .ckpt")
    parser.add_argument("--data_root", type=str, default="./dataset/REFUGE2/SDF/test")
    parser.add_argument("--out_dir", type=str, default="./Outputs/MedFlowSeg_test_REFUGE2_STAPLE")

    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--freq_attn_depth", type=int, default=1)

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
    parser.add_argument("--bg_thresh", type=float, default=0.5, help="if max class prob < bg_thresh => background")
    parser.add_argument("--base_seed", type=int, default=42, help="base seed for deterministic multi-sampling")

    # NEW: SDF visualization
    parser.add_argument("--sdf_vis_clip", type=float, default=0.1, help="clip SDF to [-clip, +clip] for visualization")
    parser.add_argument("--sdf_vis_fg_max", type=int, default=3, help="visualize at most N foreground SDF channels (c=1..N)")
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