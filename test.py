# ======================================================
# test_medflowseg_staple_multiclass_nobg_withbgmask.py
# Flow-Matching Medical Segmentation - Test / Inference
# ✅ Multi-sample inference + One-vs-Rest STAPLE fusion (multi-class)
# ✅ NO-BG ONEHOT CHANNEL, BUT WITH EXPLICIT BG IN MASK
# ✅ NEW: Simultaneously predicts and evaluates the Direct Cond Branch 
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
# Decode + Metrics + Visualization
# ======================================================
@torch.no_grad()
def onehot_to_mask_pred_with_bg(
    pred: torch.Tensor,
    thresh: float = 0.5,
) -> torch.Tensor:
    if pred.dim() != 4:
        raise ValueError(f"pred must be (B,C,H,W), got {tuple(pred.shape)}")

    fg_score = pred.clamp(0.0, 1.0)                     
    max_fg = fg_score.max(dim=1).values                 
    fg_idx = torch.argmax(fg_score, dim=1).to(torch.int64) 

    out = torch.zeros_like(fg_idx)                      
    fg_any = max_fg >= thresh
    out[fg_any] = fg_idx[fg_any] + 1                   
    return out


@torch.no_grad()
def onehot_to_mask_gt_with_bg(onehot_gt: torch.Tensor) -> torch.Tensor:
    if onehot_gt.dim() != 4:
        raise ValueError(f"onehot_gt must be (B,C,H,W), got {tuple(onehot_gt.shape)}")

    fg_bin = (onehot_gt > 0.5).to(torch.int8)          
    fg_any = (fg_bin.sum(dim=1) > 0)                   
    fg_idx = torch.argmax(fg_bin, dim=1).to(torch.int64) 

    out = torch.zeros_like(fg_idx)                     
    out[fg_any] = fg_idx[fg_any] + 1                   
    return out


@torch.no_grad()
def per_class_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes_total: int) -> torch.Tensor:
    dices = []
    for c in range(num_classes_total):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())
    return torch.stack(dices)  


@torch.no_grad()
def per_class_iou(pred: torch.Tensor, gt: torch.Tensor, num_classes_total: int) -> torch.Tensor:
    ious = []
    for c in range(num_classes_total):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        union = (p + g - p * g).sum(dim=(1, 2)) + 1e-6
        ious.append((inter / union).mean())
    return torch.stack(ious)  


@torch.no_grad()
def mean_dice_foreground_only(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
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
    ious = []
    for c in range(1, num_classes + 1):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        union = (p + g - p * g).sum(dim=(1, 2)) + 1e-6
        ious.append((inter / union).mean())
    return torch.stack(ious).mean()


def colorize_multiclass_with_bg(mask: np.ndarray, num_classes: int) -> np.ndarray:
    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)  

    if num_classes <= 0:
        return out

    for cid in range(1, num_classes + 1):
        hue = cid / (num_classes + 1)
        color = (np.array(plt.cm.hsv(hue)[:3]) * 255).astype(np.uint8)
        out[mask == cid] = color
    return out


# ======================================================
# STAPLE (Binary) + OVR Multi-class Fusion
# ======================================================
@torch.no_grad()
def staple_binary(
    masks: torch.Tensor,
    iters: int = 15,
    eps: float = 1e-6,
    init_s: float = 0.90,
    init_t: float = 0.90,
) -> torch.Tensor:
    if masks.dim() != 3:
        raise ValueError("masks must be (K,H,W)")
    K, H, W = masks.shape
    m = masks.reshape(K, -1).float()  
    N = m.shape[1]

    p = m.mean(dim=0).clamp(eps, 1 - eps)  
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
    if masks_k.dim() != 4:
        raise ValueError("masks_k must be (K,B,H,W)")
    K, B, H, W = masks_k.shape
    device = masks_k.device

    probs_fg = torch.zeros((B, num_classes, H, W), device=device, dtype=torch.float32)

    for cid in range(1, num_classes + 1):
        bin_k = (masks_k == cid).to(torch.float32)  
        for b in range(B):
            probs_fg[b, cid - 1] = staple_binary(bin_k[:, b], iters=staple_iters)

    maxp, arg = probs_fg.max(dim=1)          
    final = (arg + 1).to(torch.int64)        
    final[maxp < bg_thresh] = 0              

    return final, probs_fg


# ======================================================
# Model wrapper 
# ======================================================
class MedFlowSegTester(nn.Module):
    def __init__(self, num_classes: int = 3, sample_steps: int = 10, freq_attn_depth: int = 1, img_in_ch: int = 1):
        super().__init__()
        self.img_in_ch = img_in_ch
        self.num_classes = num_classes
        self.sample_steps = sample_steps
        self.net = build_model(xt_in_ch=num_classes, freq_attn_depth=freq_attn_depth, img_in_ch=img_in_ch)

    @torch.no_grad()
    # ✅ NEW: Updated return signature to include direct prediction probabilities
    def sample(self, image: torch.Tensor, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Euler integrate dx/dt = v_theta(x, image, t) from x(0)~N(0,I) to t=1.
        returns:
            flow_pred: (B,num_classes,H,W) onehot-like tensor
            direct_pred_prob: (B,num_classes,H,W) probability map from Cond branch
        """
        B, _, H, W = image.shape

        if seed is not None:
            g = torch.Generator(device=image.device)
            g.manual_seed(int(seed))
            x = torch.randn(B, self.num_classes, H, W, device=image.device, generator=g)
        else:
            x = torch.randn(B, self.num_classes, H, W, device=image.device)

        dt = 1.0 / self.sample_steps
        direct_logits = None

        for i in range(self.sample_steps):
            t = torch.full((B,), i * dt, device=image.device)
            # ✅ NEW: Capture both flow velocity (v) and direct seg logits (seg_logits)
            v, seg_logits = self.net(x, image, t)
            
            # Since Cond branch doesn't depend on x_t or t, we just grab the logits once
            if i == 0:
                direct_logits = seg_logits

            x = x + v * dt
            
        return x, torch.sigmoid(direct_logits)


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
    max_vis: int,
    use_ema: bool,
    staple_k: int,
    staple_iters: int,
    bg_thresh: float,
    base_seed: int,
    img_in_ch: int,
    freq_attn_depth: int,
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
        img_in_ch=img_in_ch,
    ).to(device)

    load_lightning_ckpt_into_model(model, ckpt_path=ckpt, use_ema=use_ema)
    model.eval()

    # Tracker for Flow Prediction
    dices_flow_fg: List[float] = []
    ious_flow_fg: List[float] = []
    
    # ✅ NEW: Tracker for Direct Prediction
    dices_direct_fg: List[float] = []
    ious_direct_fg: List[float] = []

    per_class_dices_flow_all = []
    per_class_ious_flow_all = []
    vis_count = 0

    num_classes_total = num_classes + 1  

    for bidx, batch in enumerate(loader):
        image, onehot_gt = batch
        image = image.to(device, non_blocking=True)
        onehot_gt = onehot_gt.to(device, non_blocking=True).float()  

        masks_k = []
        direct_pred_prob = None  # ✅ NEW: Cache direct prediction per batch

        for k in range(staple_k):
            seed = base_seed + (bidx * 1000) + k
            
            # ✅ NEW: Unpack both outputs
            if seed is None:
                pred_k, d_prob = model.sample(image)
            else:
                pred_k, d_prob = model.sample(image, seed=seed)

            # Direct prediction is deterministic across samples, grab it once
            if direct_pred_prob is None:
                direct_pred_prob = d_prob

            mask_k = onehot_to_mask_pred_with_bg(pred_k, thresh=thresh)
            masks_k.append(mask_k)

        masks_k = torch.stack(masks_k, dim=0)  

        # ---------- Predict Masks ----------
        # 1. Flow-Matching + STAPLE
        pred_mask_flow, _ = ovr_staple_fuse_fg_with_bg(
            masks_k=masks_k,
            num_classes=num_classes,
            staple_iters=staple_iters,
            bg_thresh=bg_thresh,
        ) 

        # 2. Direct Cond Prediction (✅ NEW)
        pred_mask_direct = onehot_to_mask_pred_with_bg(direct_pred_prob, thresh=thresh)

        # 3. Ground Truth Mask
        gt_mask = onehot_to_mask_gt_with_bg(onehot_gt)  

        # ---------- Metrics (Flow) ----------
        dice_flow = mean_dice_foreground_only(pred_mask_flow, gt_mask, num_classes=num_classes)
        iou_flow = mean_iou_foreground_only(pred_mask_flow, gt_mask, num_classes=num_classes)
        dices_flow_fg.append(float(dice_flow.item()))
        ious_flow_fg.append(float(iou_flow.item()))

        per_class_dices_flow_all.append(per_class_dice(pred_mask_flow, gt_mask, num_classes_total).cpu())
        per_class_ious_flow_all.append(per_class_iou(pred_mask_flow, gt_mask, num_classes_total).cpu())

        # ---------- Metrics (Direct) ✅ NEW ----------
        dice_direct = mean_dice_foreground_only(pred_mask_direct, gt_mask, num_classes=num_classes)
        iou_direct = mean_iou_foreground_only(pred_mask_direct, gt_mask, num_classes=num_classes)
        dices_direct_fg.append(float(dice_direct.item()))
        ious_direct_fg.append(float(iou_direct.item()))

        # ---------- Visualization (4 columns) ✅ NEW ----------
        if vis_count < max_vis:
            img_np = image.detach().cpu().numpy()[:, 0]        
            gt_np = gt_mask.detach().cpu().numpy()             
            pred_flow_np = pred_mask_flow.detach().cpu().numpy()
            pred_dir_np = pred_mask_direct.detach().cpu().numpy() 

            for i in range(min(image.shape[0], max_vis - vis_count)):
                # Expand to 4 subplots
                fig, axs = plt.subplots(1, 4, figsize=(16, 4))

                axs[0].imshow(img_np[i], cmap="gray")
                axs[0].set_title("Image")
                axs[0].axis("off")

                axs[1].imshow(colorize_multiclass_with_bg(gt_np[i], num_classes=num_classes))
                axs[1].set_title("GT Mask")
                axs[1].axis("off")

                axs[2].imshow(colorize_multiclass_with_bg(pred_dir_np[i], num_classes=num_classes))
                axs[2].set_title("Direct Cond Pred")
                axs[2].axis("off")

                axs[3].imshow(colorize_multiclass_with_bg(pred_flow_np[i], num_classes=num_classes))
                axs[3].set_title(f"Flow Pred (STAPLE K={staple_k})")
                axs[3].axis("off")

                plt.tight_layout()
                plt.savefig(out_dir / "vis" / f"sample_{vis_count:04d}.png", dpi=200)
                plt.close()
                vis_count += 1
                if vis_count >= max_vis:
                    break

        if (bidx + 1) % 10 == 0:
            print(
                f"[{bidx+1}/{len(loader)}] "
                f"Flow Dice: {np.mean(dices_flow_fg):.4f} | "
                f"Direct Dice: {np.mean(dices_direct_fg):.4f}" # ✅ NEW print
            )

    print("\n===== TEST DONE =====")

    per_class_dices_flow_all = torch.stack(per_class_dices_flow_all)  
    per_class_ious_flow_all = torch.stack(per_class_ious_flow_all)    

    mean_dice_c = per_class_dices_flow_all.mean(dim=0)
    std_dice_c = per_class_dices_flow_all.std(dim=0)
    mean_iou_c = per_class_ious_flow_all.mean(dim=0)
    std_iou_c = per_class_ious_flow_all.std(dim=0)

    print("\n===== Flow Prediction Per-class Results (including BG=0) =====")
    for c in range(num_classes_total):
        name = "BG" if c == 0 else f"FG{c}"
        print(
            f"Class {c:02d} ({name}) | "
            f"Dice: {mean_dice_c[c]:.4f} ± {std_dice_c[c]:.4f} | "
            f"IoU:  {mean_iou_c[c]:.4f} ± {std_iou_c[c]:.4f}"
        )

    print(f"\nBatches: {len(dices_flow_fg)}")
    # ✅ NEW: Print Comparison Report
    print(f"Mean Direct Dice (FG 1..{num_classes}): {np.mean(dices_direct_fg):.4f} ± {np.std(dices_direct_fg):.4f}")
    print(f"Mean Flow Dice   (FG 1..{num_classes}): {np.mean(dices_flow_fg):.4f} ± {np.std(dices_flow_fg):.4f}")
    print(f"Mean Direct IoU  (FG 1..{num_classes}): {np.mean(ious_direct_fg):.4f} ± {np.std(ious_direct_fg):.4f}")
    print(f"Mean Flow IoU    (FG 1..{num_classes}): {np.mean(ious_flow_fg):.4f} ± {np.std(ious_flow_fg):.4f}")
    print(f"Saved vis to: {out_dir / 'vis'}")


def main():
    parser = argparse.ArgumentParser("MedFlowSeg Test - NO BG ONEHOT, BUT BG MASK (Flow + Direct Prediction)")

    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning .ckpt")
    parser.add_argument("--data_root", type=str, default="./dataset/ACDC/SDF/test")
    parser.add_argument("--out_dir", type=str, default="./Outputs/MedFlowSeg_test_ACDC")

    parser.add_argument("--num_classes", type=int, default=3, help="number of FOREGROUND classes (onehot channels), bg is implicit")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--img_in_ch", type=int, default=1)
    parser.add_argument("--freq_attn_depth", type=int, default=0)

    parser.add_argument("--thresh", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=28)

    parser.add_argument("--max_vis", type=int, default=16)

    parser.add_argument("--no_ema", action="store_true", help="disable EMA inference (default: EMA on)")

    parser.add_argument("--staple_k", type=int, default=10, help="number of stochastic samples per image (>=3 recommended)")
    parser.add_argument("--staple_iters", type=int, default=15, help="STAPLE EM iterations")
    parser.add_argument("--bg_thresh", type=float, default=0.5, help="if max FG prob < bg_thresh => background(0)")
    parser.add_argument("--base_seed", type=int, default=42, help="base seed for deterministic multi-sampling")

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
        max_vis=args.max_vis,
        use_ema=(not args.no_ema),
        staple_k=args.staple_k,
        staple_iters=args.staple_iters,
        bg_thresh=args.bg_thresh,
        base_seed=args.base_seed,
        img_in_ch=args.img_in_ch,
        freq_attn_depth=args.freq_attn_depth,
    )


if __name__ == "__main__":
    main()