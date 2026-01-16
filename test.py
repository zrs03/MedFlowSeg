# ======================================================
# test_medflowseg.py
# Flow-Matching Medical Segmentation - Test / Inference
# Threshold + Argmax decoding (SegDiff-style)
#
# ✅ EMA inference:
# - If ckpt contains "ema_shadow" (saved by train script), load EMA weights by default.
# - Fallback to normal "state_dict" weights if ema not found.
# - You can force disable EMA with --no_ema
# ======================================================

import argparse
from pathlib import Path

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
    """
    sdf_fg = sdf[:, 1:]
    fg_bin = (sdf_fg <= (fg_factor * thresh)).to(torch.int8)
    bg = torch.zeros_like(fg_bin[:, :1])
    m = torch.cat([bg, fg_bin], dim=1)
    return torch.argmax(m, dim=1)


@torch.no_grad()
def sdf_to_mask_thresh_argmax_gt(sdf_gt: torch.Tensor) -> torch.Tensor:
    """
    GT decoding: signed distance => inside<=0.
    sdf_gt: (B,C,H,W)
    """
    sdf_fg = sdf_gt[:, 1:]
    fg_bin = (sdf_fg <= 0.0).to(torch.int8)
    bg = torch.zeros_like(fg_bin[:, :1])
    m = torch.cat([bg, fg_bin], dim=1)
    return torch.argmax(m, dim=1)


@torch.no_grad()
def mean_dice_foreground(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """
    Mean Dice over foreground classes only (1..num_classes-1).
    pred, gt: (B,H,W)
    """
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())
    return torch.stack(dices).mean()


def colorize_three_fg_else_black(mask: np.ndarray) -> np.ndarray:
    """
    Only color class 1/2/3; everything else (including background 0) is pure black.
    """
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask == 1] = [255, 0, 0]
    out[mask == 2] = [0, 255, 0]
    out[mask == 3] = [0, 0, 255]
    return out


# ======================================================
# Model wrapper (same network + sampling as train)
# ======================================================
class MedFlowSegTester(nn.Module):
    def __init__(self, num_classes: int = 4, sample_steps: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.sample_steps = sample_steps
        self.net = build_model(xt_in_ch=num_classes, freq_attn_depth=2)

    @torch.no_grad()
    def sample(self, image: torch.Tensor) -> torch.Tensor:
        """
        Euler integrate dx/dt = v_theta(x, image, t) from x(0)~N(0,I) to t=1.
        """
        B, _, H, W = image.shape
        x = torch.randn(B, self.num_classes, H, W, device=image.device)

        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((B,), i * dt, device=image.device)
            v = self.net(x, image, t)
            x = x + v * dt
        return x


def _strip_prefixes(state: dict) -> dict:
    """
    Strip common Lightning prefixes to match model.net keys.
    """
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
    """
    Load Lightning .ckpt weights into our plain nn.Module wrapper.

    ✅ If use_ema=True and ckpt contains "ema_shadow", load EMA weights.
    ✅ Else fallback to ckpt["state_dict"].
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) Try EMA
    if use_ema and isinstance(ckpt.get("ema_shadow", None), dict):
        ema_state = ckpt["ema_shadow"]
        state = _strip_prefixes(ema_state)
        missing, unexpected = model.net.load_state_dict(state, strict=False)
        print("[INFO] Loaded EMA weights from ckpt (ema_shadow).")
    else:
        # 2) Fallback to normal weights
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    # data
    loader = DataLoader(
        FolderDataset(data_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # model
    model = MedFlowSegTester(num_classes=num_classes, sample_steps=sample_steps).to(device)
    load_lightning_ckpt_into_model(model, ckpt_path=ckpt, use_ema=use_ema)
    model.eval()

    dices = []
    vis_count = 0

    for bidx, batch in enumerate(loader):
        image, sdf_gt = batch
        image = image.to(device, non_blocking=True)
        sdf_gt = sdf_gt.to(device, non_blocking=True)

        sdf_pred = model.sample(image)

        pred_mask = sdf_to_mask_thresh_argmax_pred(sdf_pred, thresh=thresh, fg_factor=fg_factor)
        gt_mask = sdf_to_mask_thresh_argmax_gt(sdf_gt)

        dice = mean_dice_foreground(pred_mask, gt_mask, num_classes=num_classes)
        dices.append(dice.item())

        # vis
        if vis_count < max_vis:
            img_np = image.detach().cpu().numpy()[:, 0]
            pred_np = pred_mask.detach().cpu().numpy()
            gt_np = gt_mask.detach().cpu().numpy()

            for i in range(min(image.shape[0], max_vis - vis_count)):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img_np[i], cmap="gray")
                axs[0].set_title("Image")
                axs[0].axis("off")

                axs[1].imshow(colorize_three_fg_else_black(gt_np[i]))
                axs[1].set_title("GT (thresh+argmax)")
                axs[1].axis("off")

                axs[2].imshow(colorize_three_fg_else_black(pred_np[i]))
                axs[2].set_title("Pred (thresh+argmax)")
                axs[2].axis("off")

                plt.tight_layout()
                plt.savefig(out_dir / "vis" / f"sample_{vis_count:04d}.png", dpi=200)
                plt.close()
                vis_count += 1
                if vis_count >= max_vis:
                    break

        if (bidx + 1) % 10 == 0:
            print(f"[{bidx+1}/{len(loader)}] mean dice so far: {np.mean(dices):.4f}")

    print("\n===== TEST DONE =====")
    print(f"Batches: {len(dices)}")
    print(f"Mean Dice (foreground 1..{num_classes-1}): {np.mean(dices):.4f}  ±  {np.std(dices):.4f}")
    print(f"Saved vis to: {out_dir / 'vis'}")


def main():
    parser = argparse.ArgumentParser("MedFlowSeg Test (Threshold+Argmax)")

    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning .ckpt")
    parser.add_argument("--data_root", type=str, default="./dataset/ACDC/SDF/test",
                        help="e.g. ./dataset/ACDC/SDF/test or val")
    parser.add_argument("--out_dir", type=str, default="./Outputs/MedFlowSeg_test")

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)

    parser.add_argument("--thresh", type=float, default=0.01, help="pred threshold; sdf is normalized to [-1,1]")
    parser.add_argument("--fg_factor", type=float, default=3.0, help="pred uses sdf <= fg_factor*thresh")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_vis", type=int, default=16, help="save at most N visualizations")

    # ✅ EMA: default ON, use --no_ema to disable
    parser.add_argument("--no_ema", action="store_true", help="disable EMA inference (default: EMA on)")

    args = parser.parse_args()
    pl.seed_everything(42)

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
        use_ema=(not args.no_ema),  # ✅ default ON
    )


if __name__ == "__main__":
    main()