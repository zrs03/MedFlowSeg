# ======================================================
# train_medflowseg.py
# Flow-Matching Medical Segmentation (Lightning v2)
# Threshold + Argmax decoding (SegDiff-style)
# + EMA (Exponential Moving Average)
#
# ✅ EMA default ON with "--no_ema" to disable
# ✅ Save EMA shadow into Lightning checkpoint so test/inference can load EMA weights
#
# ✅ VIS Fix (改法2): 记录并可视化前 N 个 val batches（而不是只记录 batch_idx==0）
#    - 新增参数: --vis_batches  (默认 1)
#
# ✅ NEW: Resume training support (no other changes)
#    - --resume_ckpt path/to/ckpt.ckpt
# ======================================================

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.MedFlowSeg import build_model
from utils.dataset import FolderDataset


# ======================================================
# Utils
# ======================================================
def sdf_to_mask_thresh_argmax_pred(sdf: torch.Tensor, thresh: float, fg_factor: float = 3.0) -> torch.Tensor:
    """
    Pred: threshold + argmax (SegDiff-style)

    sdf: (B, C, H, W), signed distance (normalized to [-1,1] in your preprocessing)
    - Use only foreground channels 1..C-1 for thresholding.
    - Foreground binary: sdf <= fg_factor*thresh
    - Add background channel=0 then argmax to get labels 0..C-1

    returns: (B, H, W) with values in {0..C-1}
    """
    sdf_fg = sdf[:, 1:]
    fg_bin = (sdf_fg <= (fg_factor * thresh)).to(torch.int8)

    bg = torch.zeros_like(fg_bin[:, :1])
    m = torch.cat([bg, fg_bin], dim=1)
    return torch.argmax(m, dim=1)


def sdf_to_mask_thresh_argmax_gt(sdf_gt: torch.Tensor) -> torch.Tensor:
    """
    GT: threshold + argmax.
    Since GT sdf is signed distance: inside is negative, boundary=0, outside positive.
    So foreground is sdf <= 0.

    sdf_gt: (B, C, H, W)
    returns: (B, H, W) with values in {0..C-1}
    """
    sdf_fg = sdf_gt[:, 1:]
    fg_bin = (sdf_fg <= 0.0).to(torch.int8)

    bg = torch.zeros_like(fg_bin[:, :1])
    m = torch.cat([bg, fg_bin], dim=1)
    return torch.argmax(m, dim=1)


def mean_dice_foreground(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Mean Dice over foreground classes only (1..num_classes-1).
    pred, gt: (B,H,W) with labels in {0..num_classes-1}
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
    mask: (H,W) int labels
    """
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)  # black
    out[mask == 1] = [255, 0, 0]
    out[mask == 2] = [0, 255, 0]
    out[mask == 3] = [0, 0, 255]
    return out


class EMA:
    """
    Exponential Moving Average for model parameters.

    Train:
        optimizer.step()
        ema.update(model)

    Val with EMA weights:
        ema.store(model)
        ema.copy_to(model)
        ... val ...
        ema.restore(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def store(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.device))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}


# ======================================================
# Lightning Module
# ======================================================
class LitMedFlowSeg(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 4,
        lr: float = 1e-4,
        sample_steps: int = 10,
        vis_every: int = 5,
        vis_batches: int = 1,  # ✅ 改法2：可视化前 N 个 val batches
        output_root: str = "./Outputs/MedFlowSeg",
        thresh: float = 0.2,
        fg_factor: float = 3.0,
        # EMA (default ON)
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Flow-matching model (expects x_t, image, t)
        self.model = build_model(xt_in_ch=num_classes, freq_attn_depth=1)

        self.num_classes = num_classes
        self.lr = lr
        self.sample_steps = sample_steps

        self.vis_every = int(vis_every)
        self.vis_batches = int(max(1, vis_batches))

        self.output_root = Path(output_root)
        self.vis_root = self.output_root / "vis"
        self.vis_root.mkdir(parents=True, exist_ok=True)

        self.thresh = float(thresh)
        self.fg_factor = float(fg_factor)

        self._val_vis_cache = []

        # EMA (delay init to on_fit_start so device placement is done)
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema: EMA | None = None

        # for resume from ckpt
        self._ema_shadow_from_ckpt = None
        self._ema_decay_from_ckpt = None

    # -----------------------------
    # EMA hooks
    # -----------------------------
    def on_fit_start(self):
        # init EMA after Lightning moves model to correct device(s)
        if self.use_ema and self.ema is None:
            self.ema = EMA(self.model, decay=self.ema_decay)

            # if resumed and ckpt had EMA shadow, restore it
            if self._ema_shadow_from_ckpt is not None:
                self.ema.shadow = {k: v.to(self.device) for k, v in self._ema_shadow_from_ckpt.items()}
            if self._ema_decay_from_ckpt is not None:
                self.ema.decay = float(self._ema_decay_from_ckpt)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # called after optimizer.step() by default -> safe place to update EMA
        if self.use_ema and self.ema is not None:
            self.ema.update(self.model)

    def on_validation_epoch_start(self):
        # swap to EMA weights for the whole validation epoch
        if self.use_ema and self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

    # ✅ save / load EMA into checkpoint so test can use EMA
    def on_save_checkpoint(self, checkpoint):
        if self.use_ema and self.ema is not None:
            checkpoint["ema_shadow"] = {k: v.detach().cpu() for k, v in self.ema.shadow.items()}
            checkpoint["ema_decay"] = float(self.ema.decay)

    def on_load_checkpoint(self, checkpoint):
        # cache; apply in on_fit_start after ema exists and device is set
        self._ema_shadow_from_ckpt = checkpoint.get("ema_shadow", None)
        self._ema_decay_from_ckpt = checkpoint.get("ema_decay", None)

    # --------------------------------------------------
    # Flow Matching sampling (Euler integrate dx/dt = v_theta)
    # --------------------------------------------------
    @torch.no_grad()
    def sample(self, image: torch.Tensor) -> torch.Tensor:
        """
        Start from x(0) ~ N(0, I) and integrate to t=1.
        """
        B, _, H, W = image.shape
        x = torch.randn(B, self.num_classes, H, W, device=image.device)

        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((B,), i * dt, device=image.device)
            v = self.model(x, image, t)
            x = x + v * dt

        return x

    # --------------------------------------------------
    # Training: Conditional Flow Matching (linear bridge)
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        image, x1 = batch  # x1 is sdf_gt, (B,C,H,W)

        B = image.shape[0]
        device = image.device

        t = torch.rand(B, device=device)
        t_ = t[:, None, None, None]

        x0 = torch.randn_like(x1)
        x_t = (1.0 - t_) * x0 + t_ * x1

        v_star = x1 - x0

        v_pred = self.model(x_t, image, t)
        loss = nn.functional.mse_loss(v_pred, v_star)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        image, sdf_gt = batch

        sdf_pred = self.sample(image)

        pred_mask = sdf_to_mask_thresh_argmax_pred(sdf_pred, thresh=self.thresh, fg_factor=self.fg_factor)
        gt_mask = sdf_to_mask_thresh_argmax_gt(sdf_gt)

        dice = mean_dice_foreground(pred_mask, gt_mask, self.num_classes)

        self.log(
            "val_mean_dice",
            dice,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # ✅ 改法2：缓存前 N 个 batch（而不是只缓存 batch_idx==0）
        if batch_idx < self.vis_batches:
            self._val_vis_cache.append(
                {
                    "image": image.detach().cpu(),
                    "pred": pred_mask.detach().cpu(),
                    "gt": gt_mask.detach().cpu(),
                    "batch_idx": int(batch_idx),
                }
            )

    # --------------------------------------------------
    # End of validation epoch
    # --------------------------------------------------
    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        if (epoch + 1) % self.vis_every != 0:
            self._val_vis_cache.clear()
        else:
            if len(self._val_vis_cache) != 0:
                save_dir = self.vis_root / f"epoch_{epoch:04d}"
                save_dir.mkdir(parents=True, exist_ok=True)

                # ✅ 逐个 batch 可视化；每个 batch 最多画 4 张
                for cache in self._val_vis_cache:
                    img = cache["image"][:, 0].numpy()
                    pred = cache["pred"].numpy()
                    gt = cache["gt"].numpy()
                    bidx = cache.get("batch_idx", 0)

                    for i in range(min(4, img.shape[0])):
                        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                        axs[0].imshow(img[i], cmap="gray")
                        axs[0].set_title("Image")
                        axs[0].axis("off")

                        axs[1].imshow(colorize_three_fg_else_black(gt[i]))
                        axs[1].set_title("GT")
                        axs[1].axis("off")

                        axs[2].imshow(colorize_three_fg_else_black(pred[i]))
                        axs[2].set_title("Pred")
                        axs[2].axis("off")

                        plt.tight_layout()
                        plt.savefig(save_dir / f"batch{bidx:03d}_sample{i}.png")
                        plt.close()

            self._val_vis_cache.clear()

        # restore original (non-EMA) weights after validation epoch
        if self.use_ema and self.ema is not None:
            self.ema.restore(self.model)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser("MedFlowSeg Training")

    # -------- Data --------
    parser.add_argument("--train_root", type=str, default="./dataset/ACDC/Aug/train")
    parser.add_argument("--val_root", type=str, default="./dataset/ACDC/SDF/val")

    # -------- Model --------
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--sample_steps", type=int, default=50)

    # -------- Optimization --------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)

    # -------- Output --------
    parser.add_argument("--output_root", type=str, default="./Outputs/MedFlowSeg-ACDC")
    parser.add_argument("--vis_every", type=int, default=5)
    parser.add_argument("--vis_batches", type=int, default=3, help="visualize first N validation batches")

    # threshold for pred decoding (since your sdf is normalized to [-1,1])
    parser.add_argument("--thresh", type=float, default=1e-3)
    parser.add_argument("--fg_factor", type=float, default=1.0)

    # ✅ EMA: default ON, use --no_ema to disable
    parser.add_argument("--no_ema", action="store_true", help="disable EMA (default: EMA on)")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # ✅ Resume
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="path to Lightning checkpoint (.ckpt) to resume from (empty = no resume)",
    )

    # -------- System --------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=4)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    output_root = Path(args.output_root)
    ckpt_root = output_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # -------- Logger --------
    logger = TensorBoardLogger(save_dir=output_root, name="logs")

    # -------- Dataset --------
    train_loader = DataLoader(
        FolderDataset(args.train_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        FolderDataset(args.val_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------- Model --------
    model = LitMedFlowSeg(
        num_classes=args.num_classes,
        lr=args.lr,
        sample_steps=args.sample_steps,
        vis_every=args.vis_every,
        vis_batches=args.vis_batches,  # ✅
        output_root=args.output_root,
        thresh=args.thresh,
        fg_factor=args.fg_factor,
        use_ema=(not args.no_ema),  # ✅ default ON
        ema_decay=args.ema_decay,
    )

    # -------- Checkpoint --------
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_root,
        filename="epoch{epoch:03d}-dice{val_mean_dice:.4f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # -------- Trainer --------
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        strategy="ddp_find_unused_parameters_true",
        max_epochs=args.epochs,
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=20,
    )

    # ✅ Resume support (Lightning will restore optimizer/scheduler/epoch/global_step/etc.)
    ckpt_path = args.resume_ckpt.strip() if isinstance(args.resume_ckpt, str) else ""
    if ckpt_path == "":
        ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
0

if __name__ == "__main__":
    main()