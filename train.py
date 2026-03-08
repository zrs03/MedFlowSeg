# ======================================================
# train_medflowseg.py
# Flow-Matching Medical Segmentation (Lightning v2)
# ✅ Adapted for NO-BACKGROUND channels in sdf/onehot (C=3 for ACDC labels 1/2/3)
# ✅ Added sample-wise class weighting based on per-sample class pixel ratio
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
def sdf_to_mask_thresh_argmax_pred_no_bg(
    sdf: torch.Tensor, thresh: float, fg_factor: float = 3.0
) -> torch.Tensor:
    """
    Pred decoding for NO-BG sdf channels.

    sdf: (B, C, H, W) where C=3 corresponds to labels {1,2,3} (no background channel stored).
    We create an explicit background channel so argmax can output {0..C}:
      - Foreground binary per class: sdf <= fg_factor*thresh
      - Background binary: 1 if all foreground binaries are 0 at that pixel
      - Argmax over (bg + fg_bins) -> labels in {0..C}
        where:
          0 -> background
          1 -> class-1 (ACDC label 1)
          2 -> class-2 (ACDC label 2)
          3 -> class-3 (ACDC label 3)
    """
    if sdf.ndim != 4:
        raise ValueError(f"sdf must be (B,C,H,W), got {tuple(sdf.shape)}")

    fg_bin = (sdf <= (fg_factor * thresh)).to(torch.int8)  # (B,C,H,W)
    bg_bin = (fg_bin.sum(dim=1, keepdim=True) == 0).to(torch.int8)  # (B,1,H,W)

    m = torch.cat([bg_bin, fg_bin], dim=1)  # (B,1+C,H,W)
    return torch.argmax(m, dim=1)  # (B,H,W) in {0..C}


def sdf_to_mask_thresh_argmax_gt_no_bg(sdf_gt: torch.Tensor) -> torch.Tensor:
    """
    GT decoding for NO-BG sdf channels.

    sdf_gt: (B, C, H, W) where C=3 corresponds to labels {1,2,3}.
    For signed distance maps: inside is negative, boundary=0, outside positive.
    So foreground is sdf <= 0.

    We add explicit background channel:
      bg=1 where all fg bins are 0.
    """
    if sdf_gt.ndim != 4:
        raise ValueError(f"sdf_gt must be (B,C,H,W), got {tuple(sdf_gt.shape)}")

    fg_bin = (sdf_gt <= 0.0).to(torch.int8)  # (B,C,H,W)
    bg_bin = (fg_bin.sum(dim=1, keepdim=True) == 0).to(torch.int8)  # (B,1,H,W)

    m = torch.cat([bg_bin, fg_bin], dim=1)  # (B,1+C,H,W)
    return torch.argmax(m, dim=1)  # (B,H,W) in {0..C}


def mean_dice_foreground(pred: torch.Tensor, gt: torch.Tensor, num_classes_with_bg: int) -> torch.Tensor:
    """
    Mean Dice over foreground classes only (1..num_classes_with_bg-1),
    skip classes where GT is empty.

    pred, gt: (B,H,W) with labels in {0..num_classes_with_bg-1}
    """
    dices = []
    for c in range(1, num_classes_with_bg):
        g = (gt == c).float()
        p = (pred == c).float()
        if g.sum() == 0:
            continue

        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())

    if len(dices) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(dices).mean()


def colorize_multiclass(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    通用多类别可视化
    - background (0) 为黑色
    - 支持任意 num_classes
    - 自动生成不同颜色
    """
    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    if num_classes <= 1:
        return out

    for c in range(1, num_classes):
        hue = c / num_classes
        color = np.array(plt.cm.hsv(hue)[:3]) * 255
        out[mask == c] = color.astype(np.uint8)

    return out


def compute_samplewise_class_weights_from_sdf_gt(
    sdf_gt: torch.Tensor,
    eps: float = 1e-6,
    max_class_weight: float = 5.0,
) -> torch.Tensor:
    """
    根据每个样本内各类别像素比例，计算每个样本自己的通道权重。

    sdf_gt: (B,C,H,W), 无背景通道
    前景定义为 sdf <= 0

    返回:
        weights: (B,C)

    逻辑:
        counts_c = 每个样本中每个类的前景像素数
        ratio_c = counts_c / sum(counts_all_classes)
        weight_c = 1 / (ratio_c + eps)

        然后对每个样本做归一化:
            让该样本所有出现类别的平均权重 = 1

        最后截断到 max_class_weight
    """
    if sdf_gt.ndim != 4:
        raise ValueError(f"sdf_gt must be (B,C,H,W), got {tuple(sdf_gt.shape)}")

    fg = (sdf_gt <= 0.0).float()                 # (B,C,H,W)
    counts = fg.sum(dim=(2, 3))                  # (B,C)
    present = counts > 0                         # (B,C)

    total_fg = counts.sum(dim=1, keepdim=True)   # (B,1)
    ratios = counts / (total_fg + eps)           # (B,C)

    weights = torch.zeros_like(ratios)           # (B,C)
    weights[present] = 1.0 / (ratios[present] + eps)

    present_count = present.sum(dim=1, keepdim=True)  # (B,1)
    weight_sum = weights.sum(dim=1, keepdim=True)     # (B,1)

    valid = present_count > 0
    norm = torch.ones_like(weight_sum)
    norm[valid] = weight_sum[valid] / present_count[valid].float()

    weights = weights / (norm + eps)

    # 如果某个样本没有任何前景，回退成全1
    no_fg = present_count.squeeze(1) == 0
    if no_fg.any():
        weights[no_fg] = 1.0

    weights = torch.clamp(weights, min=0.0, max=max_class_weight)
    return weights


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
class MedFlowSeg(pl.LightningModule):
    def __init__(
        self,
        img_in_ch: int = 1,
        num_classes: int = 3,  # foreground channels count, no bg stored
        lr: float = 1e-4,
        sample_steps: int = 10,
        vis_every: int = 5,
        vis_batches: int = 1,
        output_root: str = "./Outputs/MedFlowSeg",
        thresh: float = 0.2,
        fg_factor: float = 3.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        class_weight_eps: float = 1e-6,
        max_class_weight: float = 5.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(xt_in_ch=num_classes, img_in_ch=img_in_ch)

        self.num_classes_fg = int(num_classes)
        self.num_classes_with_bg = self.num_classes_fg + 1

        self.lr = lr
        self.sample_steps = sample_steps

        self.vis_every = int(vis_every)
        self.vis_batches = int(max(1, vis_batches))

        self.output_root = Path(output_root)
        self.vis_root = self.output_root / "vis"
        self.vis_root.mkdir(parents=True, exist_ok=True)

        self.thresh = float(thresh)
        self.fg_factor = float(fg_factor)

        self.class_weight_eps = float(class_weight_eps)
        self.max_class_weight = float(max_class_weight)

        self._val_vis_cache = []

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema: EMA | None = None

        self._ema_shadow_from_ckpt = None
        self._ema_decay_from_ckpt = None

    # -----------------------------
    # EMA hooks
    # -----------------------------
    def on_fit_start(self):
        if self.use_ema and self.ema is None:
            self.ema = EMA(self.model, decay=self.ema_decay)

            if self._ema_shadow_from_ckpt is not None:
                self.ema.shadow = {k: v.to(self.device) for k, v in self._ema_shadow_from_ckpt.items()}
            if self._ema_decay_from_ckpt is not None:
                self.ema.decay = float(self._ema_decay_from_ckpt)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema and self.ema is not None:
            self.ema.update(self.model)

    def on_validation_epoch_start(self):
        if self.use_ema and self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema and self.ema is not None:
            checkpoint["ema_shadow"] = {k: v.detach().cpu() for k, v in self.ema.shadow.items()}
            checkpoint["ema_decay"] = float(self.ema.decay)

    def on_load_checkpoint(self, checkpoint):
        self._ema_shadow_from_ckpt = checkpoint.get("ema_shadow", None)
        self._ema_decay_from_ckpt = checkpoint.get("ema_decay", None)

    # --------------------------------------------------
    # Flow Matching sampling
    # --------------------------------------------------
    @torch.no_grad()
    def sample(self, image: torch.Tensor) -> torch.Tensor:
        """
        Start from x(0) ~ N(0, I) and integrate to t=1.
        Output sdf_pred has shape (B, C, H, W) where C=num_classes_fg (no bg stored).
        """
        B, _, H, W = image.shape
        x = torch.randn(B, self.num_classes_fg, H, W, device=image.device)

        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((B,), i * dt, device=image.device)
            v = self.model(x, image, t)
            x = x + v * dt

        return x

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        image, x1 = batch  # x1: sdf_gt, (B,C,H,W)

        B = image.shape[0]
        device = image.device

        t = torch.rand(B, device=device)
        t_ = t[:, None, None, None]

        x0 = torch.randn_like(x1)
        x_t = (1.0 - t_) * x0 + t_ * x1

        v_star = x1 - x0
        v_pred = self.model(x_t, image, t)

        # 按每个样本内类别像素比例，动态计算通道权重
        class_w = compute_samplewise_class_weights_from_sdf_gt(
            x1,
            eps=self.class_weight_eps,
            max_class_weight=self.max_class_weight,
        )  # (B,C)

        per_pixel = (v_pred - v_star) ** 2        # (B,C,H,W)
        class_w = class_w[:, :, None, None]       # (B,C,1,1)

        loss = (per_pixel * class_w).mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/class_w_mean", class_w.mean(), prog_bar=False)
        self.log("train/class_w_max", class_w.max(), prog_bar=False)
        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        image, sdf_gt = batch  # sdf_gt: (B,C,H,W), C=num_classes_fg (no bg)

        sdf_pred = self.sample(image)

        pred_mask = sdf_to_mask_thresh_argmax_pred_no_bg(
            sdf_pred, thresh=self.thresh, fg_factor=self.fg_factor
        )
        gt_mask = sdf_to_mask_thresh_argmax_gt_no_bg(sdf_gt)

        dice = mean_dice_foreground(pred_mask, gt_mask, self.num_classes_with_bg)

        self.log(
            "val_mean_dice",
            dice,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

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

                        axs[1].imshow(colorize_multiclass(gt[i], self.num_classes_with_bg))
                        axs[1].set_title("GT")
                        axs[1].axis("off")

                        axs[2].imshow(colorize_multiclass(pred[i], self.num_classes_with_bg))
                        axs[2].set_title("Pred")
                        axs[2].axis("off")

                        plt.tight_layout()
                        plt.savefig(save_dir / f"batch{bidx:03d}_sample{i}.png")
                        plt.close()

            self._val_vis_cache.clear()

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
    parser.add_argument("--img_in_ch", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--sample_steps", type=int, default=50)

    # -------- Optimization --------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=144)

    # -------- Output --------
    parser.add_argument("--output_root", type=str, default="./Outputs/MedFlowSeg-ACDC")
    parser.add_argument("--vis_every", type=int, default=10)
    parser.add_argument("--vis_batches", type=int, default=2, help="visualize first N validation batches")

    # threshold for pred decoding
    parser.add_argument("--thresh", type=float, default=1e-3)
    parser.add_argument("--fg_factor", type=float, default=1.0)

    # sample-wise class weighting
    parser.add_argument("--class_weight_eps", type=float, default=1e-6)
    parser.add_argument("--max_class_weight", type=float, default=5.0)

    # EMA
    parser.add_argument("--no_ema", action="store_true", help="disable EMA (default: EMA on)")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Resume
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="",
        help="path to Lightning checkpoint (.ckpt) to resume from (empty = no resume)",
    )

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=4)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    output_root = Path(args.output_root)
    ckpt_root = output_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir=output_root, name="logs")

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

    model = MedFlowSeg(
        img_in_ch=args.img_in_ch,
        num_classes=args.num_classes,
        lr=args.lr,
        sample_steps=args.sample_steps,
        vis_every=args.vis_every,
        vis_batches=args.vis_batches,
        output_root=args.output_root,
        thresh=args.thresh,
        fg_factor=args.fg_factor,
        use_ema=(not args.no_ema),
        ema_decay=args.ema_decay,
        class_weight_eps=args.class_weight_eps,
        max_class_weight=args.max_class_weight,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_root,
        filename="epoch{epoch:03d}-dice{val_mean_dice:.4f}",
        monitor="val_mean_dice",
        mode="max",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        max_epochs=args.epochs,
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=20,
    )

    ckpt_path = args.resume_ckpt.strip() if isinstance(args.resume_ckpt, str) else ""
    if ckpt_path == "":
        ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()