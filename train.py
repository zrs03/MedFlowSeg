# ======================================================
# train_medflowseg.py
# Flow-Matching Medical Segmentation (Lightning v2)
# ✅ OneHot version (NO-BACKGROUND channels, NO SDF)
# ✅ Removed class-wise weighting
# ✅ With probability seg_prob, add seg-head loss: L_dice + beta * L_ce
# ✅ Compatible with model returning: (v_pred, seg_logits)
# ✅ ALIGNED VISUALIZATION: Explicit --rgb_vis, --vis_alpha, and color logic
# ======================================================

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
def onehot_to_mask_argmax_pred_no_bg(
    pred: torch.Tensor, thresh: float = 0.5
) -> torch.Tensor:
    if pred.ndim != 4:
        raise ValueError(f"pred must be (B,C,H,W), got {tuple(pred.shape)}")

    fg_score = pred.clamp(0.0, 1.0)  # (B,C,H,W)
    max_fg = fg_score.max(dim=1, keepdim=True).values  # (B,1,H,W)
    bg_score = (max_fg < thresh).float()  # (B,1,H,W)

    m = torch.cat([bg_score, fg_score], dim=1)  # (B,1+C,H,W)
    return torch.argmax(m, dim=1)  # (B,H,W) in {0..C}


def onehot_to_mask_argmax_gt_no_bg(onehot_gt: torch.Tensor) -> torch.Tensor:
    if onehot_gt.ndim != 4:
        raise ValueError(f"onehot_gt must be (B,C,H,W), got {tuple(onehot_gt.shape)}")

    fg_bin = (onehot_gt > 0.5).to(torch.int8)  # (B,C,H,W)
    bg_bin = (fg_bin.sum(dim=1, keepdim=True) == 0).to(torch.int8)  # (B,1,H,W)

    m = torch.cat([bg_bin, fg_bin], dim=1)  # (B,1+C,H,W)
    return torch.argmax(m, dim=1)  # (B,H,W) in {0..C}


def mean_dice_foreground(pred: torch.Tensor, gt: torch.Tensor, num_classes_with_bg: int) -> torch.Tensor:
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


def overlay_mask_on_image(img_raw: np.ndarray, mask: np.ndarray, num_classes: int, alpha: float = 0.5) -> np.ndarray:
    """
    将 mask 按透明度 alpha 叠加到原图上。
    自动适配单通道灰度图 (H, W) 和 三通道 RGB 图 (H, W, 3)。
    """
    img_min, img_max = img_raw.min(), img_raw.max()
    if img_max > img_min:
        img_norm = (img_raw - img_min) / (img_max - img_min) * 255.0
    else:
        img_norm = np.clip(img_raw * 255.0, 0, 255)
        
    if img_raw.ndim == 2:
        img_base = np.stack((img_norm,) * 3, axis=-1).astype(np.uint8)
    elif img_raw.ndim == 3 and img_raw.shape[-1] == 3:
        img_base = img_norm.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape for overlay: {img_raw.shape}")

    colored_mask = colorize_multiclass_with_bg(mask, num_classes)

    out = img_base.copy()
    fg_mask = mask > 0
    
    out[fg_mask] = (
        img_base[fg_mask] * (1.0 - alpha) + colored_mask[fg_mask] * alpha
    ).astype(np.uint8)

    return out


def multiclass_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    ignore_bg: bool = True,
) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must be (B,K,H,W), got {tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"target must be (B,H,W), got {tuple(target.shape)}")

    b, k, h, w = logits.shape
    prob = torch.softmax(logits, dim=1)  # (B,K,H,W)
    target_1h = F.one_hot(target.long(), num_classes=k).permute(0, 3, 1, 2).float()  # (B,K,H,W)

    dims = (0, 2, 3)
    inter = (prob * target_1h).sum(dim=dims)               # (K,)
    denom = prob.sum(dim=dims) + target_1h.sum(dim=dims)   # (K,)
    dice = (2.0 * inter + eps) / (denom + eps)             # (K,)

    if ignore_bg and k > 1:
        dice = dice[1:]

    return 1.0 - dice.mean()


class EMA:
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
        num_classes: int = 3,
        lr: float = 1e-4,
        sample_steps: int = 10,
        vis_every: int = 5,
        vis_batches: int = 1,
        output_root: str = "./Outputs/MedFlowSeg",
        thresh: float = 0.5,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        seg_prob: float = 0.1,
        seg_ce_beta: float = 10.0,
        rgb_vis: bool = False,      # ✅ 引入 rgb_vis 控制
        vis_alpha: float = 0.8,     # ✅ 引入不透明度控制
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(xt_in_ch=num_classes, img_in_ch=img_in_ch)

        self.num_classes_fg = int(num_classes)
        self.num_classes_with_bg = self.num_classes_fg + 1

        self.lr = float(lr)
        self.sample_steps = int(sample_steps)

        self.vis_every = int(vis_every)
        self.vis_batches = int(max(1, vis_batches))

        self.output_root = Path(output_root)
        self.vis_root = self.output_root / "vis"
        self.vis_root.mkdir(parents=True, exist_ok=True)

        self.thresh = float(thresh)

        self.seg_prob = float(seg_prob)
        self.seg_ce_beta = float(seg_ce_beta)
        if not (0.0 <= self.seg_prob <= 1.0):
            raise ValueError("seg_prob must be in [0,1].")
            
        self.rgb_vis = rgb_vis
        self.vis_alpha = float(vis_alpha)
        if not (0.0 <= self.vis_alpha <= 1.0):
            raise ValueError("vis_alpha must be between 0.0 and 1.0")

        self._val_vis_cache = []

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema: EMA | None = None

        self._ema_shadow_from_ckpt = None
        self._ema_decay_from_ckpt = None

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

    @torch.no_grad()
    def sample(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        x = torch.randn(B, self.num_classes_fg, H, W, device=image.device)

        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((B,), i * dt, device=image.device)
            v, _ = self.model(x, image, t)
            x = x + v * dt

        return x

    def training_step(self, batch, batch_idx):
        image, x1 = batch
        x1 = x1.float()

        B = image.shape[0]
        device = image.device

        t = torch.rand(B, device=device)
        t_ = t[:, None, None, None]

        x0 = torch.randn_like(x1)
        x_t = (1.0 - t_) * x0 + t_ * x1

        v_star = x1 - x0
        v_pred, seg_logits = self.model(x_t, image, t)

        loss_fm = F.l1_loss(v_pred, v_star)

        target = onehot_to_mask_argmax_gt_no_bg(x1).long()

        bg_logit = torch.zeros(
            B, 1, seg_logits.shape[-2], seg_logits.shape[-1],
            device=seg_logits.device,
            dtype=seg_logits.dtype,
        )
        seg_logits_with_bg = torch.cat([bg_logit, seg_logits], dim=1)

        loss_seg_ce = F.cross_entropy(seg_logits_with_bg, target)
        loss_seg_dice = multiclass_dice_loss_from_logits(
            seg_logits_with_bg, target, ignore_bg=True
        )
        
        raw_loss_seg = loss_seg_dice + self.seg_ce_beta * loss_seg_ce
        
        apply_seg_weight = 0.1 if torch.rand(1).item() < self.seg_prob else 0.0
        loss_seg = apply_seg_weight * raw_loss_seg
        
        loss = loss_fm + loss_seg

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_fm", loss_fm, prog_bar=False)
        self.log("train/loss_seg_effective", loss_seg, prog_bar=False)
        self.log("train/loss_seg_raw", raw_loss_seg, prog_bar=False)
        self.log("train/loss_seg_ce", loss_seg_ce, prog_bar=False)
        self.log("train/loss_seg_dice", loss_seg_dice, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, onehot_gt = batch
        onehot_gt = onehot_gt.float()

        pred = self.sample(image)

        pred_mask = onehot_to_mask_argmax_pred_no_bg(
            pred, thresh=self.thresh
        )
        gt_mask = onehot_to_mask_argmax_gt_no_bg(onehot_gt)

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

    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        if (epoch + 1) % self.vis_every != 0:
            self._val_vis_cache.clear()
        else:
            if len(self._val_vis_cache) != 0:
                save_dir = self.vis_root / f"epoch_{epoch:04d}"
                save_dir.mkdir(parents=True, exist_ok=True)

                for cache in self._val_vis_cache:
                    img_tensor = cache["image"]  # (B, C, H, W)
                    pred = cache["pred"].numpy()
                    gt = cache["gt"].numpy()
                    bidx = cache.get("batch_idx", 0)
                    
                    # ✅ 同步测试脚本逻辑：显式控制 rgb_vis
                    is_rgb = self.rgb_vis and (img_tensor.shape[1] >= 3)
                    
                    if is_rgb:
                        img = img_tensor[:, :3].permute(0, 2, 3, 1).numpy()
                    else:
                        img = img_tensor[:, 0].numpy()

                    for i in range(min(4, img.shape[0])):
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                        img_i = img[i]
                        gt_i = gt[i]
                        pred_i = pred[i]

                        # ✅ 传递 num_classes_fg（前景类数量）且使用配置的透明度 vis_alpha
                        overlay_gt = overlay_mask_on_image(img_i, gt_i, self.num_classes_fg, alpha=self.vis_alpha)
                        overlay_pred = overlay_mask_on_image(img_i, pred_i, self.num_classes_fg, alpha=self.vis_alpha)

                        # 原图显示处理
                        if is_rgb:
                            img_disp = img_i.copy()
                            img_min, img_max = img_disp.min(), img_disp.max()
                            if img_max > img_min:
                                img_disp = (img_disp - img_min) / (img_max - img_min)
                            else:
                                img_disp = np.clip(img_disp, 0, 1)
                            axs[0].imshow(img_disp)
                        else:
                            axs[0].imshow(img_i, cmap="gray")
                            
                        axs[0].set_title("Raw Image" + (" (RGB)" if is_rgb else " (Gray)"))
                        axs[0].axis("off")

                        axs[1].imshow(overlay_gt)
                        axs[1].set_title("Image + GT Mask")
                        axs[1].axis("off")

                        axs[2].imshow(overlay_pred)
                        axs[2].set_title("Image + Pred Mask")
                        axs[2].axis("off")

                        plt.tight_layout()
                        plt.savefig(save_dir / f"batch{bidx:03d}_sample{i}.png", dpi=200) # 统一 dpi
                        plt.close()

            self._val_vis_cache.clear()

        if self.use_ema and self.ema is not None:
            self.ema.restore(self.model)

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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=144)

    # -------- Output --------
    parser.add_argument("--output_root", type=str, default="./Outputs/MedFlowSeg-ACDC")
    parser.add_argument("--vis_every", type=int, default=10)
    parser.add_argument("--vis_batches", type=int, default=2)

    # ✅ 新增不透明度和RGB控制参数，对齐测试代码
    parser.add_argument("--rgb_vis", action="store_true", help="enable RGB visualization")
    parser.add_argument("--vis_alpha", type=float, default=0.8, help="Opacity for segmentation mask overlay (0.0 to 1.0)")

    # onehot 解码阈值
    parser.add_argument("--thresh", type=float, default=0.5)

    # seg-head loss
    parser.add_argument("--seg_prob", type=float, default=1.0)
    parser.add_argument("--seg_ce_beta", type=float, default=10.0)

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
    parser.add_argument("--gpus", type=int, default=1)

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
        use_ema=(not args.no_ema),
        ema_decay=args.ema_decay,
        seg_prob=args.seg_prob,
        seg_ce_beta=args.seg_ce_beta,
        rgb_vis=args.rgb_vis,       
        vis_alpha=args.vis_alpha,   
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