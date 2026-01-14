# ======================================================
# train_medflowseg.py
# Flow-Matching Medical Segmentation (Lightning v2)
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
def sdf_to_mask(sdf: torch.Tensor) -> torch.Tensor:
    return torch.argmin(sdf, dim=1)


def mean_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    dices = []
    for c in range(1, num_classes):  # exclude background
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + 1e-6
        dices.append((2 * inter / denom).mean())
    return torch.stack(dices).mean()


def colorize(mask: np.ndarray, num_classes: int) -> np.ndarray:
    colors = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
        ],
        dtype=np.uint8,
    )
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(num_classes):
        out[mask == c] = colors[c % len(colors)]
    return out


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
        output_root: str = "./Outputs/MedFlowSeg",
    ):
        super().__init__()
        self.save_hyperparameters()

        # Flow-matching model (expects x, image, t)
        self.model = build_model(xt_in_ch=num_classes)

        self.num_classes = num_classes
        self.lr = lr
        self.sample_steps = sample_steps

        self.vis_every = vis_every
        self.output_root = Path(output_root)
        self.vis_root = self.output_root / "vis"
        self.vis_root.mkdir(parents=True, exist_ok=True)

        self._val_vis_cache = []

    # --------------------------------------------------
    # Flow Matching sampling (Euler)
    # --------------------------------------------------
    @torch.no_grad()
    def sample(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        x = torch.randn(B, self.num_classes, H, W, device=image.device)

        for i in range(self.sample_steps):
            t = torch.full((B,), i / self.sample_steps, device=image.device)
            v = self.model(x, image, t)
            x = x + v / self.sample_steps

        return x

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def training_step(self, batch, batch_idx):
        image, sdf = batch

        B = image.shape[0]
        t = torch.rand(B, device=image.device)

        v_pred = self.model(sdf, image, t)
        loss = nn.functional.mse_loss(v_pred, sdf)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        image, sdf_gt = batch

        sdf_pred = self.sample(image)
        pred_mask = sdf_to_mask(sdf_pred)
        gt_mask = sdf_to_mask(sdf_gt)

        dice = mean_dice(pred_mask, gt_mask, self.num_classes)

        self.log(
            "val/mean_dice",
            dice,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if batch_idx == 0:
            self._val_vis_cache.append(
                {
                    "image": image.detach().cpu(),
                    "pred": pred_mask.detach().cpu(),
                    "gt": gt_mask.detach().cpu(),
                }
            )

    # --------------------------------------------------
    # End of validation epoch
    # --------------------------------------------------
    def on_validation_epoch_end(self):
        epoch = self.current_epoch

        if (epoch + 1) % self.vis_every != 0:
            self._val_vis_cache.clear()
            return

        if len(self._val_vis_cache) == 0:
            return

        save_dir = self.vis_root / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        cache = self._val_vis_cache[0]
        img = cache["image"][:, 0].numpy()
        pred = cache["pred"].numpy()
        gt = cache["gt"].numpy()

        for i in range(min(4, img.shape[0])):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].imshow(img[i], cmap="gray")
            axs[0].set_title("Image")
            axs[0].axis("off")

            axs[1].imshow(colorize(gt[i], self.num_classes))
            axs[1].set_title("GT")
            axs[1].axis("off")

            axs[2].imshow(colorize(pred[i], self.num_classes))
            axs[2].set_title("Pred")
            axs[2].axis("off")

            plt.tight_layout()
            plt.savefig(save_dir / f"sample_{i}.png")
            plt.close()

        self._val_vis_cache.clear()

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
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=10)

    # -------- Optimization --------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)

    # -------- Output --------
    parser.add_argument("--output_root", type=str, default="./Outputs/MedFlowSeg")
    parser.add_argument("--vis_every", type=int, default=5)

    # -------- System --------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    output_root = Path(args.output_root)
    ckpt_root = output_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # -------- Logger --------
    logger = TensorBoardLogger(
        save_dir=output_root,
        name="logs",
    )

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
        output_root=args.output_root,
    )

    # -------- Checkpoint --------
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_root,
        filename="epoch{epoch:03d}-dice{val/mean_dice:.4f}",
        monitor="val/mean_dice",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    # -------- Trainer --------
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        max_epochs=args.epochs,
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=20,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()