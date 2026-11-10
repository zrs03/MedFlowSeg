#!/usr/bin/env python3
"""MedFlowSeg folder inference demo.

Read images from a demo folder, run stochastic flow sampling + OVR-STAPLE
fusion, and save segmentation masks and visualizations.

Example:
    python demo.py \
        --ckpt checkpoints/best.ckpt \
        --demo_dir ./demo \
        --out_dir ./demo_results \
        --num_classes 3 \
        --img_in_ch 1 \
        --sample_steps 50 \
        --staple_k 10
"""

import argparse
import colorsys
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.MedFlowSeg import build_model


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


# ======================================================
# Decode + visualization
# ======================================================
@torch.no_grad()
def onehot_to_mask_pred_with_bg(
    pred: torch.Tensor,
    thresh: float = 0.5,
) -> torch.Tensor:
    """Convert foreground-channel predictions (B,C,H,W) to labels 0..C."""
    if pred.dim() != 4:
        raise ValueError(f"pred must be (B,C,H,W), got {tuple(pred.shape)}")

    fg_score = pred.clamp(0.0, 1.0)
    max_fg = fg_score.max(dim=1, keepdim=True).values
    bg_score = (max_fg < thresh).float()

    scores = torch.cat([bg_score, fg_score], dim=1)
    return torch.argmax(scores, dim=1)


def make_palette(num_classes: int) -> np.ndarray:
    """Return RGB colors for labels 0..num_classes; label 0 is black."""
    palette = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    for class_id in range(1, num_classes + 1):
        hue = class_id / (num_classes + 1)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        palette[class_id] = np.round(np.asarray(rgb) * 255).astype(np.uint8)
    return palette


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask.shape}")
    if mask.min() < 0 or mask.max() >= len(palette):
        raise ValueError("Mask contains class IDs outside the palette range")
    return palette[mask.astype(np.int64)]


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Convert HxW or HxWx3 image to uint8 RGB for visualization."""
    arr = image.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        valid = arr[finite]
        low, high = float(valid.min()), float(valid.max())
        if high > low:
            arr = (arr - low) / (high - low)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr[~finite] = 0.0

    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unsupported display image shape: {arr.shape}")
    return arr


def overlay_mask(
    image_rgb: np.ndarray,
    color_mask: np.ndarray,
    label_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = image_rgb.copy()
    foreground = label_mask > 0
    out[foreground] = (
        image_rgb[foreground].astype(np.float32) * (1.0 - alpha)
        + color_mask[foreground].astype(np.float32) * alpha
    ).astype(np.uint8)
    return out


def save_label_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_value = int(mask.max()) if mask.size else 0
    if max_value <= 255:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(path)
    elif max_value <= 65535:
        Image.fromarray(mask.astype(np.uint16), mode="I;16").save(path)
    else:
        raise ValueError("More than 65535 class IDs cannot be saved as a PNG mask")


def save_comparison(
    raw_rgb: np.ndarray,
    direct_overlay: np.ndarray,
    flow_overlay: np.ndarray,
    path: Path,
) -> None:
    """Save a horizontal raw/direct/flow comparison without matplotlib."""
    h, w, _ = raw_rgb.shape
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(Image.fromarray(raw_rgb), (0, 0))
    canvas.paste(Image.fromarray(direct_overlay), (w, 0))
    canvas.paste(Image.fromarray(flow_overlay), (w * 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# ======================================================
# STAPLE (binary) + one-vs-rest multi-class fusion
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
        raise ValueError(f"masks must be (K,H,W), got {tuple(masks.shape)}")

    k_count, height, width = masks.shape
    m = masks.reshape(k_count, -1).float()
    num_pixels = m.shape[1]

    p = m.mean(dim=0).clamp(eps, 1.0 - eps)
    sensitivity = torch.full((k_count,), init_s, device=m.device)
    specificity = torch.full((k_count,), init_t, device=m.device)

    for _ in range(iters):
        log_p1 = torch.zeros(num_pixels, device=m.device)
        log_p0 = torch.zeros(num_pixels, device=m.device)

        for i in range(k_count):
            mi = m[i]
            si = sensitivity[i].clamp(eps, 1.0 - eps)
            ti = specificity[i].clamp(eps, 1.0 - eps)
            log_p1 += mi * torch.log(si) + (1.0 - mi) * torch.log(1.0 - si)
            log_p0 += mi * torch.log(1.0 - ti) + (1.0 - mi) * torch.log(ti)

        a = torch.log(p.clamp(eps, 1.0 - eps)) + log_p1
        b = torch.log((1.0 - p).clamp(eps, 1.0 - eps)) + log_p0
        p_new = torch.sigmoid(a - b).clamp(eps, 1.0 - eps)

        denom1 = p_new.sum().clamp_min(eps)
        denom0 = (1.0 - p_new).sum().clamp_min(eps)

        for i in range(k_count):
            mi = m[i]
            sensitivity[i] = ((p_new * mi).sum() / denom1).clamp(eps, 1.0 - eps)
            specificity[i] = (
                ((1.0 - p_new) * (1.0 - mi)).sum() / denom0
            ).clamp(eps, 1.0 - eps)

        p = p_new

    return p.reshape(height, width)


@torch.no_grad()
def ovr_staple_fuse_fg_with_bg(
    masks_k: torch.Tensor,
    num_classes: int,
    staple_iters: int = 15,
    bg_thresh: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse K label masks (K,B,H,W), returning labels and FG probabilities."""
    if masks_k.dim() != 4:
        raise ValueError(f"masks_k must be (K,B,H,W), got {tuple(masks_k.shape)}")

    _, batch_size, height, width = masks_k.shape
    probs_fg = torch.zeros(
        (batch_size, num_classes, height, width),
        device=masks_k.device,
        dtype=torch.float32,
    )

    for class_id in range(1, num_classes + 1):
        binary_k = (masks_k == class_id).float()
        for batch_id in range(batch_size):
            probs_fg[batch_id, class_id - 1] = staple_binary(
                binary_k[:, batch_id], iters=staple_iters
            )

    max_prob, argmax_fg = probs_fg.max(dim=1)
    final_mask = (argmax_fg + 1).long()
    final_mask[max_prob < bg_thresh] = 0
    return final_mask, probs_fg


# ======================================================
# Model wrapper + checkpoint loading
# ======================================================
class MedFlowSegTester(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        sample_steps: int = 50,
        freq_attn_depth: int = 0,
        img_in_ch: int = 1,
    ) -> None:
        super().__init__()
        if sample_steps <= 0:
            raise ValueError("sample_steps must be > 0")

        self.num_classes = num_classes
        self.sample_steps = sample_steps
        self.net = build_model(
            xt_in_ch=num_classes,
            freq_attn_depth=freq_attn_depth,
            img_in_ch=img_in_ch,
        )

    @torch.no_grad()
    def sample(
        self,
        image: torch.Tensor,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = image.shape

        if seed is not None:
            generator = torch.Generator(device=image.device)
            generator.manual_seed(int(seed))
            x = torch.randn(
                batch_size,
                self.num_classes,
                height,
                width,
                device=image.device,
                generator=generator,
            )
        else:
            x = torch.randn(
                batch_size,
                self.num_classes,
                height,
                width,
                device=image.device,
            )

        dt = 1.0 / self.sample_steps
        direct_logits: Optional[torch.Tensor] = None

        for step in range(self.sample_steps):
            t = torch.full(
                (batch_size,),
                step * dt,
                device=image.device,
                dtype=image.dtype,
            )
            velocity, seg_logits = self.net(x, image, t)
            if step == 0:
                direct_logits = seg_logits
            x = x + velocity * dt

        if direct_logits is None:
            raise RuntimeError("direct_logits was not produced")
        return x, torch.sigmoid(direct_logits)


def strip_checkpoint_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    prefixes = ("model.", "net.", "module.")

    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def load_lightning_ckpt_into_model(
    model: MedFlowSegTester,
    ckpt_path: str,
    use_ema: bool = True,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if use_ema and isinstance(checkpoint.get("ema_shadow"), dict):
        raw_state = checkpoint["ema_shadow"]
        source = "EMA weights (ema_shadow)"
    else:
        raw_state = checkpoint.get("state_dict", checkpoint)
        source = "normal weights"
        if use_ema:
            print("[WARN] EMA requested, but ema_shadow was not found.")

    if not isinstance(raw_state, dict):
        raise TypeError("Checkpoint does not contain a valid state dictionary")

    state = strip_checkpoint_prefixes(raw_state)
    incompatible = model.net.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded {source} from: {ckpt_path}")

    if incompatible.missing_keys:
        print(f"[WARN] Missing keys: {len(incompatible.missing_keys)}")
        print("       ", incompatible.missing_keys[:10])
    if incompatible.unexpected_keys:
        print(f"[WARN] Unexpected keys: {len(incompatible.unexpected_keys)}")
        print("       ", incompatible.unexpected_keys[:10])


# ======================================================
# Image loading / preprocessing
# ======================================================
def list_images(folder: Path, recursive: bool) -> List[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path, img_in_ch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return model image array and an RGB/grayscale array for visualization."""
    with Image.open(path) as image:
        if img_in_ch == 1:
            image = image.convert("F")
            model_array = np.asarray(image, dtype=np.float32)
            display_array = model_array.copy()
        elif img_in_ch == 3:
            image = image.convert("RGB")
            model_array = np.asarray(image, dtype=np.float32)
            display_array = model_array.copy()
        else:
            raise ValueError("This demo supports --img_in_ch 1 or 3")

    return model_array, display_array


def normalize_input(array: np.ndarray, mode: str) -> np.ndarray:
    arr = array.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    arr[~finite] = 0.0
    valid = arr[finite]

    if mode == "minmax":
        low, high = float(valid.min()), float(valid.max())
        if high > low:
            arr = (arr - low) / (high - low)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    elif mode == "zero_one":
        # Suitable for ordinary 8-bit/16-bit image files.
        high = float(valid.max())
        if high > 1.0:
            divisor = 255.0 if high <= 255.0 else 65535.0
            arr = arr / divisor
        arr = np.clip(arr, 0.0, 1.0)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return arr.astype(np.float32)


def resize_model_array(
    array: np.ndarray,
    input_size: Optional[Tuple[int, int]],
) -> np.ndarray:
    if input_size is None:
        return array

    target_h, target_w = input_size
    if array.ndim == 2:
        pil = Image.fromarray(array.astype(np.float32), mode="F")
        resized = pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)

    if array.ndim == 3 and array.shape[-1] == 3:
        # PIL float RGB is not broadly supported, so resize channel by channel.
        channels = []
        for channel in range(3):
            pil = Image.fromarray(array[..., channel].astype(np.float32), mode="F")
            resized = pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
            channels.append(np.asarray(resized, dtype=np.float32))
        return np.stack(channels, axis=-1)

    raise ValueError(f"Unsupported model array shape: {array.shape}")


def array_to_tensor(array: np.ndarray, img_in_ch: int) -> torch.Tensor:
    if img_in_ch == 1:
        if array.ndim != 2:
            raise ValueError(f"Expected grayscale HxW array, got {array.shape}")
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    else:
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Expected RGB HxWx3 array, got {array.shape}")
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.float().contiguous()


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    original_h, original_w = tensor.shape[-2:]
    if multiple <= 1:
        return tensor, (original_h, original_w)

    pad_h = (multiple - original_h % multiple) % multiple
    pad_w = (multiple - original_w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (original_h, original_w)

    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (original_h, original_w)


def resize_mask_to_original(
    mask: np.ndarray,
    original_hw: Tuple[int, int],
) -> np.ndarray:
    original_h, original_w = original_hw
    if mask.shape == (original_h, original_w):
        return mask

    pil = Image.fromarray(mask.astype(np.int32), mode="I")
    resized = pil.resize((original_w, original_h), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def output_relative_path(image_path: Path, demo_dir: Path, recursive: bool) -> Path:
    if recursive:
        return image_path.relative_to(demo_dir).with_suffix(".png")
    return Path(image_path.stem + ".png")


# ======================================================
# Inference
# ======================================================
@torch.no_grad()
def run_demo(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    demo_dir = Path(args.demo_dir)
    out_dir = Path(args.out_dir)
    if not demo_dir.is_dir():
        raise FileNotFoundError(f"Demo folder does not exist: {demo_dir}")

    image_paths = list_images(demo_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in {demo_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    output_dirs = {
        "flow_mask": out_dir / "flow_mask",
        "flow_color": out_dir / "flow_color",
        "flow_overlay": out_dir / "flow_overlay",
        "direct_mask": out_dir / "direct_mask",
        "direct_color": out_dir / "direct_color",
        "direct_overlay": out_dir / "direct_overlay",
        "comparison": out_dir / "comparison",
        "flow_prob": out_dir / "flow_prob",
    }
    for folder in output_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    model = MedFlowSegTester(
        num_classes=args.num_classes,
        sample_steps=args.sample_steps,
        freq_attn_depth=args.freq_attn_depth,
        img_in_ch=args.img_in_ch,
    ).to(device)
    load_lightning_ckpt_into_model(model, args.ckpt, use_ema=not args.no_ema)
    model.eval()

    palette = make_palette(args.num_classes)
    records: List[Dict[str, str]] = []

    for index, image_path in enumerate(image_paths):
        model_array, display_array = read_image(image_path, args.img_in_ch)
        original_hw = model_array.shape[:2]

        normalized = normalize_input(model_array, args.normalize)
        normalized = resize_model_array(normalized, args.input_size)
        image_tensor = array_to_tensor(normalized, args.img_in_ch)
        image_tensor, inference_hw = pad_to_multiple(image_tensor, args.pad_multiple)
        image_tensor = image_tensor.to(device, non_blocking=True)

        sampled_masks: List[torch.Tensor] = []
        direct_probability: Optional[torch.Tensor] = None

        for sample_id in range(args.staple_k):
            seed = args.base_seed + index * 1000 + sample_id
            flow_prediction, direct_prob = model.sample(image_tensor, seed=seed)
            if direct_probability is None:
                direct_probability = direct_prob
            sampled_masks.append(
                onehot_to_mask_pred_with_bg(flow_prediction, thresh=args.thresh)
            )

        if direct_probability is None:
            raise RuntimeError("Direct prediction was not produced")

        masks_k = torch.stack(sampled_masks, dim=0)
        flow_mask_t, flow_probs_t = ovr_staple_fuse_fg_with_bg(
            masks_k=masks_k,
            num_classes=args.num_classes,
            staple_iters=args.staple_iters,
            bg_thresh=args.bg_thresh,
        )
        direct_mask_t = onehot_to_mask_pred_with_bg(
            direct_probability, thresh=args.thresh
        )

        valid_h, valid_w = inference_hw
        flow_mask = flow_mask_t[0, :valid_h, :valid_w].cpu().numpy().astype(np.int64)
        direct_mask = direct_mask_t[0, :valid_h, :valid_w].cpu().numpy().astype(np.int64)
        flow_probs = flow_probs_t[0, :, :valid_h, :valid_w].cpu().numpy()

        flow_mask = resize_mask_to_original(flow_mask, original_hw)
        direct_mask = resize_mask_to_original(direct_mask, original_hw)

        # Probability maps are saved at inference resolution to avoid interpolating
        # calibrated STAPLE probabilities. Their shape is (C,H,W).
        relative_png = output_relative_path(image_path, demo_dir, args.recursive)
        relative_npy = relative_png.with_suffix(".npy")

        flow_color = colorize_mask(flow_mask, palette)
        direct_color = colorize_mask(direct_mask, palette)
        raw_rgb = normalize_for_display(display_array)
        flow_overlay = overlay_mask(raw_rgb, flow_color, flow_mask, args.vis_alpha)
        direct_overlay = overlay_mask(raw_rgb, direct_color, direct_mask, args.vis_alpha)

        flow_mask_path = output_dirs["flow_mask"] / relative_png
        direct_mask_path = output_dirs["direct_mask"] / relative_png
        flow_color_path = output_dirs["flow_color"] / relative_png
        direct_color_path = output_dirs["direct_color"] / relative_png
        flow_overlay_path = output_dirs["flow_overlay"] / relative_png
        direct_overlay_path = output_dirs["direct_overlay"] / relative_png
        comparison_path = output_dirs["comparison"] / relative_png
        flow_prob_path = output_dirs["flow_prob"] / relative_npy

        save_label_mask(flow_mask, flow_mask_path)
        save_label_mask(direct_mask, direct_mask_path)

        for path, array in (
            (flow_color_path, flow_color),
            (direct_color_path, direct_color),
            (flow_overlay_path, flow_overlay),
            (direct_overlay_path, direct_overlay),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)

        save_comparison(raw_rgb, direct_overlay, flow_overlay, comparison_path)

        if args.save_prob:
            flow_prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(flow_prob_path, flow_probs.astype(np.float32))

        records.append({
            "input": str(image_path),
            "flow_mask": str(flow_mask_path),
            "flow_overlay": str(flow_overlay_path),
            "direct_mask": str(direct_mask_path),
            "direct_overlay": str(direct_overlay_path),
        })

        print(f"[{index + 1:04d}/{len(image_paths):04d}] {image_path.name}")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\n===== DEMO DONE =====")
    print(f"Processed images: {len(image_paths)}")
    print(f"Results saved to: {out_dir.resolve()}")
    print(f"Index CSV: {csv_path.resolve()}")


def parse_input_size(values: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--input_size requires exactly two integers: H W")
    height, width = int(values[0]), int(values[1])
    if height <= 0 or width <= 0:
        raise ValueError("--input_size values must be positive")
    return height, width


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "MedFlowSeg demo folder inference (Direct + Flow/STAPLE)"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning .ckpt path")
    parser.add_argument("--demo_dir", type=str, default="./demo")
    parser.add_argument("--out_dir", type=str, default="./demo_results")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders")

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--img_in_ch", type=int, choices=(1, 3), default=1)
    parser.add_argument("--freq_attn_depth", type=int, default=0)

    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--staple_k", type=int, default=10)
    parser.add_argument("--staple_iters", type=int, default=15)
    parser.add_argument("--bg_thresh", type=float, default=0.5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true")

    parser.add_argument(
        "--normalize",
        choices=("minmax", "zero_one", "none"),
        default="minmax",
        help="Must match the preprocessing used during training",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional model input size; default keeps the original size",
    )
    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=16,
        help="Pad H/W to this multiple before inference; use 1 to disable",
    )
    parser.add_argument("--vis_alpha", type=float, default=0.8)
    parser.add_argument(
        "--save_prob",
        action="store_true",
        help="Save OVR-STAPLE foreground probabilities as .npy",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.input_size = parse_input_size(args.input_size)

    if args.num_classes <= 0:
        parser.error("--num_classes must be > 0")
    if args.sample_steps <= 0:
        parser.error("--sample_steps must be > 0")
    if args.staple_k < 2:
        parser.error("--staple_k must be >= 2; >= 3 is recommended")
    if args.staple_iters <= 0:
        parser.error("--staple_iters must be > 0")
    if not 0.0 <= args.thresh <= 1.0:
        parser.error("--thresh must be between 0 and 1")
    if not 0.0 <= args.bg_thresh <= 1.0:
        parser.error("--bg_thresh must be between 0 and 1")
    if not 0.0 <= args.vis_alpha <= 1.0:
        parser.error("--vis_alpha must be between 0 and 1")
    if args.pad_multiple <= 0:
        parser.error("--pad_multiple must be >= 1")

    seed_everything(args.base_seed)
    run_demo(args)


if __name__ == "__main__":
    main()