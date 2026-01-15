# -*- coding: utf-8 -*-
"""
Flow-Matching Segmentation Network (ResUNet backbone) with:
- Fore/Back separation (x_t: ch0=background, ch1..=foreground)
- Fore/Back conditional encoders (independent)
- Frequency-domain Cross Attention on deepest K encoder skip levels (optional)
- ✅ NEW (3): Bottleneck coupling (Fore<->Back) with FreqCrossAttention at the deepest latent
- ✅ NEW (4): Decoder Attention-Gated Skip connections
- ✅ NEW (4): Optional Deep Supervision / Multi-Scale heads (kept backward-compatible by default)

Default behavior (backward-compatible):
- forward(...) returns only v of shape (B, xt_in_ch, H, W)

If cfg.deep_supervision=True and forward(..., return_aux=True):
- returns (v, aux) where aux is a list of upsampled multi-scale predictions (all at HxW)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Time embedding + FiLM
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        if t.dim() == 2 and t.shape[-1] == 1:
            t = t[:, 0]
        t = t.float()

        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class GroupNorm32(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32):
        super().__init__()
        g = min(num_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class ResBlockFiLM(nn.Module):
    """ResBlock with FiLM time conditioning: (1+gamma)*h + beta."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_ch: int,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.norm1 = GroupNorm32(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = GroupNorm32(out_ch)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(time_ch, 2 * out_ch))

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.shortcut = (
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
                if use_conv_shortcut
                else nn.Conv2d(in_ch, out_ch, 1)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        h2 = self.norm2(h)
        gamma, beta = self.film(t_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h2 = (1.0 + gamma) * h2 + beta

        h2 = self.conv2(self.dropout(F.silu(h2)))
        return h2 + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


# -----------------------------
# Spatial Cross Attention (real)
# -----------------------------
class SpatialCrossAttention(nn.Module):
    def __init__(self, q_ch: int, kv_ch: int, heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim

        self.to_q = nn.Conv2d(q_ch, inner, 1, bias=False)
        self.to_k = nn.Conv2d(kv_ch, inner, 1, bias=False)
        self.to_v = nn.Conv2d(kv_ch, inner, 1, bias=False)

        self.out = nn.Conv2d(inner, q_ch, 1)
        self.scale = head_dim**-0.5

        self.norm_q = GroupNorm32(q_ch)
        self.norm_kv = GroupNorm32(kv_ch)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = self.norm_q(x)
        c = self.norm_kv(c)

        q = self.to_q(x).view(b, self.heads, self.head_dim, h * w)
        k = self.to_k(c).view(b, self.heads, self.head_dim, h * w)
        v = self.to_v(c).view(b, self.heads, self.head_dim, h * w)

        attn = torch.einsum("bhdi,bhdj->bhij", q * self.scale, k).softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(b, self.heads * self.head_dim, h, w)
        return self.out(out)


# -----------------------------
# Frequency-domain Cross Attention (FFT -> XAttn -> IFFT)
# -----------------------------
class FreqCrossAttention(nn.Module):
    """Symmetric frequency-domain cross-attention between two (B,C,H,W) real features."""

    def __init__(self, ch: int, heads: int = 4, head_dim: int = 32, gate_init: float = 0.0):
        super().__init__()
        self.ch = ch
        self.attn = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        # ✅ stability: gate starts near 0 so early training doesn't blow up
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # lightweight norm on delta for stability
        self.delta_norm = GroupNorm32(2 * ch)

    @staticmethod
    def _to_ri(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def _to_complex(ri: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = ri.shape
        c = c2 // 2
        return torch.complex(ri[:, :c], ri[:, c:])

    def _update(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        X = torch.fft.fft2(x, dim=(-2, -1))
        C = torch.fft.fft2(c, dim=(-2, -1))
        Xri = self._to_ri(X)
        Cri = self._to_ri(C)
        delta = self.attn(Xri, Cri)
        delta = self.delta_norm(delta)
        Xri = Xri + self.gate * delta
        X2 = self._to_complex(Xri)
        return torch.fft.ifft2(X2, dim=(-2, -1)).real

    def forward(self, fore: torch.Tensor, back: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fore2 = self._update(fore, back)
        back2 = self._update(back, fore)
        return fore2, back2


# -----------------------------
# ✅ Decoder attention-gated skip
# -----------------------------
class AttentionGate(nn.Module):
    """
    Classic attention-gated skip for UNet:
    gate = sigmoid( conv( relu( Wg * x + Ws * skip ) ) )
    skip_out = skip * gate
    """

    def __init__(self, x_ch: int, skip_ch: int, inter_ch: Optional[int] = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(8, min(x_ch, skip_ch) // 2)

        self.Wg = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            GroupNorm32(inter_ch),
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            GroupNorm32(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x and skip are at same spatial resolution
        a = F.silu(self.Wg(x) + self.Ws(skip))
        gate = self.psi(a)  # (B,1,H,W)
        return skip * gate


# -----------------------------
# Encoder / Decoder stages
# -----------------------------
class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, num_blocks: int, down: bool, dropout: float):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_blocks):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.down = Downsample(out_ch) if down else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x, t_emb)
        skip = x
        if self.down is not None:
            x = self.down(x)
        return x, skip


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        time_ch: int,
        num_blocks: int,
        up: bool,
        dropout: float,
        use_attn_gate: bool = True,
    ):
        super().__init__()
        self.up = Upsample(in_ch) if up else None
        self.use_attn_gate = use_attn_gate
        self.gate = AttentionGate(x_ch=in_ch, skip_ch=skip_ch) if use_attn_gate else None

        blocks: List[nn.Module] = []
        ch = in_ch + skip_ch
        for _ in range(num_blocks):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)

        # ✅ attention-gated skip
        if self.use_attn_gate and self.gate is not None:
            skip = self.gate(x, skip)

        x = torch.cat([x, skip], dim=1)
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# -----------------------------
# Main Network
# -----------------------------
@dataclass
class Config:
    img_in_ch: int = 1
    xt_in_ch: int = 4  # 0=bg, 1..=fg
    base_ch: int = 64
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    time_dim: int = 256
    dropout: float = 0.0

    attn_heads: int = 4
    attn_head_dim: int = 32

    # deepest K encoder levels to apply FreqCrossAttention on skips
    freq_attn_depth: int = 1

    # ✅ NEW (3): also couple at bottleneck
    use_bottleneck_coupling: bool = True
    bottleneck_gate_init: float = 0.0  # start stable

    # ✅ NEW (4): attention-gated skips
    use_attn_gated_skip: bool = True

    # ✅ NEW (4): deep supervision multi-scale heads
    deep_supervision: bool = False


class FMForeBackDeepKFreqXAttnDualDecoderResUNet(nn.Module):
    """
    - Two streams (fore/back): independent encoders + independent decoders
    - Deepest K encoder levels apply symmetric frequency-domain cross-attention on skips
    - ✅ Bottleneck coupling: symmetric FreqCrossAttention at deepest latent
    - ✅ Decoder: attention-gated skip connections
    - ✅ Optional deep supervision multi-scale heads
    """

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.xt_in_ch < 2:
            raise ValueError("xt_in_ch must be >=2 (channel0 background, remaining foreground).")
        if cfg.freq_attn_depth < 1:
            raise ValueError("freq_attn_depth must be >= 1.")
        self.cfg = cfg

        self.time_mlp = TimeMLP(time_dim=cfg.time_dim, hidden_dim=cfg.time_dim)

        back_in_ch = cfg.img_in_ch + 1
        fore_in_ch = cfg.img_in_ch + (cfg.xt_in_ch - 1)

        # stems
        self.back_stem = nn.Conv2d(back_in_ch, cfg.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(fore_in_ch, cfg.base_ch, 3, padding=1)

        # channels per level (shallow->deep)
        self.chs = [cfg.base_ch * m for m in cfg.ch_mult]
        n_levels = len(self.chs)

        # encoders (independent)
        self.back_enc = nn.ModuleList()
        self.fore_enc = nn.ModuleList()
        for i in range(n_levels):
            in_ch = self.chs[i - 1] if i > 0 else cfg.base_ch
            out_ch = self.chs[i]
            down = i < (n_levels - 1)
            self.back_enc.append(EncoderStage(in_ch, out_ch, cfg.time_dim, cfg.num_res_blocks, down, cfg.dropout))
            self.fore_enc.append(EncoderStage(in_ch, out_ch, cfg.time_dim, cfg.num_res_blocks, down, cfg.dropout))

        # deepest K levels for skip attention
        K = min(cfg.freq_attn_depth, n_levels)
        self.attn_level_set = set(range(n_levels - K, n_levels))
        self.freq_attn_modules = nn.ModuleDict()
        for i in sorted(self.attn_level_set):
            self.freq_attn_modules[str(i)] = FreqCrossAttention(
                self.chs[i], cfg.attn_heads, cfg.attn_head_dim, gate_init=cfg.bottleneck_gate_init
            )

        # ✅ NEW (3): bottleneck coupling module
        self.bottleneck_coupler = (
            FreqCrossAttention(self.chs[-1], cfg.attn_heads, cfg.attn_head_dim, gate_init=cfg.bottleneck_gate_init)
            if cfg.use_bottleneck_coupling
            else None
        )

        # decoders (independent)
        self.back_dec = nn.ModuleList()
        self.fore_dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.chs[i]
            in_ch = self.chs[i + 1] if i < (n_levels - 1) else self.chs[i]
            out_ch = self.chs[i]
            up = i < (n_levels - 1)

            self.back_dec.append(
                DecoderStage(
                    in_ch,
                    skip_ch,
                    out_ch,
                    cfg.time_dim,
                    cfg.num_res_blocks,
                    up,
                    cfg.dropout,
                    use_attn_gate=cfg.use_attn_gated_skip,
                )
            )
            self.fore_dec.append(
                DecoderStage(
                    in_ch,
                    skip_ch,
                    out_ch,
                    cfg.time_dim,
                    cfg.num_res_blocks,
                    up,
                    cfg.dropout,
                    use_attn_gate=cfg.use_attn_gated_skip,
                )
            )

        # output head: concat decoded fore/back at base resolution
        self.out_norm = GroupNorm32(cfg.base_ch * 2)
        self.out_conv = nn.Conv2d(cfg.base_ch * 2, cfg.xt_in_ch, 3, padding=1)

        # ✅ NEW (4): multi-scale heads (one per decoder stage output)
        # We upsample each to full-res if deep supervision enabled.
        self.deep_supervision = cfg.deep_supervision
        if self.deep_supervision:
            self.ms_norms = nn.ModuleList()
            self.ms_convs = nn.ModuleList()
            # decoder outputs are at chs[i] for i=0..n-1, but we decode in reverse order.
            # At each stage output, we have fore_h/back_h with same channels (out_ch).
            for out_ch in [self.chs[i] for i in reversed(range(n_levels))]:
                self.ms_norms.append(GroupNorm32(out_ch * 2))
                self.ms_convs.append(nn.Conv2d(out_ch * 2, cfg.xt_in_ch, 3, padding=1))

    def forward(
        self,
        x_t: torch.Tensor,
        image: torch.Tensor,
        t: Union[torch.Tensor, float],
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if isinstance(t, float):
            t = torch.tensor(t, device=x_t.device)
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.expand(x_t.shape[0])
        elif isinstance(t, torch.Tensor) and t.dim() == 1 and t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])

        t_emb = self.time_mlp(t)

        # split x_t
        x_back = x_t[:, 0:1]
        x_fore = x_t[:, 1:]

        # concat with condition image
        back_in = torch.cat([image, x_back], dim=1)
        fore_in = torch.cat([image, x_fore], dim=1)

        # stems
        back = self.back_stem(back_in)
        fore = self.fore_stem(fore_in)

        # encode and collect skips
        back_skips: List[torch.Tensor] = []
        fore_skips: List[torch.Tensor] = []

        for level, (enc_b, enc_f) in enumerate(zip(self.back_enc, self.fore_enc)):
            back, b_skip = enc_b(back, t_emb)
            fore, f_skip = enc_f(fore, t_emb)

            # skip-level coupling on deepest K skips
            if level in self.attn_level_set:
                attn = self.freq_attn_modules[str(level)]
                f_skip, b_skip = attn(f_skip, b_skip)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)

        # ✅ NEW (3): bottleneck coupling at deepest latent
        # (the last encoder stage has down=False, so back/fore here are at deepest resolution)
        if self.bottleneck_coupler is not None:
            fore, back = self.bottleneck_coupler(fore, back)

        # start decoding from (coupled) bottleneck latents
        back_h = fore_skips[-1].new_empty(0)  # just to satisfy type check (won't be used)
        fore_h = fore_skips[-1].new_empty(0)

        # Use the coupled bottleneck as initial hidden state
        back_h = back
        fore_h = fore

        aux_preds: List[torch.Tensor] = []

        # decode (independent) with gated skips
        # note: decoder lists are already in deep->shallow order
        for stage_idx, (dec_b, dec_f, b_skip, f_skip) in enumerate(
            zip(self.back_dec, self.fore_dec, reversed(back_skips), reversed(fore_skips))
        ):
            back_h = dec_b(back_h, b_skip, t_emb)
            fore_h = dec_f(fore_h, f_skip, t_emb)

            # ✅ NEW (4): multi-scale head at each decoder stage output
            if self.deep_supervision and return_aux:
                h_ms = torch.cat([fore_h, back_h], dim=1)
                h_ms = F.silu(self.ms_norms[stage_idx](h_ms))
                v_ms = self.ms_convs[stage_idx](h_ms)
                # upsample to full resolution (same as final output)
                # current spatial size is fore_h.size(-2/-1); final is same as image
                if v_ms.shape[-2:] != image.shape[-2:]:
                    v_ms = F.interpolate(v_ms, size=image.shape[-2:], mode="bilinear", align_corners=False)
                aux_preds.append(v_ms)

        # final head: concat decoded fore/back at base resolution
        h = torch.cat([fore_h, back_h], dim=1)  # (B, 2*base_ch, H, W)
        h = F.silu(self.out_norm(h))
        v = self.out_conv(h)

        if self.deep_supervision and return_aux:
            return v, aux_preds
        return v


# -----------------------------
# Factory
# -----------------------------
def build_model(
    img_in_ch: int = 1,
    xt_in_ch: int = 4,
    base_ch: int = 64,
    ch_mult: Sequence[int] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    time_dim: int = 256,
    dropout: float = 0.0,
    attn_heads: int = 4,
    attn_head_dim: int = 32,
    freq_attn_depth: int = 1,
    # ✅ expose new knobs
    use_bottleneck_coupling: bool = True,
    bottleneck_gate_init: float = 0.0,
    use_attn_gated_skip: bool = True,
    deep_supervision: bool = False,
) -> FMForeBackDeepKFreqXAttnDualDecoderResUNet:
    cfg = Config(
        img_in_ch=img_in_ch,
        xt_in_ch=xt_in_ch,
        base_ch=base_ch,
        ch_mult=tuple(ch_mult),
        num_res_blocks=num_res_blocks,
        time_dim=time_dim,
        dropout=dropout,
        attn_heads=attn_heads,
        attn_head_dim=attn_head_dim,
        freq_attn_depth=freq_attn_depth,
        use_bottleneck_coupling=use_bottleneck_coupling,
        bottleneck_gate_init=bottleneck_gate_init,
        use_attn_gated_skip=use_attn_gated_skip,
        deep_supervision=deep_supervision,
    )
    return FMForeBackDeepKFreqXAttnDualDecoderResUNet(cfg)


# -----------------------------
# Sanity
# -----------------------------
if __name__ == "__main__":
    net = build_model(
        img_in_ch=1,
        xt_in_ch=4,
        base_ch=32,
        freq_attn_depth=2,
        use_bottleneck_coupling=True,
        use_attn_gated_skip=True,
        deep_supervision=True,
    )
    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v, aux = net(x_t, img, t, return_aux=True)
    print("v:", v.shape)
    print("aux:", len(aux), aux[0].shape if len(aux) else None)