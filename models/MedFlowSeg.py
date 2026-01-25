# -*- coding: utf-8 -*-
"""
✅ FA is a bound pair of two TriFFT blocks:
    - Step 1: TriFFT WITH Gaussian High-pass operator (time-conditioned sigma)
             step1 uses cond as Query (q=cond, kv=fore/back) to UPDATE cond only
             ✅ CHANGE: filter is applied BEFORE attention (pre-filter)
    - Step 2: TriFFT WITHOUT the operator (pure attention update)
             step2 keeps original behavior: UPDATE fore/back, cond as KV only

✅ BoundFAModule stack count is a hyperparameter: cfg.fa_stacks
✅ NO input-level concat(image, x_t)

Gaussian HP in frequency domain:
  g_hp(r,t) = 1 - exp( - r^2 / (2*sigma(t)^2) )
where r is the normalized radial frequency coordinate computed from torch.fft.fftfreq + fftshift.

✅ MOD (per your request):
- Encoder: back/fore/cond ("main stream" passed to next level) go through FA(TriFFT).
- Skip tensors b_skip/f_skip/c_skip remain RAW (no FA), and are used for decoder skip connections.
  i.e. FA affects the down-path features, not the skip features.

Implementation detail:
- We run only enc_*.blocks to obtain raw skip (pre-downsample).
- Save raw skips.
- Apply FA on copies (main stream).
- Downsample the FA-processed main stream via enc_*.down to feed next level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, Optional, Dict

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
# Spatial Cross Attention (multi-head)
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
# Time-conditioned Gaussian High-pass gate (frequency-domain)
# r uses true frequency bins from torch.fft.fftfreq + fftshift, then normalized to [0,1]
# g_hp(r,t) = 1 - exp( -r^2 / (2*sigma(t)^2) )
# -----------------------------
class TimeGaussianHighPassGate(nn.Module):
    def __init__(
        self,
        time_ch: int,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
        sigma_init: float = 0.15,
        per_channel: bool = False,
        ch: int = 0,  # base channels; RI has 2*ch channels
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.per_channel = bool(per_channel)
        self.ch = int(ch)
        self.eps = float(eps)

        out_dim = (2 * ch) if self.per_channel else 1
        self.to_sigma = nn.Linear(time_ch, out_dim)

        sigma_init = min(max(sigma_init, sigma_min + 1e-6), sigma_max - 1e-6)
        s = (sigma_init - sigma_min) / (sigma_max - sigma_min)
        bias = math.log(s / (1.0 - s))
        nn.init.constant_(self.to_sigma.weight, 0.0)
        nn.init.constant_(self.to_sigma.bias, bias)

        self._grid_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def _radius_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]

        fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)  # (H,)
        fx = torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)  # (W,)

        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        yy, xx = torch.meshgrid(fy, fx, indexing="ij")  # (H,W)
        r = torch.sqrt(xx * xx + yy * yy)               # (H,W)

        r = r / (r.max().clamp_min(self.eps))
        r = r[None, None, :, :]  # (1,1,H,W)

        self._grid_cache[key] = r
        return r

    def forward(self, t_emb: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = t_emb.shape[0]
        r = self._radius_grid(h, w, t_emb.device, t_emb.dtype)  # (1,1,H,W)

        sigma = torch.sigmoid(self.to_sigma(t_emb))  # (B,out_dim) in (0,1)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma
        sigma = sigma.clamp_min(self.eps)

        if self.per_channel:
            sigma = sigma.view(b, 2 * self.ch, 1, 1)
            g_hp = 1.0 - torch.exp(-(r * r) / (2.0 * sigma * sigma))
        else:
            sigma = sigma.view(b, 1, 1, 1)
            g_hp = 1.0 - torch.exp(-(r * r) / (2.0 * sigma * sigma))

        return g_hp


# -----------------------------
# Tri-FFT Cross Attention Module
# ✅ CHANGE: Gaussian HP filter is applied BEFORE attention via _pre_filter()
# -----------------------------
class TriFFTCrossAttention(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int = 4,
        head_dim: int = 32,
        use_time_radial_gate: bool = True,
        g_sigma_min: float = 0.05,
        g_sigma_max: float = 0.50,
        g_sigma_init: float = 0.15,
        g_per_channel: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.use_time_radial_gate = bool(use_time_radial_gate)

        self.attn_fb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.attn_fc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.attn_bc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)

        self.post_f = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.post_b = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.norm_f = GroupNorm32(2 * ch)
        self.norm_b = GroupNorm32(2 * ch)

        self.attn_cf = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.attn_cb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.post_c = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.norm_c = GroupNorm32(2 * ch)

        if self.use_time_radial_gate:
            self.radial_gate = TimeGaussianHighPassGate(
                time_ch=time_ch,
                sigma_min=g_sigma_min,
                sigma_max=g_sigma_max,
                sigma_init=g_sigma_init,
                per_channel=g_per_channel,
                ch=ch,
            )
        else:
            self.radial_gate = None

    @staticmethod
    def _to_ri(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def _to_complex(ri: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = ri.shape
        c = c2 // 2
        return torch.complex(ri[:, :c], ri[:, c:])

    def _pre_filter(self, x_ri: torch.Tensor, t_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if (not self.use_time_radial_gate) or (self.radial_gate is None) or (t_emb is None):
            return x_ri
        _, _, h, w = x_ri.shape
        g_hp = self.radial_gate(t_emb, h, w)
        return x_ri * g_hp

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        update_fb: bool = True,
        update_c: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Fz = torch.fft.fftshift(
            torch.fft.fft2(fore, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )
        Bz = torch.fft.fftshift(
            torch.fft.fft2(back, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )
        Cz = torch.fft.fftshift(
            torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )

        Fri = self._to_ri(Fz)
        Bri = self._to_ri(Bz)
        Cri = self._to_ri(Cz)

        Fri_q = self._pre_filter(Fri, t_emb)
        Bri_q = self._pre_filter(Bri, t_emb)
        Cri_q = self._pre_filter(Cri, t_emb)

        Fri_kv = Fri
        Bri_kv = Bri
        Cri_kv = Cri

        if update_fb:
            d_fb = self.attn_fb(Fri_q, Bri_kv)
            d_fc = self.attn_fc(Fri_q, Cri_kv)
            Fri2 = self.norm_f(self.post_f(Fri + d_fb + d_fc))

            d_bc = self.attn_bc(Bri_q, Cri_kv)
            Bri2 = self.norm_b(self.post_b(Bri + d_bc))
        else:
            Fri2, Bri2 = Fri, Bri

        if update_c:
            d_cf = self.attn_cf(Cri_q, Fri_kv)
            d_cb = self.attn_cb(Cri_q, Bri_kv)
            Cri2 = self.norm_c(self.post_c(Cri + d_cf + d_cb))
        else:
            Cri2 = Cri

        F2c = self._to_complex(Fri2)
        B2c = self._to_complex(Bri2)
        C2c = self._to_complex(Cri2)

        F2c = torch.fft.ifftshift(F2c, dim=(-2, -1))
        B2c = torch.fft.ifftshift(B2c, dim=(-2, -1))
        C2c = torch.fft.ifftshift(C2c, dim=(-2, -1))

        fore_out = torch.fft.ifft2(F2c, dim=(-2, -1), norm="ortho").real
        back_out = torch.fft.ifft2(B2c, dim=(-2, -1), norm="ortho").real
        cond_out = torch.fft.ifft2(C2c, dim=(-2, -1), norm="ortho").real
        return fore_out, back_out, cond_out


# -----------------------------
# Bound FA module = (cond-update) then (fore/back-update)
# -----------------------------
class BoundFAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        use_time_radial_gate: bool = True,
        g_sigma_min: float = 0.05,
        g_sigma_max: float = 0.50,
        g_sigma_init: float = 0.15,
        g_per_channel: bool = False,
    ):
        super().__init__()

        self.step1 = TriFFTCrossAttention(
            ch=ch,
            time_ch=time_ch,
            heads=heads,
            head_dim=head_dim,
            use_time_radial_gate=use_time_radial_gate,
            g_sigma_min=g_sigma_min,
            g_sigma_max=g_sigma_max,
            g_sigma_init=g_sigma_init,
            g_per_channel=g_per_channel,
        )

        self.step2 = TriFFTCrossAttention(
            ch=ch,
            time_ch=time_ch,
            heads=heads,
            head_dim=head_dim,
            use_time_radial_gate=False,
            g_sigma_min=g_sigma_min,
            g_sigma_max=g_sigma_max,
            g_sigma_init=g_sigma_init,
            g_per_channel=g_per_channel,
        )

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c0 = cond
        fore, back, cond = self.step1(fore, back, cond, t_emb=t_emb, update_fb=False, update_c=True)
        fore, back, cond = self.step2(fore, back, cond, t_emb=t_emb, update_fb=True, update_c=False)
        return fore, back, cond


# -----------------------------
# Decoder attention-gated skip
# -----------------------------
class AttentionGate(nn.Module):
    def __init__(self, x_ch: int, skip_ch: int, inter_ch: Optional[int] = None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(8, min(x_ch, skip_ch) // 2)

        self.Wg = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), GroupNorm32(inter_ch))
        self.Ws = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), GroupNorm32(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        a = F.silu(self.Wg(x) + self.Ws(skip))
        gate = self.psi(a)
        return skip * gate


# -----------------------------
# Encoder stage (FiLM)
# -----------------------------
class EncoderStageFiLM(nn.Module):
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


# -----------------------------
# Decoder stage with Cond injection (ADD to skip)
# -----------------------------
class DecoderStageCondAddSkip(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        cond_ch: int,
        out_ch: int,
        time_ch: int,
        num_blocks: int,
        up: bool,
        dropout: float,
        use_attn_gate: bool = True,
        cond_proj: bool = True,
        fuse_norm: bool = True,
    ):
        super().__init__()
        self.up = Upsample(in_ch) if up else None

        self.use_attn_gate = use_attn_gate
        self.gate = AttentionGate(x_ch=in_ch, skip_ch=skip_ch) if use_attn_gate else None

        if cond_proj:
            self.cond_proj = nn.Conv2d(cond_ch, skip_ch, 1, bias=False)
        else:
            if cond_ch != skip_ch:
                raise ValueError("cond_proj=False requires cond_ch == skip_ch")
            self.cond_proj = nn.Identity()

        self.fuse_norm = GroupNorm32(skip_ch) if fuse_norm else nn.Identity()

        blocks: List[nn.Module] = []
        ch = in_ch + skip_ch
        for _ in range(num_blocks):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond_skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)

        if self.use_attn_gate and self.gate is not None:
            skip = self.gate(x, skip)

        cond_skip = self.cond_proj(cond_skip)
        skip = self.fuse_norm(skip + cond_skip)

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
    xt_in_ch: int = 4
    base_ch: int = 64
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    time_dim: int = 256
    dropout: float = 0.0

    attn_heads: int = 4
    attn_head_dim: int = 32

    freq_attn_depth: int = 1
    use_bottleneck_coupling: bool = True
    use_attn_gated_skip: bool = True

    use_cond_add_to_skip: bool = True
    cond_proj: bool = True
    cond_fuse_norm: bool = True

    deep_supervision: bool = False

    use_time_radial_gate: bool = True
    g_sigma_min: float = 0.05
    g_sigma_max: float = 0.50
    g_sigma_init: float = 0.15
    g_per_channel: bool = False

    fa_stacks: int = 1


class FMForeBackCondDeepKTriFFTDualDecoderResUNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.xt_in_ch < 2:
            raise ValueError("xt_in_ch must be >=2 (channel0 background, remaining foreground).")
        if cfg.freq_attn_depth < 1:
            raise ValueError("freq_attn_depth must be >= 1.")
        if not cfg.use_cond_add_to_skip:
            raise ValueError("This file implements Method A (use_cond_add_to_skip=True).")
        if cfg.fa_stacks < 1:
            raise ValueError("fa_stacks must be >= 1")

        self.cfg = cfg
        self.time_mlp = TimeMLP(time_dim=cfg.time_dim, hidden_dim=cfg.time_dim)

        back_in_ch = 1
        fore_in_ch = cfg.xt_in_ch - 1
        cond_in_ch = cfg.img_in_ch

        self.back_stem = nn.Conv2d(back_in_ch, cfg.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(fore_in_ch, cfg.base_ch, 3, padding=1)
        self.cond_stem = nn.Conv2d(cond_in_ch, cfg.base_ch, 3, padding=1)

        self.chs = [cfg.base_ch * m for m in cfg.ch_mult]
        n_levels = len(self.chs)

        self.back_enc = nn.ModuleList()
        self.fore_enc = nn.ModuleList()
        self.cond_enc = nn.ModuleList()
        for i in range(n_levels):
            in_ch = self.chs[i - 1] if i > 0 else cfg.base_ch
            out_ch = self.chs[i]
            down = i < (n_levels - 1)

            self.back_enc.append(EncoderStageFiLM(in_ch, out_ch, cfg.time_dim, cfg.num_res_blocks, down, cfg.dropout))
            self.fore_enc.append(EncoderStageFiLM(in_ch, out_ch, cfg.time_dim, cfg.num_res_blocks, down, cfg.dropout))
            self.cond_enc.append(EncoderStageFiLM(in_ch, out_ch, cfg.time_dim, cfg.num_res_blocks, down, cfg.dropout))

        K = min(cfg.freq_attn_depth, n_levels)
        self.attn_level_set = set(range(n_levels - K, n_levels))

        def make_bound_fa_list(ch: int) -> nn.ModuleList:
            mods = nn.ModuleList()
            for _ in range(cfg.fa_stacks):
                mods.append(
                    BoundFAModule(
                        ch=ch,
                        time_ch=cfg.time_dim,
                        heads=cfg.attn_heads,
                        head_dim=cfg.attn_head_dim,
                        use_time_radial_gate=cfg.use_time_radial_gate,
                        g_sigma_min=cfg.g_sigma_min,
                        g_sigma_max=cfg.g_sigma_max,
                        g_sigma_init=cfg.g_sigma_init,
                        g_per_channel=cfg.g_per_channel,
                    )
                )
            return mods

        self.couple_modules = nn.ModuleDict()
        for i in sorted(self.attn_level_set):
            self.couple_modules[str(i)] = make_bound_fa_list(self.chs[i])

        self.bottleneck_coupler = make_bound_fa_list(self.chs[-1]) if cfg.use_bottleneck_coupling else None

        self.back_dec = nn.ModuleList()
        self.fore_dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.chs[i]
            cond_ch = self.chs[i]
            in_ch = self.chs[i + 1] if i < (n_levels - 1) else self.chs[i]
            out_ch = self.chs[i]
            up = i < (n_levels - 1)

            self.back_dec.append(
                DecoderStageCondAddSkip(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    cond_ch=cond_ch,
                    out_ch=out_ch,
                    time_ch=cfg.time_dim,
                    num_blocks=cfg.num_res_blocks,
                    up=up,
                    dropout=cfg.dropout,
                    use_attn_gate=cfg.use_attn_gated_skip,
                    cond_proj=cfg.cond_proj,
                    fuse_norm=cfg.cond_fuse_norm,
                )
            )
            self.fore_dec.append(
                DecoderStageCondAddSkip(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    cond_ch=cond_ch,
                    out_ch=out_ch,
                    time_ch=cfg.time_dim,
                    num_blocks=cfg.num_res_blocks,
                    up=up,
                    dropout=cfg.dropout,
                    use_attn_gate=cfg.use_attn_gated_skip,
                    cond_proj=cfg.cond_proj,
                    fuse_norm=cfg.cond_fuse_norm,
                )
            )

        self.out_norm = GroupNorm32(cfg.base_ch * 2)
        self.out_conv = nn.Conv2d(cfg.base_ch * 2, cfg.xt_in_ch, 3, padding=1)

        self.deep_supervision = cfg.deep_supervision
        if self.deep_supervision:
            self.ms_norms = nn.ModuleList()
            self.ms_convs = nn.ModuleList()
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

        x_back = x_t[:, 0:1]
        x_fore = x_t[:, 1:]

        back = self.back_stem(x_back)
        fore = self.fore_stem(x_fore)
        cond = self.cond_stem(image)

        back_skips: List[torch.Tensor] = []
        fore_skips: List[torch.Tensor] = []
        cond_skips: List[torch.Tensor] = []

        # ============================================================
        # ✅ MOD: main stream (back/fore/cond) goes through FA and then down
        # ✅     raw skips (b_skip/f_skip/c_skip) stay unchanged for decoder
        # ============================================================
        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            # 1) Run only blocks to get "pre-downsample" features (raw skip)
            xb = back
            for blk in enc_b.blocks:
                xb = blk(xb, t_emb)
            b_skip = xb  # RAW skip (no FA)

            xf = fore
            for blk in enc_f.blocks:
                xf = blk(xf, t_emb)
            f_skip = xf  # RAW skip (no FA)

            xc = cond
            for blk in enc_c.blocks:
                xc = blk(xc, t_emb)
            c_skip = xc  # RAW skip (no FA)

            # 2) Save RAW skips for decoder
            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c_skip)

            # 3) Main stream: apply FA on copies (affects only back/fore down-path)
            b_main, f_main, c_tmp = b_skip, f_skip, c_skip
            
            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    # cond participates in FA, but we DO NOT write it back to main-stream cond
                    f_main, b_main, c_tmp = fa(f_main, b_main, c_tmp, t_emb=t_emb)
            
            # 4) Downsample main stream to feed next level
            back = enc_b.down(b_main) if enc_b.down is not None else b_main
            fore = enc_f.down(f_main) if enc_f.down is not None else f_main
            
            cond = enc_c.down(c_skip) if enc_c.down is not None else c_skip

        # bottleneck coupling (keeps original behavior: acts on main stream)
        if self.bottleneck_coupler is not None:
            for fa in self.bottleneck_coupler:
                fore, back, cond = fa(fore, back, cond, t_emb=t_emb)

        back_h = back
        fore_h = fore
        aux_preds: List[torch.Tensor] = []

        for stage_idx, (dec_b, dec_f, b_skip, f_skip, c_skip) in enumerate(
            zip(self.back_dec, self.fore_dec, reversed(back_skips), reversed(fore_skips), reversed(cond_skips))
        ):
            back_h = dec_b(back_h, b_skip, c_skip, t_emb)
            fore_h = dec_f(fore_h, f_skip, c_skip, t_emb)

            if self.deep_supervision and return_aux:
                h_ms = torch.cat([fore_h, back_h], dim=1)
                h_ms = F.silu(self.ms_norms[stage_idx](h_ms))
                v_ms = self.ms_convs[stage_idx](h_ms)
                if v_ms.shape[-2:] != image.shape[-2:]:
                    v_ms = F.interpolate(v_ms, size=image.shape[-2:], mode="bilinear", align_corners=False)
                aux_preds.append(v_ms)

        h = torch.cat([fore_h, back_h], dim=1)
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
    ch_mult: Sequence[int] = (1, 2, 4, 4, 8),
    num_res_blocks: int = 2,
    time_dim: int = 256,
    dropout: float = 0.0,
    attn_heads: int = 4,
    attn_head_dim: int = 32,
    freq_attn_depth: int = 1,
    use_bottleneck_coupling: bool = True,
    use_attn_gated_skip: bool = True,
    deep_supervision: bool = False,
    use_cond_add_to_skip: bool = True,
    cond_proj: bool = True,
    cond_fuse_norm: bool = True,
    use_time_radial_gate: bool = True,
    g_sigma_min: float = 0.05,
    g_sigma_max: float = 0.50,
    g_sigma_init: float = 0.15,
    g_per_channel: bool = False,
    fa_stacks: int = 2,
) -> FMForeBackCondDeepKTriFFTDualDecoderResUNet:
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
        use_attn_gated_skip=use_attn_gated_skip,
        deep_supervision=deep_supervision,
        use_cond_add_to_skip=use_cond_add_to_skip,
        cond_proj=cond_proj,
        cond_fuse_norm=cond_fuse_norm,
        use_time_radial_gate=use_time_radial_gate,
        g_sigma_min=g_sigma_min,
        g_sigma_max=g_sigma_max,
        g_sigma_init=g_sigma_init,
        g_per_channel=g_per_channel,
        fa_stacks=fa_stacks,
    )
    return FMForeBackCondDeepKTriFFTDualDecoderResUNet(cfg)


if __name__ == "__main__":
    net = build_model(
        img_in_ch=1,
        xt_in_ch=4,
        base_ch=32,
        freq_attn_depth=2,
        fa_stacks=2,
        use_bottleneck_coupling=True,
        deep_supervision=True,
        use_time_radial_gate=True,
        g_sigma_min=0.05,
        g_sigma_max=0.50,
        g_sigma_init=0.15,
    )
    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v, aux = net(x_t, img, t, return_aux=True)
    print("v:", v.shape)
    print("aux:", len(aux), aux[0].shape if len(aux) else None)