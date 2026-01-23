# -*- coding: utf-8 -*-
"""
...（省略原注释，内容不变）

✅ NEW: FA is now a *bound pair* of two TriFFT blocks, treated as ONE FA module:
    - Step 1: TriFFT WITH the radial operator (time-conditioned gate)
    - Step 2: TriFFT WITHOUT the operator (pure attention update)

✅ NEW (your request): BoundFAModule stack count is now a hyperparameter.
    - cfg.fa_stacks (default=2) controls how many BoundFAModule blocks to stack
      at each original FA insertion position (encoder selected levels + bottleneck).
    - (fore, back, cond) -> [BoundFAModule]^fa_stacks

✅ CHANGE: NO input-level concat(image, x_t).
    - back encoder sees ONLY x_back
    - fore encoder sees ONLY x_fore
    - cond encoder sees ONLY image
    - fusion happens ONLY in FA modules and decoder skip fusion
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
        # NOTE: in your current network, x and c at the same level share H,W.
        # Keeping your original behavior (use x's H,W). If you want fully general
        # cross-attn across different resolutions, change hk,wk = c.shape[-2:].
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
# Time-conditioned radial spectral gate g(r,t)=exp(-alpha(t)*r^p)
# -----------------------------
class TimeRadialSpectralGate(nn.Module):
    def __init__(
        self,
        time_ch: int,
        p: float = 2.0,
        alpha_init: float = 0.0,
        per_channel: bool = False,
        ch: int = 0,
    ):
        super().__init__()
        self.p = float(p)
        self.per_channel = bool(per_channel)
        self.ch = int(ch)

        out_dim = (2 * ch) if self.per_channel else 1
        self.to_alpha = nn.Linear(time_ch, out_dim)

        inv = math.log(math.expm1(max(alpha_init, 1e-6)))
        nn.init.constant_(self.to_alpha.weight, 0.0)
        nn.init.constant_(self.to_alpha.bias, inv)

        self._grid_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def _radius_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]

        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, w).expand(h, w)
        r = torch.sqrt(xx * xx + yy * yy)
        r = r / (r.max().clamp_min(1e-6))
        r = r[None, None, :, :]  # (1,1,H,W)

        self._grid_cache[key] = r
        return r

    def forward(self, t_emb: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = t_emb.shape[0]
        r = self._radius_grid(h, w, t_emb.device, t_emb.dtype)

        alpha = F.softplus(self.to_alpha(t_emb))  # >=0
        if self.per_channel:
            alpha = alpha.view(b, 2 * self.ch, 1, 1)
            mask = torch.exp(-alpha * (r ** self.p))
        else:
            alpha = alpha.view(b, 1, 1, 1)
            mask = torch.exp(-alpha * (r ** self.p))
        return mask


# -----------------------------
# Tri-FFT Cross Attention Module
# (optionally uses sign(Δ) * max(|Δ|,|gΔ|))
# -----------------------------
class TriFFTCrossAttention(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int = 4,
        head_dim: int = 32,
        use_time_radial_gate: bool = True,
        radial_p: float = 2.0,
        radial_alpha_init: float = 0.0,
        radial_per_channel: bool = False,
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

        if self.use_time_radial_gate:
            self.radial_gate = TimeRadialSpectralGate(
                time_ch=time_ch,
                p=radial_p,
                alpha_init=radial_alpha_init,
                per_channel=radial_per_channel,
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

    def _apply_gate(self, delta_ri: torch.Tensor, t_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if (not self.use_time_radial_gate) or (self.radial_gate is None) or (t_emb is None):
            return delta_ri
        _, _, h, w = delta_ri.shape
        g = self.radial_gate(t_emb, h, w)
        delta_g = delta_ri * g
        return torch.sign(delta_ri) * torch.maximum(delta_ri.abs(), delta_g.abs())

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Fz = torch.fft.fftshift(torch.fft.fft2(fore, dim=(-2, -1)), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(back, dim=(-2, -1)), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1)), dim=(-2, -1))

        Fri = self._to_ri(Fz)
        Bri = self._to_ri(Bz)
        Cri = self._to_ri(Cz)

        d_fb = self._apply_gate(self.attn_fb(Fri, Bri), t_emb)
        d_fc = self._apply_gate(self.attn_fc(Fri, Cri), t_emb)
        Fri2 = Fri + d_fb + d_fc
        Fri2 = self.norm_f(self.post_f(Fri2))

        d_bc = self._apply_gate(self.attn_bc(Bri, Cri), t_emb)
        Bri2 = Bri + d_bc
        Bri2 = self.norm_b(self.post_b(Bri2))

        F2c = self._to_complex(Fri2)
        B2c = self._to_complex(Bri2)

        F2c = torch.fft.ifftshift(F2c, dim=(-2, -1))
        B2c = torch.fft.ifftshift(B2c, dim=(-2, -1))

        fore_out = torch.fft.ifft2(F2c, dim=(-2, -1)).real
        back_out = torch.fft.ifft2(B2c, dim=(-2, -1)).real
        return fore_out, back_out


# -----------------------------
# Bound FA module = (op-on) then (op-off)
# -----------------------------
class BoundFAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        radial_p: float,
        radial_alpha_init: float,
        radial_per_channel: bool,
        use_time_radial_gate: bool = True,
    ):
        super().__init__()

        self.step1 = TriFFTCrossAttention(
            ch=ch,
            time_ch=time_ch,
            heads=heads,
            head_dim=head_dim,
            use_time_radial_gate=use_time_radial_gate,
            radial_p=radial_p,
            radial_alpha_init=radial_alpha_init,
            radial_per_channel=radial_per_channel,
        )

        self.step2 = TriFFTCrossAttention(
            ch=ch,
            time_ch=time_ch,
            heads=heads,
            head_dim=head_dim,
            use_time_radial_gate=False,  # forced OFF
            radial_p=radial_p,
            radial_alpha_init=radial_alpha_init,
            radial_per_channel=radial_per_channel,
        )

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f0, b0 = fore, back
        fore, back = self.step1(fore, back, cond, t_emb=t_emb)
        fore, back = self.step2(fore, back, cond, t_emb=t_emb)
        return fore + f0, back + b0


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

    # operator knobs (used by FA step1 only)
    use_time_radial_gate: bool = True
    radial_p: float = 2.0
    radial_alpha_init: float = 0.0
    radial_per_channel: bool = False

    # ✅ NEW: number of BoundFAModule stacked per location (default=2)
    fa_stacks: int = 2


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

        # ✅ stems: no image concat
        back_in_ch = 1               # x_back only
        fore_in_ch = cfg.xt_in_ch - 1  # x_fore only
        cond_in_ch = cfg.img_in_ch   # image only

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
                        radial_p=cfg.radial_p,
                        radial_alpha_init=cfg.radial_alpha_init,
                        radial_per_channel=cfg.radial_per_channel,
                        use_time_radial_gate=cfg.use_time_radial_gate,
                    )
                )
            return mods

        # encoder-level FA insertion (same original positions)
        self.couple_modules = nn.ModuleDict()
        for i in sorted(self.attn_level_set):
            self.couple_modules[str(i)] = make_bound_fa_list(self.chs[i])

        # bottleneck FA insertion (same original position)
        self.bottleneck_coupler = make_bound_fa_list(self.chs[-1]) if cfg.use_bottleneck_coupling else None

        # decoders
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

        # split x_t
        x_back = x_t[:, 0:1]
        x_fore = x_t[:, 1:]

        # no input cat
        back = self.back_stem(x_back)
        fore = self.fore_stem(x_fore)
        cond = self.cond_stem(image)

        back_skips: List[torch.Tensor] = []
        fore_skips: List[torch.Tensor] = []
        cond_skips: List[torch.Tensor] = []

        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            back, b_skip = enc_b(back, t_emb)
            fore, f_skip = enc_f(fore, t_emb)
            cond, c_skip = enc_c(cond, t_emb)

            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_skip, b_skip = fa(f_skip, b_skip, c_skip, t_emb=t_emb)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c_skip)

        if self.bottleneck_coupler is not None:
            for fa in self.bottleneck_coupler:
                fore, back = fa(fore, back, cond, t_emb=t_emb)

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
    radial_p: float = 2.0,
    radial_alpha_init: float = 0.0,
    radial_per_channel: bool = False,
    # ✅ NEW: BoundFAModule stack count (default=2)
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
        radial_p=radial_p,
        radial_alpha_init=radial_alpha_init,
        radial_per_channel=radial_per_channel,
        fa_stacks=fa_stacks,
    )
    return FMForeBackCondDeepKTriFFTDualDecoderResUNet(cfg)


if __name__ == "__main__":
    net = build_model(
        img_in_ch=1,
        xt_in_ch=4,
        base_ch=32,
        freq_attn_depth=2,
        fa_stacks=2,  # ✅ default is 2; explicit here for clarity
        use_bottleneck_coupling=True,
        deep_supervision=True,
    )
    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v, aux = net(x_t, img, t, return_aux=True)
    print("v:", v.shape)
    print("aux:", len(aux), aux[0].shape if len(aux) else None)