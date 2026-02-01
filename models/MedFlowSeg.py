# -*- coding: utf-8 -*-
"""
✅ FA: one-pass (no step1), fore/back updated using cond as KV
✅ REMOVE Modulator entirely (NO rank / NO basis decomposition)
✅ Keep frequency-domain attention ops:
    - (CHANGED,方案B) 直接从 f_in/b_in 预测更新后的 Fri/Bri（不再做 mask 乘法）
    - TriFFT attention updates fore/back using cond as KV (cond not updated)

✅ HP inject: unchanged
✅ NEW: add a Cond Decoder (U-Net style)
    - Cond decoder connects to cond encoder via skip connections
    - Shared decoder NO LONGER uses cond encoder skips
    - Shared decoder now uses cond decoder features (per-level)

✅ Decoder: concat(back, fore) -> single decoder -> output v directly
✅ Decoder stage: direct add cond_skip (no gate)   (now cond_skip comes from cond decoder)

(方案B说明)
- 去掉 Mf/Mb mask 分支（sigmoid mask + apply_mask_to_complex）
- 改为：
    Fri,Bri,Cri = to_ri(Fz),to_ri(Bz),to_ri(Cz)
    f_in = cat([Fri,Cri]) -> Head_f -> Fri_new   (B,2C,H,W)
    b_in = cat([Bri,Cri]) -> Head_b -> Bri_new   (B,2C,H,W)
  然后进入三角交互 (fb/fc/bc) 不变
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, Union

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
# Spatial Cross Attention (multi-head)  ✅ with Norm
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
        out = torch.einsum("bhij,bhdj->bhdi", attn, v).reshape(
            b, self.heads * self.head_dim, h, w
        )
        return self.out(out)


# -----------------------------
# Gaussian HP gate (time-conditioned per-channel sigma)
# -----------------------------
class TimeGaussianHighPassGatePerChannel(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
        sigma_init: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ch = int(ch)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)

        self.to_sigma = nn.Linear(time_ch, self.ch)

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

        fy = torch.fft.fftfreq(h, d=1.0, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(w, d=1.0, device=device, dtype=dtype)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(xx * xx + yy * yy)
        r = r / (r.max().clamp_min(self.eps))
        r = r[None, None, :, :]
        self._grid_cache[key] = r
        return r

    def forward(self, t_emb: torch.Tensor, h: int, w: int) -> torch.Tensor:
        r = self._radius_grid(h, w, t_emb.device, t_emb.dtype)

        sigma = torch.sigmoid(self.to_sigma(t_emb))
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma
        sigma = sigma.clamp_min(self.eps).view(t_emb.shape[0], self.ch, 1, 1)

        g_hp = 1.0 - torch.exp(-(r * r) / (2.0 * sigma * sigma))
        return g_hp


class CondGaussianHighPassOperator(nn.Module):
    """Injection HP: c2 = IFFT( FFT(c1) * g_hp ).real"""

    def __init__(
        self,
        ch: int,
        time_ch: int,
        sigma_min: float = 0.05,
        sigma_max: float = 0.50,
        sigma_init: float = 0.15,
    ):
        super().__init__()
        self.ch = int(ch)
        self.gate = TimeGaussianHighPassGatePerChannel(
            ch=self.ch,
            time_ch=time_ch,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_init=sigma_init,
        )

    def forward(self, c1: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = c1.shape
        if c != self.ch:
            raise ValueError(f"CondGaussianHighPassOperator expected C={self.ch}, got {c}")

        Cz = torch.fft.fftshift(torch.fft.fft2(c1, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        g_hp = self.gate(t_emb, h, w).to(Cz.dtype)
        Cz = Cz * g_hp
        Cz = torch.fft.ifftshift(Cz, dim=(-2, -1))
        c2 = torch.fft.ifft2(Cz, dim=(-2, -1), norm="ortho").real
        return c2


class CondGateResidual(nn.Module):
    """x' = sigmoid(Conv1x1(cond_hp))*x + x"""

    def __init__(self, x_ch: int, cond_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(cond_ch, x_ch, 1, bias=True)

    def forward(self, x: torch.Tensor, cond_hp: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(cond_hp))
        return gate * x + x


# -----------------------------
# (方案B) Frequency-domain direct head: from (Fri,Cri)->Fri'  (FiLM conditioned on time)
# 输入: (B,4C,H,W) 输出: (B,2C,H,W)
# -----------------------------
class FreqDirectHeadFiLM(nn.Module):
    def __init__(self, ch: int, time_ch: int, hidden_mult: int = 2):
        super().__init__()
        self.ch = int(ch)
        hid = int(hidden_mult * ch)

        # in: 4C  -> hid
        self.in_norm = GroupNorm32(4 * ch)
        self.conv_in = nn.Conv2d(4 * ch, hid, 3, padding=1)

        # FiLM on hid
        self.hid_norm = GroupNorm32(hid)
        self.to_film = nn.Sequential(nn.SiLU(), nn.Linear(time_ch, 2 * hid))

        self.conv_mid = nn.Conv2d(hid, hid, 3, padding=1)

        # out: hid -> 2C (直接预测新的 Fri/Bri)
        self.conv_out = nn.Conv2d(hid, 2 * ch, 1)

        # 小初始化：避免一上来把频谱打爆
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x_ri4c: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(F.silu(self.in_norm(x_ri4c)))

        h = self.hid_norm(h)
        gamma, beta = self.to_film(t_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = (1.0 + gamma) * h + beta

        h = F.silu(self.conv_mid(F.silu(h)))
        out = self.conv_out(h)
        return out  # (B,2C,H,W)


# -----------------------------
# FA module (one-pass) + (方案B) direct freq update
# -----------------------------
class FAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        noise_strength: float = 1.0,  # 兼容保留，但方案B不再使用
    ):
        super().__init__()
        self.ch = int(ch)
        self.noise_strength = float(noise_strength)

        self.attn_fb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.attn_fc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.attn_bc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)

        self.post_f = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.post_b = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.norm_f = GroupNorm32(2 * ch)
        self.norm_b = GroupNorm32(2 * ch)

        # (方案B) 直接预测新的 Fri/Bri
        self.direct_head_f = FreqDirectHeadFiLM(ch=ch, time_ch=time_ch)
        self.direct_head_b = FreqDirectHeadFiLM(ch=ch, time_ch=time_ch)

    @staticmethod
    def _to_ri(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def _to_complex(ri: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = ri.shape
        c = c2 // 2
        return torch.complex(ri[:, :c], ri[:, c:])

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) FFT -> complex
        Fz = torch.fft.fftshift(torch.fft.fft2(fore, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(back, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        # 2) complex -> ri
        Fri = self._to_ri(Fz)  # (B,2C,H,W)
        Bri = self._to_ri(Bz)
        Cri = self._to_ri(Cz)

        # 3) concat with cond as KV signal
        f_in = torch.cat([Fri, Cri], dim=1)  # (B,4C,H,W)
        b_in = torch.cat([Bri, Cri], dim=1)

        # 4) (方案B) 直接得到新的 Fri/Bri（不再mask乘法）
        Fri = self.direct_head_f(f_in, t_emb)  # (B,2C,H,W)
        Bri = self.direct_head_b(b_in, t_emb)  # (B,2C,H,W)

        # 5) Tri attention interaction (fb, fc, bc)
        d_fb = self.attn_fb(Fri, Bri)
        d_fc = self.attn_fc(Fri, Cri)
        Fri2 = self.norm_f(self.post_f(Fri + d_fb + d_fc))

        d_bc = self.attn_bc(Bri, Cri)
        Bri2 = self.norm_b(self.post_b(Bri + d_bc))

        # 6) ri -> complex -> IFFT back to spatial
        F2c = self._to_complex(Fri2)
        B2c = self._to_complex(Bri2)

        F2c = torch.fft.ifftshift(F2c, dim=(-2, -1))
        B2c = torch.fft.ifftshift(B2c, dim=(-2, -1))

        fore_out = torch.fft.ifft2(F2c, dim=(-2, -1), norm="ortho").real
        back_out = torch.fft.ifft2(B2c, dim=(-2, -1), norm="ortho").real

        # cond 不更新
        return fore_out, back_out, cond


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
# Cond decoder stage (U-Net style)
# -----------------------------
class CondDecoderStage(nn.Module):
    """
    cond decoder stage:
    upsample -> add cond encoder skip -> ResBlocksFiLM
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        time_ch: int,
        num_blocks: int,
        up: bool,
        dropout: float,
    ):
        super().__init__()
        self.up = Upsample(in_ch) if up else None
        self.skip_to_x = nn.Identity() if skip_ch == in_ch else nn.Conv2d(skip_ch, in_ch, 1, bias=True)

        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_blocks):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)
        x = x + self.skip_to_x(skip)
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# -----------------------------
# Decoder stage: direct add skip_fb + cond_feat (from cond decoder)
# -----------------------------
class DecoderStageCondAddDirect(nn.Module):
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
    ):
        super().__init__()
        self.up = Upsample(in_ch) if up else None

        self.skip_to_x = nn.Identity() if skip_ch == in_ch else nn.Conv2d(skip_ch, in_ch, 1, bias=True)
        self.cond_to_x = nn.Identity() if cond_ch == in_ch else nn.Conv2d(cond_ch, in_ch, 1, bias=True)

        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_blocks):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.fuse_norm = GroupNorm32(in_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond_skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)

        x = x + self.skip_to_x(skip) + self.cond_to_x(cond_skip)
        x = F.silu(self.fuse_norm(x))

        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# -----------------------------
# Main Network
# -----------------------------
class FMForeBackCondDeepKTriFFTSingleDecoderResUNet(nn.Module):
    def __init__(
        self,
        *,
        image_hw: Tuple[int, int],
        img_in_ch: int,
        xt_in_ch: int,
        base_ch: int,
        ch_mult: Tuple[int, ...],
        num_res_blocks: int,
        time_dim: int,
        dropout: float,
        attn_heads: int,
        attn_head_dim: int,
        freq_attn_depth: int,
        use_bottleneck_coupling: bool,
        fa_stacks: int,
        hp_inject_depth: int,
        hp_sigma_min: float,
        hp_sigma_max: float,
        hp_sigma_init: float,
        noise_strength: float = 1.0,
    ):
        super().__init__()

        if xt_in_ch < 2:
            raise ValueError("xt_in_ch must be >=2 (channel0 background, remaining foreground).")
        if freq_attn_depth < 1:
            raise ValueError("freq_attn_depth must be >= 1.")
        if fa_stacks < 1:
            raise ValueError("fa_stacks must be >= 1")

        self.image_hw = tuple(image_hw)
        self.img_in_ch = int(img_in_ch)
        self.xt_in_ch = int(xt_in_ch)
        self.base_ch = int(base_ch)
        self.ch_mult = tuple(int(x) for x in ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.time_dim = int(time_dim)
        self.dropout = float(dropout)
        self.attn_heads = int(attn_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.freq_attn_depth = int(freq_attn_depth)
        self.use_bottleneck_coupling = bool(use_bottleneck_coupling)
        self.fa_stacks = int(fa_stacks)
        self.hp_inject_depth = int(hp_inject_depth)
        self.hp_sigma_min = float(hp_sigma_min)
        self.hp_sigma_max = float(hp_sigma_max)
        self.hp_sigma_init = float(hp_sigma_init)
        self.noise_strength = float(noise_strength)

        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

        back_in_ch = 1
        fore_in_ch = self.xt_in_ch - 1
        cond_in_ch = self.img_in_ch

        self.back_stem = nn.Conv2d(back_in_ch, self.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(fore_in_ch, self.base_ch, 3, padding=1)
        self.cond_stem = nn.Conv2d(cond_in_ch, self.base_ch, 3, padding=1)

        self.chs = [self.base_ch * m for m in self.ch_mult]
        n_levels = len(self.chs)

        # encoders
        self.back_enc = nn.ModuleList()
        self.fore_enc = nn.ModuleList()
        self.cond_enc = nn.ModuleList()
        for i in range(n_levels):
            in_ch = self.chs[i - 1] if i > 0 else self.base_ch
            out_ch = self.chs[i]
            down = i < (n_levels - 1)
            self.back_enc.append(EncoderStageFiLM(in_ch, out_ch, self.time_dim, self.num_res_blocks, down, self.dropout))
            self.fore_enc.append(EncoderStageFiLM(in_ch, out_ch, self.time_dim, self.num_res_blocks, down, self.dropout))
            self.cond_enc.append(EncoderStageFiLM(in_ch, out_ch, self.time_dim, self.num_res_blocks, down, self.dropout))

        # FA levels: last K
        K = min(self.freq_attn_depth, n_levels)
        self.attn_level_set = set(range(n_levels - K, n_levels))

        def make_fa_list(ch: int) -> nn.ModuleList:
            mods = nn.ModuleList()
            for _ in range(self.fa_stacks):
                mods.append(
                    FAModule(
                        ch=ch,
                        time_ch=self.time_dim,
                        heads=self.attn_heads,
                        head_dim=self.attn_head_dim,
                        noise_strength=self.noise_strength,
                    )
                )
            return mods

        self.couple_modules = nn.ModuleDict()
        for i in sorted(self.attn_level_set):
            self.couple_modules[str(i)] = make_fa_list(self.chs[i])

        self.bottleneck_coupler = make_fa_list(self.chs[-1]) if self.use_bottleneck_coupling else None

        # HP inject levels
        K_hp = min(self.hp_inject_depth, n_levels)
        self.hp_level_set = set(range(0, K_hp))

        self.cond_hp_ops = nn.ModuleDict()
        self.cond_gate_fore = nn.ModuleDict()
        self.cond_gate_back = nn.ModuleDict()
        for i in sorted(self.hp_level_set):
            ch = self.chs[i]
            self.cond_hp_ops[str(i)] = CondGaussianHighPassOperator(
                ch=ch,
                time_ch=self.time_dim,
                sigma_min=self.hp_sigma_min,
                sigma_max=self.hp_sigma_max,
                sigma_init=self.hp_sigma_init,
            )
            self.cond_gate_fore[str(i)] = CondGateResidual(x_ch=ch, cond_ch=ch)
            self.cond_gate_back[str(i)] = CondGateResidual(x_ch=ch, cond_ch=ch)

        # cond decoder (mirror of cond encoder)
        self.cond_dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            in_ch = self.chs[i] if i == (n_levels - 1) else self.chs[i + 1]
            out_ch = self.chs[i]
            skip_ch = self.chs[i]
            up = i < (n_levels - 1)
            self.cond_dec.append(
                CondDecoderStage(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    time_ch=self.time_dim,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                )
            )

        # SINGLE shared decoder after concat(back, fore)
        self.dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = 2 * self.chs[i]
            cond_ch = self.chs[i]

            in_ch = (2 * self.chs[i + 1]) if i < (n_levels - 1) else (2 * self.chs[i])
            out_ch = 2 * self.chs[i]
            up = i < (n_levels - 1)

            self.dec.append(
                DecoderStageCondAddDirect(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    cond_ch=cond_ch,
                    out_ch=out_ch,
                    time_ch=self.time_dim,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                )
            )

        self.out_v = nn.Conv2d(2 * self.base_ch, self.xt_in_ch, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        image: torch.Tensor,
        t: Union[torch.Tensor, float],
    ) -> torch.Tensor:
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

        fore_skips: List[torch.Tensor] = []
        back_skips: List[torch.Tensor] = []
        cond_skips: List[torch.Tensor] = []

        # -----------------------------
        # Encoder (3 branches)
        # -----------------------------
        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            back, b_skip = enc_b(back, t_emb)
            fore, f_skip = enc_f(fore, t_emb)
            cond, c1 = enc_c(cond, t_emb)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c1)

            b_main, f_main, c_main = b_skip, f_skip, c1

            # FA uses cond encoder skip as KV (cond not updated)
            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_main, b_main, _ = fa(f_main, b_main, c_main, t_emb=t_emb)

            # HP inject (uses RAW c1)
            if level in self.hp_level_set:
                hp_op = self.cond_hp_ops[str(level)]
                gate_f = self.cond_gate_fore[str(level)]
                gate_b = self.cond_gate_back[str(level)]

                c2 = hp_op(c1, t_emb)
                f_main = gate_f(f_main, c2)
                b_main = gate_b(b_main, c2)

            back = enc_b.down(b_main) if enc_b.down is not None else b_main
            fore = enc_f.down(f_main) if enc_f.down is not None else f_main
            cond = enc_c.down(c_main) if enc_c.down is not None else c_main

        # bottleneck coupling
        if self.bottleneck_coupler is not None:
            for fa in self.bottleneck_coupler:
                fore, back, _ = fa(fore, back, cond, t_emb=t_emb)

        # -----------------------------
        # Cond Decoder (mirror U-Net)
        # -----------------------------
        cond_dec_feats: List[torch.Tensor] = []
        h_cond = cond
        for dec_c, c_skip in zip(self.cond_dec, reversed(cond_skips)):
            h_cond = dec_c(h_cond, c_skip, t_emb)
            cond_dec_feats.append(h_cond)

        # -----------------------------
        # Shared Decoder (uses cond_dec_feats)
        # -----------------------------
        h = torch.cat([back, fore], dim=1)

        for dec, f_skip, b_skip, c_dec in zip(
            self.dec,
            reversed(fore_skips),
            reversed(back_skips),
            cond_dec_feats,
        ):
            skip_fb = torch.cat([b_skip, f_skip], dim=1)
            h = dec(h, skip_fb, c_dec, t_emb)

        h = F.silu(h)
        v = self.out_v(h)
        return v


# -----------------------------
# Factory
# -----------------------------
def build_model(
    image_hw: Tuple[int, int] = (256, 256),
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
    use_bottleneck_coupling: bool = True,
    fa_stacks: int = 1,
    hp_inject_depth: int = 1,
    hp_sigma_min: float = 0.05,
    hp_sigma_max: float = 0.50,
    hp_sigma_init: float = 0.15,
    noise_strength: float = 1.0,
) -> FMForeBackCondDeepKTriFFTSingleDecoderResUNet:
    return FMForeBackCondDeepKTriFFTSingleDecoderResUNet(
        image_hw=tuple(image_hw),
        img_in_ch=img_in_ch,
        xt_in_ch=xt_in_ch,
        base_ch=base_ch,
        ch_mult=tuple(int(x) for x in ch_mult),
        num_res_blocks=num_res_blocks,
        time_dim=time_dim,
        dropout=dropout,
        attn_heads=attn_heads,
        attn_head_dim=attn_head_dim,
        freq_attn_depth=freq_attn_depth,
        use_bottleneck_coupling=use_bottleneck_coupling,
        fa_stacks=fa_stacks,
        hp_inject_depth=hp_inject_depth,
        hp_sigma_min=hp_sigma_min,
        hp_sigma_max=hp_sigma_max,
        hp_sigma_init=hp_sigma_init,
        noise_strength=noise_strength,
    )


if __name__ == "__main__":
    net = build_model(
        image_hw=(256, 256),
        img_in_ch=1,
        xt_in_ch=4,
        base_ch=32,
        ch_mult=(1, 2, 4, 4),
        freq_attn_depth=2,
        fa_stacks=2,
        use_bottleneck_coupling=True,
        hp_inject_depth=1,
        noise_strength=1.0,
    )

    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v = net(x_t, img, t)
    print("v:", v.shape)  # (2, 4, 256, 256)