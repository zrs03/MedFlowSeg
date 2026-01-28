# -*- coding: utf-8 -*-
"""
✅ Keep TriFFT attention (frequency-domain cross-attention) for FA.
✅ REMOVE original high-frequency gate inside TriFFT attention.

✅ Keep: Per-level Cond High-pass + Gated Residual Injection (NO concat)
    1) cond encoder produces c1 (RAW, used for skip)
    2) c1 -> HP operator -> c2 (per-level params, NOT shared)
    3) inject into fore/back by gated residual: x' = sigmoid(Conv1x1(c2)) * x + x
    4) downsample fore/back using injected x', and downsample cond using c1 (RAW)

✅ Step1 cond FFT filter (Scheme-B UPDATED):
    - Bases Br/Bi are generated FROM cond by conv (dynamic, per-sample), NO interpolation
    - Mixing weights a are generated FROM BOTH (cond, t_emb)
    - Compose Wr/Wi and complex-multiply on Cz (fftshift'ed)

✅ Change requested:
    - REMOVE last2_levels restriction
    - If a level has FA module, it may use modulator
    - Use freq_mod_rank_list instead of single freq_mod_rank:
        * freq_attn_depth=1  -> freq_mod_rank_list=(4,)
        * freq_attn_depth=2  -> freq_mod_rank_list=(4,8)
      Mapping:
        - FA levels are the last K levels: sorted(attn_level_set)
        - freq_mod_rank_list[i] corresponds to i-th FA level in ascending order
        - If list shorter than K: pad using the last value
        - If list longer than K: extra ignored

✅ Injection HP operator: Gaussian high-pass (FFT*g_hp -> IFFT), time-conditioned per-channel sigma

✅ Decoder (NO concat):
    - g = sigmoid(Conv1x1(cond_skip)), skip' = skip * g
    - x = x + proj_skip(skip') + proj_cond(cond_skip)
    - then ResBlocks on x only

✅ FAModule = step1(update cond only, optional cond FFT filter) + step2(update fore/back only, no filter)

✅ Requested now:
    - 删除 cfg/Config，参数全部写在 build_model

✅ THIS CHANGE:
    - 把原本删掉的 Norm 加回来：
        * SpatialCrossAttention 的 norm_q / norm_kv
        * FAModule 的 post+norm（post 用 GN+SiLU+Conv；再接一个 GN）
    - 其他不改
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

        # ✅ restore original norms
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
# Gaussian HP gate (time-conditioned per-channel sigma)
# g_hp(r,t,c) = 1 - exp( -r^2 / (2*sigma(t,c)^2) )
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
        r = r[None, None, :, :]  # (1,1,H,W)

        self._grid_cache[key] = r
        return r

    def forward(self, t_emb: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b = t_emb.shape[0]
        r = self._radius_grid(h, w, t_emb.device, t_emb.dtype)  # (1,1,H,W)

        sigma = torch.sigmoid(self.to_sigma(t_emb))  # (B,C)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma
        sigma = sigma.clamp_min(self.eps).view(b, self.ch, 1, 1)  # (B,C,1,1)

        g_hp = 1.0 - torch.exp(-(r * r) / (2.0 * sigma * sigma))  # (B,C,H,W)
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
        g_hp = self.gate(t_emb, h, w).to(Cz.dtype)  # real -> complex dtype
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
# Step1 Cond FFT Filter (Scheme-B)
# -----------------------------
class CondTimeDynamicBasisComplexFreqModulator(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        hw: Tuple[int, int],
        rank_r: int = 4,
        cond_hidden: int = 64,
        basis_hidden: int = 128,
    ):
        super().__init__()
        self.ch = int(ch)
        self.h, self.w = int(hw[0]), int(hw[1])
        self.rank_r = int(rank_r)

        if self.h < 2 or self.w < 2:
            raise ValueError("hw must be >= (2,2)")
        if self.rank_r < 1:
            raise ValueError("rank_r must be >= 1")

        self.basis_net = nn.Sequential(
            nn.Conv2d(self.ch, basis_hidden, 3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(basis_hidden, basis_hidden, 3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(basis_hidden, 2 * self.rank_r * self.ch, 1, bias=True),
        )
        nn.init.zeros_(self.basis_net[-1].weight)
        nn.init.zeros_(self.basis_net[-1].bias)

        self.cond_pool_net = nn.Sequential(
            nn.Conv2d(self.ch, cond_hidden, 3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(cond_hidden, cond_hidden, 3, padding=1, bias=True),
            nn.SiLU(),
        )
        self.cond_to_u = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))

        a_hidden = max(32, 2 * self.rank_r)
        self.to_a = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch + cond_hidden, a_hidden),
            nn.SiLU(),
            nn.Linear(a_hidden, self.rank_r),
        )
        nn.init.zeros_(self.to_a[-1].weight)
        nn.init.zeros_(self.to_a[-1].bias)

    def apply(self, Cz: torch.Tensor, cond_feat: torch.Tensor, t_emb: Optional[torch.Tensor]) -> torch.Tensor:
        b, c, h, w = Cz.shape
        if c != self.ch:
            raise ValueError(f"Modulator expected C={self.ch}, got {c}")
        if h != self.h or w != self.w:
            raise ValueError(
                f"Modulator expected (H,W)=({self.h},{self.w}), got ({h},{w}). "
                "NO interpolation; keep level resolution fixed."
            )
        if t_emb is None:
            return Cz
        if cond_feat.shape != (b, self.ch, self.h, self.w):
            raise ValueError(f"cond_feat must be (B,{self.ch},{self.h},{self.w}), got {tuple(cond_feat.shape)}")

        bases = self.basis_net(cond_feat)  # (B, 2*R*C, H, W)
        bases = bases.view(b, 2, self.rank_r, self.ch, self.h, self.w)
        Br = bases[:, 0]  # (B,R,C,H,W)
        Bi = bases[:, 1]  # (B,R,C,H,W)

        u = self.cond_to_u(self.cond_pool_net(cond_feat))  # (B,cond_hidden)
        a = self.to_a(torch.cat([t_emb, u], dim=1))        # (B,R)

        Wr = 1.0 + torch.einsum("br,brchw->bchw", a, Br)
        Wi = torch.einsum("br,brchw->bchw", a, Bi)

        Cr, Ci = Cz.real, Cz.imag
        Cr2 = Cr * Wr - Ci * Wi
        Ci2 = Cr * Wi + Ci * Wr
        return torch.complex(Cr2, Ci2)


# -----------------------------
# FA module  ✅ restore post+norm blocks
# -----------------------------
class FAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        cond_filter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ch = ch
        self.cond_filter = cond_filter

        # step1
        self.s1_attn_fb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s1_attn_fc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s1_attn_bc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)

        self.s1_post_f = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.s1_post_b = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.s1_norm_f = GroupNorm32(2 * ch)
        self.s1_norm_b = GroupNorm32(2 * ch)

        self.s1_attn_cf = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s1_attn_cb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s1_post_c = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.s1_norm_c = GroupNorm32(2 * ch)

        # step2
        self.s2_attn_fb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s2_attn_fc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s2_attn_bc = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)

        self.s2_post_f = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.s2_post_b = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.s2_norm_f = GroupNorm32(2 * ch)
        self.s2_norm_b = GroupNorm32(2 * ch)

        self.s2_attn_cf = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s2_attn_cb = SpatialCrossAttention(q_ch=2 * ch, kv_ch=2 * ch, heads=heads, head_dim=head_dim)
        self.s2_post_c = nn.Sequential(GroupNorm32(2 * ch), nn.SiLU(), nn.Conv2d(2 * ch, 2 * ch, 1))
        self.s2_norm_c = GroupNorm32(2 * ch)

    @staticmethod
    def _to_ri(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def _to_complex(ri: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = ri.shape
        c = c2 // 2
        return torch.complex(ri[:, :c], ri[:, c:])

    def _apply_cond_filter(
        self,
        Cz: torch.Tensor,
        cond_feat: torch.Tensor,
        t_emb: Optional[torch.Tensor],
        filt: Optional[nn.Module],
    ) -> torch.Tensor:
        if filt is None:
            return Cz
        return filt.apply(Cz, cond_feat, t_emb)

    def _tri_fft_pass(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: Optional[torch.Tensor],
        update_fb: bool,
        update_c: bool,
        cond_fft_filter: Optional[nn.Module],
        attn_fb: SpatialCrossAttention,
        attn_fc: SpatialCrossAttention,
        attn_bc: SpatialCrossAttention,
        post_f: nn.Module,
        post_b: nn.Module,
        norm_f: nn.Module,
        norm_b: nn.Module,
        attn_cf: SpatialCrossAttention,
        attn_cb: SpatialCrossAttention,
        post_c: nn.Module,
        norm_c: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Fz = torch.fft.fftshift(torch.fft.fft2(fore, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(back, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        Cz = self._apply_cond_filter(Cz, cond, t_emb, cond_fft_filter)

        Fri = self._to_ri(Fz)
        Bri = self._to_ri(Bz)
        Cri = self._to_ri(Cz)

        if update_fb:
            d_fb = attn_fb(Fri, Bri)
            d_fc = attn_fc(Fri, Cri)
            Fri2 = norm_f(post_f(Fri + d_fb + d_fc))

            d_bc = attn_bc(Bri, Cri)
            Bri2 = norm_b(post_b(Bri + d_bc))
        else:
            Fri2, Bri2 = Fri, Bri

        if update_c:
            d_cf = attn_cf(Cri, Fri2 if update_fb else Fri)
            d_cb = attn_cb(Cri, Bri2 if update_fb else Bri)
            Cri2 = norm_c(post_c(Cri + d_cf + d_cb))
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

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # step1: update cond only (optional cond FFT filter)
        fore, back, cond = self._tri_fft_pass(
            fore, back, cond, t_emb,
            update_fb=False, update_c=True,
            cond_fft_filter=self.cond_filter,
            attn_fb=self.s1_attn_fb, attn_fc=self.s1_attn_fc, attn_bc=self.s1_attn_bc,
            post_f=self.s1_post_f, post_b=self.s1_post_b, norm_f=self.s1_norm_f, norm_b=self.s1_norm_b,
            attn_cf=self.s1_attn_cf, attn_cb=self.s1_attn_cb, post_c=self.s1_post_c, norm_c=self.s1_norm_c,
        )

        # step2: update fore/back only (no filter)
        fore, back, cond = self._tri_fft_pass(
            fore, back, cond, t_emb,
            update_fb=True, update_c=False,
            cond_fft_filter=None,
            attn_fb=self.s2_attn_fb, attn_fc=self.s2_attn_fc, attn_bc=self.s2_attn_bc,
            post_f=self.s2_post_f, post_b=self.s2_post_b, norm_f=self.s2_norm_f, norm_b=self.s2_norm_b,
            attn_cf=self.s2_attn_cf, attn_cb=self.s2_attn_cb, post_c=self.s2_post_c, norm_c=self.s2_norm_c,
        )
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
# Decoder stage (NO concat, NO fuse_norm)  (unchanged)
# -----------------------------
class DecoderStageCondAdd(nn.Module):
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
    ):
        super().__init__()
        self.up = Upsample(in_ch) if up else None

        self.use_attn_gate = use_attn_gate
        self.gate = AttentionGate(x_ch=in_ch, skip_ch=skip_ch) if use_attn_gate else None

        self.cond_to_skip_gate = nn.Conv2d(cond_ch, skip_ch, 1, bias=True)

        self.skip_to_x = nn.Identity() if skip_ch == in_ch else nn.Conv2d(skip_ch, in_ch, 1, bias=True)
        self.cond_to_x = nn.Identity() if cond_ch == in_ch else nn.Conv2d(cond_ch, in_ch, 1, bias=True)

        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_blocks):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond_skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)

        if self.use_attn_gate and self.gate is not None:
            skip = self.gate(x, skip)

        g = torch.sigmoid(self.cond_to_skip_gate(cond_skip))  # (B,skip_ch,H,W)
        skip_p = skip * g

        x = x + self.skip_to_x(skip_p) + self.cond_to_x(cond_skip)

        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# -----------------------------
# Main Network (NO cfg)
# -----------------------------
class FMForeBackCondDeepKTriFFTDualDecoderResUNet(nn.Module):
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
        use_step1_cond_freq_mod: bool,
        freq_mod_rank_list: Optional[Tuple[int, ...]],
        use_attn_gated_skip: bool,
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
        self.use_step1_cond_freq_mod = bool(use_step1_cond_freq_mod)
        self.freq_mod_rank_list = None if freq_mod_rank_list is None else tuple(int(x) for x in freq_mod_rank_list)
        self.use_attn_gated_skip = bool(use_attn_gated_skip)

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
        fa_levels_sorted = sorted(self.attn_level_set)

        H0, W0 = self.image_hw

        def level_hw(level: int) -> Tuple[int, int]:
            h = H0 // (2 ** level)
            w = W0 // (2 ** level)
            if h < 2 or w < 2:
                raise ValueError(f"level_hw too small at level={level}: ({h},{w}) from image_hw={self.image_hw}")
            return (h, w)

        def rank_for_fa_level(level: int) -> Optional[int]:
            if not self.use_step1_cond_freq_mod:
                return None
            if self.freq_mod_rank_list is None:
                return None
            ranks = list(self.freq_mod_rank_list)
            if len(ranks) == 0:
                return None
            idx = fa_levels_sorted.index(level)
            if idx >= len(ranks):
                return int(ranks[-1])
            return int(ranks[idx])

        def make_fa_list(level: int, ch: int) -> nn.ModuleList:
            mods = nn.ModuleList()

            cond_filter = None
            rank_r = rank_for_fa_level(level)
            if rank_r is not None and rank_r > 0:
                hL, wL = level_hw(level)
                cond_filter = CondTimeDynamicBasisComplexFreqModulator(
                    ch=ch, time_ch=self.time_dim, hw=(hL, wL), rank_r=rank_r
                )

            for _ in range(self.fa_stacks):
                mods.append(
                    FAModule(
                        ch=ch,
                        time_ch=self.time_dim,
                        heads=self.attn_heads,
                        head_dim=self.attn_head_dim,
                        cond_filter=cond_filter,
                    )
                )
            return mods

        self.couple_modules = nn.ModuleDict()
        for i in fa_levels_sorted:
            self.couple_modules[str(i)] = make_fa_list(i, self.chs[i])

        self.bottleneck_coupler = make_fa_list(n_levels - 1, self.chs[-1]) if self.use_bottleneck_coupling else None

        # HP inject levels: first K_hp levels
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
                DecoderStageCondAdd(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    cond_ch=cond_ch,
                    out_ch=out_ch,
                    time_ch=self.time_dim,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                    use_attn_gate=self.use_attn_gated_skip,
                )
            )
            self.fore_dec.append(
                DecoderStageCondAdd(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    cond_ch=cond_ch,
                    out_ch=out_ch,
                    time_ch=self.time_dim,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                    use_attn_gate=self.use_attn_gated_skip,
                )
            )

        self.out_conv = nn.Conv2d(self.base_ch * 2, self.xt_in_ch, 3, padding=1)

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

        back_skips: List[torch.Tensor] = []
        fore_skips: List[torch.Tensor] = []
        cond_skips: List[torch.Tensor] = []

        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            back, b_skip = enc_b(back, t_emb)
            fore, f_skip = enc_f(fore, t_emb)
            cond, c1 = enc_c(cond, t_emb)  # RAW cond skip

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c1)

            b_main, f_main, c_main = b_skip, f_skip, c1

            # FA
            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                c_tmp = c_main
                for fa in fa_list:
                    f_main, b_main, c_tmp = fa(f_main, b_main, c_tmp, t_emb=t_emb)

            # HP inject
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
            c_tmp = cond
            for fa in self.bottleneck_coupler:
                fore, back, c_tmp = fa(fore, back, c_tmp, t_emb=t_emb)
            cond = c_tmp

        # Decoder
        back_h = back
        fore_h = fore
        for dec_b, dec_f, b_skip, f_skip, c_skip in zip(
            self.back_dec, self.fore_dec, reversed(back_skips), reversed(fore_skips), reversed(cond_skips)
        ):
            back_h = dec_b(back_h, b_skip, c_skip, t_emb)
            fore_h = dec_f(fore_h, f_skip, c_skip, t_emb)

        h = torch.cat([fore_h, back_h], dim=1)
        h = F.silu(h)
        v = self.out_conv(h)
        return v


# -----------------------------
# Factory (all params here)
# -----------------------------
def build_model(
    image_hw: Tuple[int, int] = (256, 256),
    img_in_ch: int = 1,
    xt_in_ch: int = 4,
    base_ch: int = 64,
    ch_mult: Sequence[int] = (1, 2, 4, 4, 8),
    num_res_blocks: int = 2,
    time_dim: int = 256,
    dropout: float = 0.0,
    attn_heads: int = 4,
    attn_head_dim: int = 32,
    freq_attn_depth: int = 2,
    use_bottleneck_coupling: bool = True,
    fa_stacks: int = 1,
    hp_inject_depth: int = 1,
    hp_sigma_min: float = 0.05,
    hp_sigma_max: float = 0.50,
    hp_sigma_init: float = 0.15,
    use_step1_cond_freq_mod: bool = True,
    freq_mod_rank_list: Optional[Sequence[int]] = (8, 16),
    use_attn_gated_skip: bool = True,
) -> FMForeBackCondDeepKTriFFTDualDecoderResUNet:
    return FMForeBackCondDeepKTriFFTDualDecoderResUNet(
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
        use_step1_cond_freq_mod=use_step1_cond_freq_mod,
        freq_mod_rank_list=None if freq_mod_rank_list is None else tuple(int(x) for x in freq_mod_rank_list),
        use_attn_gated_skip=use_attn_gated_skip,
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
        use_step1_cond_freq_mod=True,
        freq_mod_rank_list=(4, 8),
        use_attn_gated_skip=True,
    )
    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v = net(x_t, img, t)
    print("v:", v.shape)