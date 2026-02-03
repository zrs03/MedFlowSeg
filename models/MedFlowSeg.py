# -*- coding: utf-8 -*-
"""
✅ FA: one-pass (no step1), fore/back updated using cond as KV
✅ REMOVE Modulator entirely (NO rank / NO basis decomposition)
✅ Keep frequency-domain attention ops:
    - (方案B) direct heads predict new Fri/Bri (no mask multiply)
    - TriFFT attention updates fore/back using cond as KV

✅ REMOVE: all HP inject related code/params

✅ NEW: Scheme-1 HOG interaction (DIRECT use, no hog encoder, no hog->FA injection)
    - Compute dense freq-HOG on raw split x_back/x_fore:
        FFT -> phase(phi) + magnitude(mag)
        soft-bin phase into hog_bins -> p: (B,C,H,W,K)
        (optional) magnitude weighting; then normalize across K to get distribution
    - Compute similarity sim_c(h,w) between each fore-channel HOG distribution and back HOG distribution
        sim_c: (B,Cf,H,W)
    - Use SOFTMAX over channels to aggregate fore->back (NO topk):
        w = softmax(sim_c / tau)  over channel dim
        alpha_back = sum_c w_c * alpha_c
        F_mix = sum_c w_c * Fz_c
    - Use alpha_c for per-fore-channel routing; use alpha_back for back routing
    - IFFT back to spatial -> x_back2/x_fore2 -> then proceed as usual

✅ 方案E+（逐通道、cond+t）:
    - 用 t_emb + cond_emb 生成每个 fore 通道的每个 bin 权重 w_{c,k}(t,cond)
    - cond_emb 来自 raw image 的轻量 summary（Conv -> GAP -> (B,time_dim)）
    - wk = softmax_k( Linear([t_emb,cond_emb]) -> (B, Cf*K) -> reshape (B,Cf,K) )
    - sim_c = sum_k pf_c * pb * wk_{c,k}
    - 后续 routing 不变
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Time embedding + FiLM
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

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
# Spatial Cross Attention (multi-head) ✅ with Norm
# -----------------------------
class SpatialCrossAttention(nn.Module):
    def __init__(self, q_ch: int, kv_ch: int, heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        inner = self.heads * self.head_dim

        self.to_q = nn.Conv2d(q_ch, inner, 1, bias=False)
        self.to_k = nn.Conv2d(kv_ch, inner, 1, bias=False)
        self.to_v = nn.Conv2d(kv_ch, inner, 1, bias=False)

        self.out = nn.Conv2d(inner, q_ch, 1)
        self.scale = self.head_dim**-0.5

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
# Dense frequency-domain "HOG" (NO gradient direction) + magnitude
# We return hog_map (B, C*K, H, W); later we reshape back to (B,C,H,W,K).
# -----------------------------
def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class FreqDenseHOG(nn.Module):
    def __init__(
        self,
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        eps: float = 1e-8,
        mag_weight: bool = True,
    ):
        super().__init__()
        if hog_bins < 2:
            raise ValueError("hog_bins must be >= 2")
        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.eps = float(eps)
        self.mag_weight = bool(mag_weight)

        k = torch.arange(self.hog_bins).float()
        theta = -math.pi + (2.0 * math.pi) * k / float(self.hog_bins)  # (K,)
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) real
        return: hog_map (B, C*hog_bins, H, W)
        """
        b, c, h, w = x.shape

        X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        mag = X.abs().clamp_min(self.eps)  # (B,C,H,W)
        phi = torch.angle(X)               # (B,C,H,W)

        theta = self.theta.view(1, 1, 1, 1, self.hog_bins).to(device=x.device, dtype=phi.dtype)
        delta = _wrap_pi(phi.unsqueeze(-1) - theta)  # (B,C,H,W,K)

        logits = torch.cos(delta) / max(self.hog_tau, 1e-6)
        p = F.softmax(logits, dim=-1)  # (B,C,H,W,K)

        if self.mag_weight:
            p = p * mag.unsqueeze(-1)

        p = p.permute(0, 1, 4, 2, 3).contiguous()     # (B,C,K,H,W)
        hog_map = p.view(b, c * self.hog_bins, h, w)  # (B,C*K,H,W)
        return hog_map


# -----------------------------
# (方案B) Frequency-domain direct head: from (Fri,Cri)->Fri'  (FiLM conditioned on time)
# 输入: (B,4C,H,W) 输出: (B,2C,H,W)
# -----------------------------
class FreqDirectHeadFiLM(nn.Module):
    def __init__(self, ch: int, time_ch: int, hidden_mult: int = 2):
        super().__init__()
        self.ch = int(ch)
        hid = int(hidden_mult * ch)

        self.in_norm = GroupNorm32(4 * ch)
        self.conv_in = nn.Conv2d(4 * ch, hid, 3, padding=1)

        self.hid_norm = GroupNorm32(hid)
        self.to_film = nn.Sequential(nn.SiLU(), nn.Linear(time_ch, 2 * hid))

        self.conv_mid = nn.Conv2d(hid, hid, 3, padding=1)
        self.conv_out = nn.Conv2d(hid, 2 * ch, 1)

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
# (cond not updated here; it is used as KV only)
# -----------------------------
class FAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        noise_strength: float = 1.0,  # kept for compatibility, unused in方案B
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
        Fz = torch.fft.fftshift(torch.fft.fft2(fore, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(back, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        Fri = self._to_ri(Fz)
        Bri = self._to_ri(Bz)
        Cri = self._to_ri(Cz)

        f_in = torch.cat([Fri, Cri], dim=1)
        b_in = torch.cat([Bri, Cri], dim=1)

        Fri = self.direct_head_f(f_in, t_emb)
        Bri = self.direct_head_b(b_in, t_emb)

        d_fb = self.attn_fb(Fri, Bri)
        d_fc = self.attn_fc(Fri, Cri)
        Fri2 = self.norm_f(self.post_f(Fri + d_fb + d_fc))

        d_bc = self.attn_bc(Bri, Cri)
        Bri2 = self.norm_b(self.post_b(Bri + d_bc))

        F2c = self._to_complex(Fri2)
        B2c = self._to_complex(Bri2)

        F2c = torch.fft.ifftshift(F2c, dim=(-2, -1))
        B2c = torch.fft.ifftshift(B2c, dim=(-2, -1))

        fore_out = torch.fft.ifft2(F2c, dim=(-2, -1), norm="ortho").real
        back_out = torch.fft.ifft2(B2c, dim=(-2, -1), norm="ortho").real

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
        noise_strength: float = 1.0,
        # --- HOG routing params ---
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        hog_mag_weight: bool = True,
        hog_eps: float = 1e-8,
        hog_softmax_tau: float = 0.10,  # softmax 温度：小更尖(像max)，大更平(像mean)
        # --- E+ head ---
        hog_bin_weight_hidden: int = 0,  # 0=linear; >0=MLP
        # --- optional: init wk uniform ---
        init_wk_uniform: bool = True,
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
        self.noise_strength = float(noise_strength)

        # HOG routing config
        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.hog_eps = float(hog_eps)
        self.hog_softmax_tau = float(hog_softmax_tau)

        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

        back_in_ch = 1
        self.fore_in_ch = self.xt_in_ch - 1
        cond_in_ch = self.img_in_ch

        self.back_stem = nn.Conv2d(back_in_ch, self.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(self.fore_in_ch, self.base_ch, 3, padding=1)
        self.cond_stem = nn.Conv2d(cond_in_ch, self.base_ch, 3, padding=1)

        # ✅ HOG extractor used DIRECTLY (no hog encoder)
        self.freq_hog = FreqDenseHOG(
            hog_bins=self.hog_bins,
            hog_tau=self.hog_tau,
            eps=self.hog_eps,
            mag_weight=hog_mag_weight,
        )

        # ✅ E+: cond summary (from raw image) -> (B,time_dim)
        self.cond_summary = nn.Sequential(
            nn.Conv2d(self.img_in_ch, self.time_dim, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # (B,time_dim)
        )

        # ✅ E+: per-fore-channel per-bin weights from [t_emb, cond_emb]
        out_dim = self.fore_in_ch * self.hog_bins
        in_dim = 2 * self.time_dim
        if hog_bin_weight_hidden and hog_bin_weight_hidden > 0:
            self.hog_bin_weight_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, int(hog_bin_weight_hidden)),
                nn.SiLU(),
                nn.Linear(int(hog_bin_weight_hidden), out_dim),
            )
        else:
            self.hog_bin_weight_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_dim, out_dim),
            )

        # optional: start from uniform wk (stable)
        if init_wk_uniform:
            last = self.hog_bin_weight_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

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

    # -----------------------------
    # Scheme-1 HOG routing: direct fore/back interaction BEFORE stems
    # -----------------------------
    def _hog_routing_fuse(
        self,
        x_back: torch.Tensor,
        x_fore: torch.Tensor,
        t_emb: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_back: (B,1,H,W)  real
        x_fore: (B,Cf,H,W) real
        t_emb : (B,time_dim)
        cond_emb: (B,time_dim) from raw image summary

        E+:
            wk_{c,k}(t,cond) = softmax over K for each fore channel c
            sim_c = sum_k pf_c * pb * wk_{c,k}(t,cond)
        """
        b, cf, h, w = x_fore.shape
        k = self.hog_bins
        eps = self.hog_eps

        if cf != self.fore_in_ch:
            raise ValueError(f"Unexpected fore channel count: got {cf}, expected {self.fore_in_ch}")

        # 1) HOG maps
        hog_back_map = self.freq_hog(x_back)  # (B, 1*K, H, W)
        hog_fore_map = self.freq_hog(x_fore)  # (B, Cf*K, H, W)

        # 2) reshape to (B,C,H,W,K)
        pb = hog_back_map.view(b, 1, k, h, w).permute(0, 1, 3, 4, 2).contiguous()   # (B,1,H,W,K)
        pf = hog_fore_map.view(b, cf, k, h, w).permute(0, 1, 3, 4, 2).contiguous()  # (B,Cf,H,W,K)

        # 3) normalize across K -> distributions
        pb = pb / (pb.sum(dim=-1, keepdim=True) + eps)
        pf = pf / (pf.sum(dim=-1, keepdim=True) + eps)

        # 4) E+：逐通道 bin 权重 wk_{c,k}(t,cond)
        ctx = torch.cat([t_emb, cond_emb], dim=1)          # (B, 2*time_dim)
        wk_logits = self.hog_bin_weight_head(ctx).view(b, cf, k)  # (B,Cf,K)
        wk = torch.softmax(wk_logits, dim=-1).view(b, cf, 1, 1, k)  # (B,Cf,1,1,K)

        # 5) per-fore-channel similarity with back: sim_c (B,Cf,H,W)
        sim_c = (pf * pb * wk).sum(dim=-1)  # (B,Cf,H,W)

        # 6) per-channel alpha
        alpha_c = torch.clamp(sim_c, 0.0, 1.0)   # (B,Cf,H,W)

        # 7) frequency-domain selective sharing
        Fz = torch.fft.fftshift(torch.fft.fft2(x_fore, dim=(-2, -1), norm="ortho"), dim=(-2, -1))  # (B,Cf,H,W) complex
        Bz = torch.fft.fftshift(torch.fft.fft2(x_back, dim=(-2, -1), norm="ortho"), dim=(-2, -1))  # (B,1,H,W) complex

        # broadcast back to fore channels
        B_bc = Bz.expand(-1, cf, -1, -1)  # (B,Cf,H,W)

        # ---- channel aggregation over fore channels (NO topk) ----
        tau = max(self.hog_softmax_tau, 1e-6)
        w_ch = torch.softmax(alpha_c / tau, dim=1)  # (B,Cf,H,W), sum over Cf = 1

        # back 的 alpha：softmax 加权期望 汇总
        alpha_back = torch.clamp((w_ch * alpha_c).sum(dim=1, keepdim=True), 0.0, 1.0)  # (B,1,H,W)

        # fore->back 的频谱摘要：F_mix = sum_c w_c * Fz_c
        F_mix = (w_ch.to(Fz.dtype) * Fz).sum(dim=1, keepdim=True)  # (B,1,H,W) complex

        # ---- apply mixing ----
        F_shared = 0.5 * (Fz + B_bc)          # (B,Cf,H,W)
        B_shared = 0.5 * (Bz + F_mix)         # (B,1,H,W)

        # fore: per-channel alpha_c
        Fz2 = (1.0 - alpha_c).to(Fz.dtype) * Fz + alpha_c.to(Fz.dtype) * F_shared

        # back: aggregated alpha_back
        Bz2 = (1.0 - alpha_back).to(Bz.dtype) * Bz + alpha_back.to(Bz.dtype) * B_shared

        # 8) IFFT back to spatial
        Fz2 = torch.fft.ifftshift(Fz2, dim=(-2, -1))
        Bz2 = torch.fft.ifftshift(Bz2, dim=(-2, -1))
        x_fore2 = torch.fft.ifft2(Fz2, dim=(-2, -1), norm="ortho").real
        x_back2 = torch.fft.ifft2(Bz2, dim=(-2, -1), norm="ortho").real

        return x_back2, x_fore2

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

        t_emb = self.time_mlp(t)                 # (B,time_dim)
        cond_emb = self.cond_summary(image)      # (B,time_dim)

        # split raw inputs
        x_back = x_t[:, 0:1]   # (B,1,H,W)
        x_fore = x_t[:, 1:]    # (B,Cf,H,W)

        # ✅ Scheme-1 HOG routing BEFORE stems/encoders (E+: uses cond+t)
        x_back, x_fore = self._hog_routing_fuse(x_back, x_fore, t_emb=t_emb, cond_emb=cond_emb)

        # stems
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
            cond, c_skip = enc_c(cond, t_emb)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c_skip)

            b_main, f_main, c_main = b_skip, f_skip, c_skip

            # FA uses cond encoder skip as KV (cond not updated)
            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_main, b_main, _ = fa(f_main, b_main, c_main, t_emb=t_emb)

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
    noise_strength: float = 1.0,
    # HOG routing
    hog_bins: int = 8,
    hog_tau: float = 0.35,
    hog_mag_weight: bool = True,
    hog_softmax_tau: float = 0.10,
    # E+
    hog_bin_weight_hidden: int = 0,
    init_wk_uniform: bool = True,
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
        noise_strength=noise_strength,
        hog_bins=hog_bins,
        hog_tau=hog_tau,
        hog_mag_weight=hog_mag_weight,
        hog_softmax_tau=hog_softmax_tau,
        hog_bin_weight_hidden=hog_bin_weight_hidden,
        init_wk_uniform=init_wk_uniform,
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
        noise_strength=1.0,
        hog_bins=8,
        hog_tau=0.35,
        hog_mag_weight=True,
        hog_softmax_tau=0.10,
        hog_bin_weight_hidden=0,   # 0: linear; >0: MLP
        init_wk_uniform=True,      # True: 让 wk 初始接近均匀（更稳）
    )

    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v = net(x_t, img, t)
    print("v:", v.shape)  # (2, 4, 256, 256)