# -*- coding: utf-8 -*-
"""
✅ 严格对齐架构图 (MedFlowSeg) 的修改版：
1) Two-Stream 双分支架构 (Cond 纯空间无时序, Flow 含时间步 t)。
2) Encoder 端交互：
   - Cond Enc Level 1 -> [DB-SA] -> Flow Enc Level 1
   - Cond Enc Deep -> [FAA] -> Flow Enc Deep
   - Cond Bottleneck -> [FAA] -> Flow Bottleneck
3) Decoder 端交互 (Zero Conv)：
   - Cond Dec -> [Zero Conv] -> Flow Dec (严格把 Decoder 特征注入给 Flow Decoder)
4) 修复幻觉：彻底移除了 ctrl_proc_levels，保证传给 Flow 的 Cond 特征不被 t_emb 污染。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Time embedding + FiLM (conv) - 保留给 flow 分支
# =========================================================
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


# =========================================================
# 标准的 ResBlock (无时间嵌入，用于 cond 分支)
# =========================================================
class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        self.norm1 = GroupNorm32(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = GroupNorm32(out_ch)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class ResBlockFiLM(nn.Module):
    """ResBlock with FiLM time conditioning (用于 flow 分支)"""
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


# =========================================================
# ZeroConv (ControlNet-style injection to decoder only)
# =========================================================
class ZeroConv2d(nn.Module):
    """1x1 conv with ZERO init."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =========================================================
# ViT-style patchify & unpatchify
# =========================================================
class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=self.patch, stride=self.patch, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, c, h, w = x.shape
        p = self.patch
        if (h % p) != 0 or (w % p) != 0:
            raise ValueError(f"PatchEmbed2D requires H,W divisible by patch={p}, got {(h, w)}")
        x = self.proj(x)
        hp, wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x, (hp, wp)


class Unpatch2D(nn.Module):
    def __init__(self, embed_dim: int, out_ch: int, patch: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.out_ch = int(out_ch)
        self.patch = int(patch)
        self.to_patch = nn.Linear(embed_dim, out_ch * patch * patch, bias=True)

    def forward(self, tokens: torch.Tensor, hw_patch: Tuple[int, int]) -> torch.Tensor:
        b, n, d = tokens.shape
        hp, wp = hw_patch
        p = self.patch
        if n != hp * wp:
            raise ValueError(f"Unpatch2D got N={n} but Hp*Wp={hp * wp}")

        x = self.to_patch(tokens)
        x = x.view(b, hp, wp, self.out_ch, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, self.out_ch, hp * p, wp * p)
        return x


# =========================================================
# Token Cross-Attention
# =========================================================
class TokenCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(self.dim)
        self.ln_kv = nn.LayerNorm(self.dim)
        self.proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.Dropout(dropout))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        return self.proj(out)


# =========================================================
# Learnable Gaussian (per-channel depthwise conv)
# =========================================================
class LearnableGaussianBlur2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, sigma_init: float = 1.5, eps: float = 1e-8):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)

        self.raw = nn.Parameter(torch.zeros(self.channels, 1, self.kernel_size, self.kernel_size))
        self._init_gaussian(sigma_init)

    def _init_gaussian(self, sigma: float):
        k = self.kernel_size
        ax = torch.arange(k) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * (sigma ** 2)))
        g = g / (g.sum() + 1e-8)
        raw = torch.log(g.clamp_min(1e-12))
        raw = raw.view(1, 1, k, k).repeat(self.channels, 1, 1, 1)
        with torch.no_grad():
            self.raw.copy_(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.raw)
        w = w / (w.sum(dim=(-1, -2), keepdim=True) + self.eps)
        pad = self.kernel_size // 2
        return F.conv2d(x, w, bias=None, stride=1, padding=pad, groups=self.channels)


# =========================================================
# DBSA (独立双高斯核版)
# =========================================================
class DBSA(nn.Module):
    def __init__(
        self,
        ch: int,
        low_gauss_ks: int = 3,
        high_gauss_ks: int = 5,
        low_gauss_sigma: float = 1.5,
        high_gauss_sigma: float = 1.5,
        init_bias: float = -2.0,
    ):
        super().__init__()
        self.ch = int(ch)

        self.conv_anc = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=True)
        self.gauss_low = LearnableGaussianBlur2d(self.ch, kernel_size=low_gauss_ks, sigma_init=low_gauss_sigma)
        self.gauss_high = LearnableGaussianBlur2d(self.ch, kernel_size=high_gauss_ks, sigma_init=high_gauss_sigma)

        self.conv_high = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=True)
        self.conv_low = nn.Conv2d(self.ch, self.ch, kernel_size=3, padding=1, bias=True)

        self.conv_gate = nn.Conv2d(self.ch * 2, 1, kernel_size=1, bias=True)

        nn.init.zeros_(self.conv_gate.weight)
        nn.init.constant_(self.conv_gate.bias, init_bias)

    def forward(self, f_cond: torch.Tensor, f_flow: torch.Tensor) -> torch.Tensor:
        if f_cond.shape != f_flow.shape:
            raise ValueError("DBSA requires f_cond/f_flow same shape")

        anc = self.conv_anc(f_cond)

        low = self.gauss_low(anc)
        high = anc - self.gauss_high(anc)

        high_feat = self.conv_high(high)
        low_feat = self.conv_low(low)

        feat_cat = torch.cat([high_feat, low_feat], dim=1)
        g = torch.sigmoid(self.conv_gate(feat_cat))

        return f_flow + f_flow * g


# =========================================================
# Dense frequency-domain HOG on feature maps
# =========================================================
def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi

class SOSX(nn.Module):
    def __init__(
        self,
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        eps: float = 1e-8,
        mag_weight: bool = True,
    ):
        super().__init__()
        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.eps = float(eps)
        self.mag_weight = bool(mag_weight)

        k = torch.arange(self.hog_bins).float()
        theta = -math.pi + (2.0 * math.pi) * k / float(self.hog_bins)
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        X = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        mag = X.abs().clamp_min(self.eps)
        phi = torch.angle(X)

        theta = self.theta.view(1, 1, 1, 1, self.hog_bins).to(device=x.device, dtype=phi.dtype)
        delta = _wrap_pi(phi.unsqueeze(-1) - theta)

        logits = torch.cos(delta) / max(self.hog_tau, 1e-6)
        p = F.softmax(logits, dim=-1)

        if self.mag_weight:
            p = p * mag.unsqueeze(-1)

        p = p.permute(0, 1, 4, 2, 3).contiguous()
        hog_map = p.view(b, c * self.hog_bins, h, w)
        return hog_map


# =========================================================
# direct frequency update head with FiLM
# =========================================================
class FreqDirectHeadFiLM(nn.Module):
    def __init__(self, ch: int, time_ch: int, hidden_mult: int = 2):
        super().__init__()
        self.ch = int(ch)
        hid = int(hidden_mult * ch)

        self.in_norm = GroupNorm32(2 * ch)
        self.conv_in = nn.Conv2d(2 * ch, hid, 3, padding=1)

        self.hid_norm = GroupNorm32(hid)
        self.to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, 2 * hid)
        )

        self.conv_mid = nn.Conv2d(hid, hid, 3, padding=1)
        self.conv_out = nn.Conv2d(hid, 2 * ch, 1)

        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x_ri2c: torch.Tensor, t_emb: torch.Tensor):
        h = self.conv_in(F.silu(self.in_norm(x_ri2c)))

        h = self.hid_norm(h)
        gamma, beta = self.to_film(t_emb).chunk(2, dim=1)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = (1.0 + gamma) * h + beta
        h = F.silu(self.conv_mid(F.silu(h)))

        out = self.conv_out(h)
        return out


# =========================================================
# Token FiLM block
# =========================================================
class TokenFiLMBlock(nn.Module):
    def __init__(self, dim: int, time_ch: int, mlp_hidden_mult: int = 2):
        super().__init__()
        self.dim = int(dim)
        hid = int(mlp_hidden_mult * dim)

        self.ln = nn.LayerNorm(self.dim)
        self.to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, hid),
            nn.SiLU(),
            nn.Linear(hid, 2 * self.dim),
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
        )

        last = self.to_film[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        if isinstance(self.ffn[0], nn.Linear):
            nn.init.normal_(self.ffn[0].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.ffn[0].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        gamma, beta = self.to_film(t_emb).chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
        h = (1.0 + gamma) * h + beta
        h = self.ffn(h)
        return x + h


# =========================================================
# 2D Sin-Cos positional编码
# =========================================================
def _sincos_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((length, 0), device=device, dtype=dtype)

    pos = torch.arange(length, device=device, dtype=dtype)
    half = dim // 2

    if half > 0:
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        args = pos[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    else:
        emb = torch.zeros((length, 0), device=device, dtype=dtype)

    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    elif emb.shape[1] > dim:
        emb = emb[:, :dim]
    return emb

def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dim_y = dim // 2
    dim_x = dim - dim_y

    emb_y = _sincos_1d(h, dim_y, device, dtype)
    emb_x = _sincos_1d(w, dim_x, device, dtype)

    yy = emb_y[:, None, :].expand(h, w, dim_y)
    xx = emb_x[None, :, :].expand(h, w, dim_x)
    pos = torch.cat([yy, xx], dim=-1).reshape(1, h * w, dim)
    return pos


# =========================================================
# FAA module
# =========================================================
class FAAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        hog_bins: int,
        noise_strength: float = 1.0,
        faa_patch: int = 8,
        faa_attn_dropout: float = 0.0,
        token_dim: Optional[int] = None,
        gate_scale: float = 1.0,
        gate_R: int = 2,
    ):
        super().__init__()
        self.ch = int(ch)
        self.hog_bins = int(hog_bins)
        self.noise_strength = float(noise_strength)

        self.faa_patch = int(faa_patch)
        if token_dim is None:
            token_dim = int(heads * head_dim)
        self.token_dim = int(token_dim)

        self.gate_scale = float(gate_scale)
        self.gate_R = int(gate_R)

        ri_ch = 2 * self.ch
        hog_ch = self.ch * self.hog_bins

        self.patch_flow = PatchEmbed2D(in_ch=ri_ch, embed_dim=self.token_dim, patch=self.faa_patch)
        self.patch_cond = PatchEmbed2D(in_ch=ri_ch, embed_dim=self.token_dim, patch=self.faa_patch)

        self.patch_hflow = PatchEmbed2D(in_ch=hog_ch, embed_dim=self.token_dim, patch=self.faa_patch)
        self.patch_hcond = PatchEmbed2D(in_ch=hog_ch, embed_dim=self.token_dim, patch=self.faa_patch)

        self.unpatch_flow = Unpatch2D(embed_dim=self.token_dim, out_ch=ri_ch, patch=self.faa_patch)

        self.attn_flow = TokenCrossAttention(dim=self.token_dim, heads=heads, dropout=faa_attn_dropout)

        self.ffn_flow = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 4 * self.token_dim),
            nn.SiLU(),
            nn.Linear(4 * self.token_dim, self.token_dim),
        )

        self.direct_head_flow = FreqDirectHeadFiLM(ch=self.ch, time_ch=time_ch)

        gate_in_dim = 2 * self.token_dim
        self.gate_flow_blocks = nn.ModuleList(
            [TokenFiLMBlock(gate_in_dim, time_ch, mlp_hidden_mult=2) for _ in range(self.gate_R)]
        )
        self.gate_flow_out = nn.Sequential(nn.LayerNorm(gate_in_dim), nn.Linear(gate_in_dim, self.token_dim))
        nn.init.zeros_(self.gate_flow_out[-1].weight)
        nn.init.zeros_(self.gate_flow_out[-1].bias)

        self._pos_cache: Dict[Tuple[int, int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _to_ri(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def _to_complex(ri: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = ri.shape
        c = c2 // 2
        return torch.complex(ri[:, :c], ri[:, c:])

    def _get_pos(self, hp: int, wp: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (hp, wp, self.token_dim, device, dtype)
        pos = self._pos_cache.get(key)
        if pos is None or pos.device != device or pos.dtype != dtype:
            pos = build_2d_sincos_pos_embed(hp, wp, self.token_dim, device=device, dtype=dtype)
            self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        flow: torch.Tensor,
        cond: torch.Tensor,
        hog_flow: torch.Tensor,
        hog_cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        Fz = torch.fft.fftshift(torch.fft.fft2(flow, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        Fri = self._to_ri(Fz)
        Cri = self._to_ri(Cz)

        Fri = Fri + self.direct_head_flow(Fri, t_emb)

        F_tok, hw_p = self.patch_flow(Fri)
        C_tok, _ = self.patch_cond(Cri)

        hp, wp = hw_p
        pos = self._get_pos(hp, wp, device=F_tok.device, dtype=F_tok.dtype)
        F_tok = F_tok + pos
        C_tok = C_tok + pos

        Hf_tok, _ = self.patch_hflow(hog_flow)
        Hc_tok, _ = self.patch_hcond(hog_cond)

        F_tok2 = F_tok + self.attn_flow(F_tok, C_tok)
        F_tok2 = F_tok2 + self.ffn_flow(F_tok2)

        g_flow = torch.cat([Hc_tok, Hf_tok], dim=-1)
        for blk in self.gate_flow_blocks:
            g_flow = blk(g_flow, t_emb)
        gate_flow = torch.tanh(self.gate_flow_out(g_flow))
        F_tok3 = F_tok2 * (1.0 + self.gate_scale * gate_flow)

        Fri2 = self.unpatch_flow(F_tok3, hw_p)
        F2c = self._to_complex(Fri2)
        F2c = torch.fft.ifftshift(F2c, dim=(-2, -1))
        flow_out = torch.fft.ifft2(F2c, dim=(-2, -1), norm="ortho").real
        return flow_out


# =========================================================
# Encoder/Decoder stages
# =========================================================
class EncoderStageFiLM(nn.Module):
    """带时间控制的 EncoderStage (用于 flow 分支)"""
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, num_blocks: int, down: bool, dropout: float):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(int(num_blocks)):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.down = Downsample(out_ch) if down else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x

class EncoderStage(nn.Module):
    """标准的 EncoderStage (无时间嵌入，用于 cond 分支)"""
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, down: bool, dropout: float):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(int(num_blocks)):
            blocks.append(ResBlock(ch, out_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.down = Downsample(out_ch) if down else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class DecoderStageSingle(nn.Module):
    """单分支 decoder with FiLM (用于 flow 分支)"""
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
        self.fuse_norm = GroupNorm32(in_ch)

        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(int(num_blocks)):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)
        x = x + self.skip_to_x(skip)
        x = F.silu(self.fuse_norm(x))
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x

class DecoderStage(nn.Module):
    """标准单分支 decoder (无时间嵌入，用于 cond 分支)"""
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        num_blocks: int,
        up: bool,
        dropout: float,
    ):
        super().__init__()
        self.up = Upsample(in_ch) if up else None
        self.skip_to_x = nn.Identity() if skip_ch == in_ch else nn.Conv2d(skip_ch, in_ch, 1, bias=True)
        self.fuse_norm = GroupNorm32(in_ch)

        blocks: List[nn.Module] = []
        ch = in_ch
        for _ in range(int(num_blocks)):
            blocks.append(ResBlock(ch, out_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)
        x = x + self.skip_to_x(skip)
        x = F.silu(self.fuse_norm(x))
        for blk in self.blocks:
            x = blk(x)
        return x


# =========================================================
# Main UNet
# =========================================================
class MainNet(nn.Module):
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
        faa_stacks: int,
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        hog_mag_weight: bool = True,
        hog_eps: float = 1e-8,
        control_inject_to_decoder: bool = True,
        faa_patch: int = 8,
        faa_attn_dropout: float = 0.0,
        faa_token_dim: Optional[int] = None,
        faa_noise_strength: float = 1.0,
        faa_gate_scale: float = 1.0,
        faa_gate_R: int = 2,

        usa_low_gauss_ks: int = 3,
        usa_high_gauss_ks: int = 5,
        usa_low_gauss_sigma: float = 1.5,
        usa_high_gauss_sigma: float = 1.5,
        usa_init_bias: float = -2.0,
    ):
        super().__init__()

        if xt_in_ch < 1:
            raise ValueError("xt_in_ch must be >= 1.")
        if freq_attn_depth < 0:
            raise ValueError("freq_attn_depth must be >= 0.")
        if faa_stacks < 1:
            raise ValueError("faa_stacks must be >= 1")

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
        self.faa_stacks = int(faa_stacks)

        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.hog_eps = float(hog_eps)
        self.hog_mag_weight = bool(hog_mag_weight)

        self.control_inject_to_decoder = bool(control_inject_to_decoder)

        self.faa_patch = int(faa_patch)
        self.faa_attn_dropout = float(faa_attn_dropout)
        self.faa_token_dim = faa_token_dim
        self.faa_noise_strength = float(faa_noise_strength)

        self.faa_gate_scale = float(faa_gate_scale)
        self.faa_gate_R = int(faa_gate_R)

        # 仅留给 flow 分支的时间 MLP
        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

        # stems
        self.flow_stem = nn.Conv2d(self.xt_in_ch, self.base_ch, 3, padding=1)
        self.cond_stem = nn.Conv2d(self.img_in_ch, self.base_ch, 3, padding=1)

        self.chs = [self.base_ch * m for m in self.ch_mult]
        n_levels = len(self.chs)

        # encoders
        self.flow_enc = nn.ModuleList()
        self.cond_enc = nn.ModuleList()
        for i in range(n_levels):
            in_ch = self.chs[i - 1] if i > 0 else self.base_ch
            out_ch = self.chs[i]
            down = i < (n_levels - 1)
            # flow 保持含时 EncoderStageFiLM
            self.flow_enc.append(EncoderStageFiLM(in_ch, out_ch, self.time_dim, self.num_res_blocks, down, self.dropout))
            # cond 替换为标准无时的 EncoderStage
            self.cond_enc.append(EncoderStage(in_ch, out_ch, self.num_res_blocks, down, self.dropout))

        # main decoder
        self.dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.chs[i]
            in_ch = self.chs[i + 1] if i < (n_levels - 1) else self.chs[i]
            out_ch = self.chs[i]
            up = i < (n_levels - 1)
            # flow 保持含时 DecoderStageSingle
            self.dec.append(
                DecoderStageSingle(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    time_ch=self.time_dim,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                )
            )

        # cond decoder
        self.cond_dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.chs[i]
            in_ch = self.chs[i + 1] if i < (n_levels - 1) else self.chs[i]
            out_ch = self.chs[i]
            up = i < (n_levels - 1)
            # cond 替换为标准无时 DecoderStage
            self.cond_dec.append(
                DecoderStage(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                )
            )

        self.out_v = nn.Conv2d(self.base_ch, self.xt_in_ch, 3, padding=1)

        K_levels = min(self.freq_attn_depth, n_levels)
        self.attn_level_set = set(range(n_levels - K_levels, n_levels))

        self.freq_hog = SOSX(
            hog_bins=self.hog_bins,
            hog_tau=self.hog_tau,
            eps=self.hog_eps,
            mag_weight=self.hog_mag_weight,
        )

        self.hog_norm_levels = nn.ModuleList([
            GroupNorm32(num_channels=self.chs[i] * self.hog_bins) if (i in self.attn_level_set) else nn.Identity()
            for i in range(n_levels)
        ])
        self.hog_norm_mid = GroupNorm32(num_channels=self.chs[-1] * self.hog_bins) if self.use_bottleneck_coupling else nn.Identity()

        def make_faa_list(ch: int) -> nn.ModuleList:
            mods = nn.ModuleList()
            for _ in range(self.faa_stacks):
                mods.append(
                    FAAModule(
                        ch=ch,
                        time_ch=self.time_dim,
                        heads=self.attn_heads,
                        head_dim=self.attn_head_dim,
                        hog_bins=self.hog_bins,
                        noise_strength=self.faa_noise_strength,
                        faa_patch=self.faa_patch,
                        faa_attn_dropout=self.faa_attn_dropout,
                        token_dim=self.faa_token_dim,
                        gate_scale=self.faa_gate_scale,
                        gate_R=self.faa_gate_R,
                    )
                )
            return mods

        self.couple_modules = nn.ModuleDict()
        for i in sorted(self.attn_level_set):
            self.couple_modules[str(i)] = make_faa_list(self.chs[i])

        self.bottleneck_coupler = make_faa_list(self.chs[-1]) if self.use_bottleneck_coupling else None

        # ✅ 彻底移除 ControlResProc (ctrl_proc_levels)，防止 t_emb 污染 Cond Encoder 特征！

        # decoder injection only: 严格负责 Cond Dec 到 Flow Dec 的特征映射
        self.ctrl_zero_dec = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])

        self.dbsa = DBSA(
            ch=self.chs[0],
            low_gauss_ks=usa_low_gauss_ks,
            high_gauss_ks=usa_high_gauss_ks,
            low_gauss_sigma=usa_low_gauss_sigma,
            high_gauss_sigma=usa_high_gauss_sigma,
            init_bias=usa_init_bias,
        )
        
        # cond segmentation head
        self.cond_seg_head = nn.Sequential(
            GroupNorm32(self.chs[0]),
            nn.SiLU(),
            nn.Conv2d(self.chs[0], self.xt_in_ch, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x_t: torch.Tensor, image: torch.Tensor, t: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(t, float):
            t = torch.tensor(t, device=x_t.device)
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.expand(x_t.shape[0])
        elif isinstance(t, torch.Tensor) and t.dim() == 1 and t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])

        t_emb = self.time_mlp(t)

        flow0 = self.flow_stem(x_t)
        cond0 = self.cond_stem(image)

        # ====================================================
        # 第一阶段：独立运行纯空间 Cond 网络 (Encoder -> Decoder)
        # ====================================================
        
        # 1. Cond Encoder
        cond_skips_raw: List[torch.Tensor] = []  # 保存 Cond Encoder 层级特征
        cond = cond0
        for enc_c in self.cond_enc:
            c_skip_raw = enc_c(cond) # 完全独立于时间 t
            cond_skips_raw.append(c_skip_raw)
            cond = enc_c.down(c_skip_raw) if enc_c.down is not None else c_skip_raw

        cond_bot = cond  # Cond 网络的 Bottleneck 特征！

        # 2. Cond Decoder
        c_h = cond_bot
        n_levels = len(self.chs)
        cond_dec_feats: List[torch.Tensor] = [None] * n_levels  # type: ignore
        for cdec_idx, (cdec, c_skip_raw) in enumerate(zip(self.cond_dec, reversed(cond_skips_raw))):
            c_h = cdec(c_h, c_skip_raw) 
            level = (n_levels - 1) - cdec_idx
            cond_dec_feats[level] = c_h  # 🎈 保存每一层干净的 Decoder 特征

        # 生成 分割预测 (主要为了监督纯空间网络的形态学学习)
        seg_logits = self.cond_seg_head(cond_dec_feats[0])

        # ====================================================
        # 第二阶段：Flow 分支 (结合 DB-SA, FAA, 和 ZeroConv)
        # ====================================================
        
        # 1. Flow Encoder 
        flow_skips: List[torch.Tensor] = []
        flow = flow0

        for level, enc_flow in enumerate(self.flow_enc):
            f_skip = enc_flow(flow, t_emb)

            if level == 0:
                # 🎈 遵循架构图：DB-SA 接收 Cond Encoder 第一层（最干净的浅层）特征
                f_skip = self.dbsa(f_cond=cond_skips_raw[0], f_flow=f_skip)

            flow_skips.append(f_skip)
            f_main = f_skip
            c_main = cond_skips_raw[level]  # 🎈 遵循架构图：直接传入纯净的 Cond Encoder 特征给 FAA

            if level in self.attn_level_set:
                hog_flow = self.freq_hog(f_main)
                hog_cond = self.freq_hog(c_main)

                norm = self.hog_norm_levels[level]
                hog_flow = norm(torch.log1p(hog_flow))
                hog_cond = norm(torch.log1p(hog_cond))

                faa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in faa_list:
                    f_main = fa(
                        flow=f_main,
                        cond=c_main,
                        hog_flow=hog_flow,
                        hog_cond=hog_cond,
                        t_emb=t_emb,
                    )

            flow = enc_flow.down(f_main) if enc_flow.down is not None else f_main

        # 2. Flow Bottleneck
        if self.bottleneck_coupler is not None:
            # 🎈 遵循架构图：Bottleneck FAA 接收 cond_bot 特征
            hog_flow = self.freq_hog(flow)
            hog_cond = self.freq_hog(cond_bot)

            hog_flow = self.hog_norm_mid(torch.log1p(hog_flow))
            hog_cond = self.hog_norm_mid(torch.log1p(hog_cond))

            for fa in self.bottleneck_coupler:
                flow = fa(
                    flow=flow,
                    cond=cond_bot,
                    hog_flow=hog_flow,
                    hog_cond=hog_cond,
                    t_emb=t_emb,
                )

        # 3. Flow Decoder (核心：通过 ZeroConv 接收 Cond Decoder 特征)
        h = flow
        for dec_idx, (dec, f_skip) in enumerate(zip(self.dec, reversed(flow_skips))):
            h = dec(h, f_skip, t_emb)

            if self.control_inject_to_decoder:
                level = (n_levels - 1) - dec_idx
                # 🎈 遵循架构图大粗蓝线：Cond Decoder 完美语义 -> ZeroConv -> Flow Decoder！
                h = h + self.ctrl_zero_dec[level](cond_dec_feats[level])

        h = F.silu(h)
        
        return self.out_v(h), seg_logits

# =========================================================
# build_model 
# =========================================================
def build_model(
    image_hw: Tuple[int, int] = (256, 256),
    img_in_ch: int = 1,
    xt_in_ch: int = 4,
    base_ch: int = 64,
    ch_mult: Sequence[int] = (1, 2, 4, 4),
    num_res_blocks: int = 2,
    time_dim: int = 256,
    dropout: float = 0.0,
    attn_heads: int = 4,
    attn_head_dim: int = 32,
    freq_attn_depth: int = 0,
    use_bottleneck_coupling: bool = True,
    faa_stacks: int = 4,
    hog_bins: int = 8,
    hog_tau: float = 0.35,
    hog_mag_weight: bool = True,
    hog_eps: float = 1e-8,
    # control_resblocks_per_level 被废弃，已从此函数参数中移除。
    control_inject_to_decoder: bool = True,
    faa_patch: int = 2,
    faa_attn_dropout: float = 0.0,
    faa_token_dim: Optional[int] = None,
    faa_noise_strength: float = 1.0,
    faa_gate_scale: float = 1.0,
    faa_gate_R: int = 2,

    usa_low_gauss_ks: int = 3,
    usa_high_gauss_ks: int = 5,
    usa_low_gauss_sigma: float = 1.5,
    usa_high_gauss_sigma: float = 1.5,
    usa_init_bias: float = -2.0,
) -> MainNet:
    return MainNet(
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
        faa_stacks=faa_stacks,
        hog_bins=hog_bins,
        hog_tau=hog_tau,
        hog_mag_weight=hog_mag_weight,
        hog_eps=hog_eps,
        control_inject_to_decoder=control_inject_to_decoder,
        faa_patch=faa_patch,
        faa_attn_dropout=faa_attn_dropout,
        faa_token_dim=faa_token_dim,
        faa_noise_strength=faa_noise_strength,
        faa_gate_scale=faa_gate_scale,
        faa_gate_R=faa_gate_R,
        usa_low_gauss_ks=usa_low_gauss_ks,
        usa_high_gauss_ks=usa_high_gauss_ks,
        usa_low_gauss_sigma=usa_low_gauss_sigma,
        usa_high_gauss_sigma=usa_high_gauss_sigma,
        usa_init_bias=usa_init_bias,
    )


# =========================================================
# quick test
# =========================================================
if __name__ == "__main__":
    net = build_model(
        image_hw=(256, 256),
        img_in_ch=1,
        xt_in_ch=4,
        base_ch=32,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        time_dim=256,
        dropout=0.0,
        attn_heads=4,
        attn_head_dim=32,
        freq_attn_depth=1,
        use_bottleneck_coupling=True,
        faa_stacks=2,
        hog_bins=8,
        hog_tau=0.35,
        hog_mag_weight=True,
        hog_eps=1e-8,
        control_inject_to_decoder=True,
        faa_patch=8,
        faa_attn_dropout=0.0,
        faa_token_dim=None,
        faa_noise_strength=1.0,
        faa_gate_scale=1.0,
        faa_gate_R=2,
        usa_low_gauss_ks=3,
        usa_high_gauss_ks=5,
        usa_low_gauss_sigma=1.5,
        usa_high_gauss_sigma=1.5,
        usa_init_bias=-2.0,
    )

    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v, seg_logits = net(x_t, img, t)

    print("✅ Build Successfully with Perfect Architectural Alignment!")
    print(f"Input x_t shape : {x_t.shape}")
    print(f"Output v shape  : {v.shape}")
    print(f"Seg Logits shape: {seg_logits.shape}")