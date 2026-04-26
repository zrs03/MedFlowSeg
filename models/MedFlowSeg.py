# -*- coding: utf-8 -*-
"""
✅ MedFlowSeg 修改版：
1) Two-Stream 双分支架构 (Cond 纯空间无时序, Flow 含时间步 t)
2) DBSA 改为：
   - Cond Decoder 最后一层特征 cond_dec_feats[0] -> [DBSA] -> Flow Enc Level 1
3) FAA 保持使用 Encoder 特征：
   - Cond Enc Deep -> [FAA] -> Flow Enc Deep
   - Cond Bottleneck -> [FAA] -> Flow Bottleneck
4) 取消原本 Decoder 端 ZeroConv 注入
5) Cond 特征始终不受 t_emb 污染
6) 仅替换原 SOSX 分支：
   - 先 patchify 得到 T_f, T_c
   - 构造 T_a, T_d, T_r
   - 得到 T_z = phi([T_a, T_d, T_r])
   - 再得到 \tilde{T_f}, \tilde{T_c}
   - 把 (\tilde{T_f}, \tilde{T_c}) 送入原固定 modulator gate 分支
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Time embedding + FiLM (flow branch only)
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
# Standard ResBlock (cond branch)
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
    """ResBlock with FiLM time conditioning (flow branch)"""
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
# ViT-style patchify & unpatchify
# =========================================================
class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=self.patch,
            stride=self.patch,
            bias=True,
        )

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
        b, n, _ = tokens.shape
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
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=dropout,
            batch_first=True,
        )
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
# DBSA
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


class SpatialDirectHeadFiLM(nn.Module):
    def __init__(self, ch: int, time_ch: int, hidden_mult: int = 2):
        super().__init__()
        hid = int(hidden_mult * ch)
        self.in_norm = GroupNorm32(ch)
        self.conv_in = nn.Conv2d(ch, hid, 3, padding=1)

        self.hid_norm = GroupNorm32(hid)
        self.to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, 2 * hid)
        )

        self.conv_mid = nn.Conv2d(hid, hid, 3, padding=1)
        self.conv_out = nn.Conv2d(hid, ch, 1)

        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(F.silu(self.in_norm(x)))
        h = self.hid_norm(h)

        gamma, beta = self.to_film(t_emb).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        h = (1.0 + gamma) * h + beta
        h = F.silu(self.conv_mid(F.silu(h)))
        return self.conv_out(h)

# =========================================================
# Direct frequency update head with FiLM
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
# 2D Sin-Cos positional encoding
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
# Token Discrepancy Preparation
# =========================================================
class TokenDiscrepancyExtractor(nn.Module):
    """
    方案 A（去掉 confidence）:
    1) patchify 得到 T_f, T_c
    2) 构造 agreement / difference / residual
    3) 三路分别编码
    4) 融合得到 T_z
    5) 用 T_z 重整 T_f, T_c
    """
    def __init__(
        self,
        time_ch: int,
        ch: int,
        token_dim: int,
        patch: int,
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.ch = int(ch)
        self.token_dim = int(token_dim)
        self.patch = int(patch)
        hidden_dim = int(hidden_mult * token_dim)

        self.patch_flow = PatchEmbed2D(
            in_ch=self.ch,
            embed_dim=self.token_dim,
            patch=self.patch,
        )
        self.patch_cond = PatchEmbed2D(
            in_ch=self.ch,
            embed_dim=self.token_dim,
            patch=self.patch,
        )
        
        self.flow_direct = SpatialDirectHeadFiLM(ch, time_ch)
        self.cond_direct = SpatialDirectHeadFiLM(ch, time_ch)

        # 三路 token 分别编码
        self.enc_a = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.enc_d = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.enc_r = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )

        # fuse 成最终证据 token
        self.fuse = nn.Sequential(
            nn.LayerNorm(3 * self.token_dim),
            nn.Linear(3 * self.token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )

        # 用证据 token 去重整 flow / cond token
        self.g_flow = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.g_cond = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # 建议用小初始化，不要全零
        nn.init.normal_(self.g_flow[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.g_flow[-1].bias)
        nn.init.normal_(self.g_cond[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.g_cond[-1].bias)
    

    def forward(
        self,
        flow: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        
        flow = flow + self.flow_direct(flow, t_emb)
        cond = cond + self.cond_direct(cond, t_emb)
        
        T_f, hw_p_f = self.patch_flow(flow)
        T_c, hw_p_c = self.patch_cond(cond)

        if hw_p_f != hw_p_c:
            raise ValueError(
                f"TokenDiscrepancyExtractor needs same patch hw, got {hw_p_f} vs {hw_p_c}"
            )

        # 三类 patch 证据
        T_a = T_f * T_c
        T_d = torch.abs(T_f - T_c)
        T_r = T_f - T_c

        # 分别编码
        T_a = self.enc_a(T_a)
        T_d = self.enc_d(T_d)
        T_r = self.enc_r(T_r)

        # 融合得到修正证据 token
        T_z = self.fuse(torch.cat([T_a, T_d, T_r], dim=-1))

        # 重整输入 token
        T_f_tilde = T_f + self.g_flow(T_z)
        T_c_tilde = T_c + self.g_cond(T_z)

        return T_f_tilde, T_c_tilde, hw_p_f

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
        faa_patch: int = 8,
        faa_attn_dropout: float = 0.0,
        token_dim: Optional[int] = None,
        gate_scale: float = 1.0,
        gate_R: int = 2,
    ):
        super().__init__()
        self.ch = int(ch)

        self.faa_patch = int(faa_patch)
        if token_dim is None:
            token_dim = int(heads * head_dim)
        self.token_dim = int(token_dim)

        self.gate_scale = float(gate_scale)
        self.gate_R = int(gate_R)

        ri_ch = 2 * self.ch

        # 输入分支保持不变
        self.patch_flow = PatchEmbed2D(in_ch=ri_ch, embed_dim=self.token_dim, patch=self.faa_patch)
        self.patch_cond = PatchEmbed2D(in_ch=ri_ch, embed_dim=self.token_dim, patch=self.faa_patch)

        # token discrepancy preparation
        self.disc_prep = TokenDiscrepancyExtractor(
            ch=self.ch,
            token_dim=self.token_dim,
            patch=self.faa_patch,
            time_ch=time_ch,
        )

        # 【修改点 1】：重命名并负责还原 Cond
        self.unpatch_cond = Unpatch2D(embed_dim=self.token_dim, out_ch=ri_ch, patch=self.faa_patch)

        # 【修改点 2】：交叉注意力，现在用于更新 Cond
        self.attn_cond = TokenCrossAttention(dim=self.token_dim, heads=heads, dropout=faa_attn_dropout)

        self.ffn_cond = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 4 * self.token_dim),
            nn.SiLU(),
            nn.Linear(4 * self.token_dim, self.token_dim),
        )

        self.direct_head_flow = FreqDirectHeadFiLM(ch=self.ch, time_ch=time_ch)
        self.direct_head_cond = FreqDirectHeadFiLM(ch=self.ch, time_ch=time_ch)

        # 门控分支的重命名
        gate_in_dim = 2 * self.token_dim
        self.gate_cond_blocks = nn.ModuleList(
            [TokenFiLMBlock(gate_in_dim, time_ch, mlp_hidden_mult=2) for _ in range(self.gate_R)]
        )
        self.gate_cond_out = nn.Sequential(nn.LayerNorm(gate_in_dim), nn.Linear(gate_in_dim, self.token_dim))
        nn.init.zeros_(self.gate_cond_out[-1].weight)
        nn.init.zeros_(self.gate_cond_out[-1].bias)

        self._pos_cache: Dict[Tuple[int, int, int, torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def _to_ri(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)

    @staticmethod
    def _to_complex(ri: torch.Tensor) -> torch.Tensor:
        _, c2, _, _ = ri.shape
        c = c2 // 2
        return torch.complex(ri[:, :c], ri[:, c:])

    def _get_pos(self, hp: int, wp: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (hp, wp, self.token_dim, device, dtype)
        pos = self._pos_cache.get(key)
        if pos is None or pos.device != device or pos.dtype != dtype:
            # 假设外部有 build_2d_sincos_pos_embed
            pos = build_2d_sincos_pos_embed(hp, wp, self.token_dim, device=device, dtype=dtype)
            self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        flow: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # -------------------------------------------------
        # 傅里叶变换保持不变
        # -------------------------------------------------
        Fz = torch.fft.fftshift(torch.fft.fft2(flow, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        Fri = self._to_ri(Fz)
        Cri = self._to_ri(Cz)

        Fri = Fri + self.direct_head_flow(Fri, t_emb)
        Cri = Cri + self.direct_head_cond(Cri, t_emb)

        F_tok, hw_p = self.patch_flow(Fri)
        C_tok, _ = self.patch_cond(Cri)

        hp, wp = hw_p
        pos = self._get_pos(hp, wp, device=F_tok.device, dtype=F_tok.dtype)
        F_tok = F_tok + pos
        C_tok = C_tok + pos

        # 【修改点 5】：C 去 Attend F，更新的是 C
        C_tok2 = C_tok + self.attn_cond(C_tok, F_tok) 
        C_tok2 = C_tok2 + self.ffn_cond(C_tok2)

        # -------------------------------------------------
        # 构造 token discrepancy 保持不变
        # -------------------------------------------------
        T_f_tilde, T_c_tilde, hw_d = self.disc_prep(flow, cond, t_emb)
        
        T_f_tilde = T_f_tilde + pos
        T_c_tilde = T_c_tilde + pos
        
        if hw_d != hw_p:
            raise ValueError(f"Discrepancy token hw {hw_d} != main token hw {hw_p}")

        # -------------------------------------------------
        # 【修改点 6】：门控网络更新 C，拼接顺序通常将 condition 放在前面，target 放在后面，因此调换为 [T_f_tilde, T_c_tilde]
        # -------------------------------------------------
        g_cond = torch.cat([T_f_tilde, T_c_tilde], dim=-1)
        for blk in self.gate_cond_blocks:
            g_cond = blk(g_cond, t_emb)
        gate_cond = torch.tanh(self.gate_cond_out(g_cond))

        # 作用于 C
        C_tok3 = C_tok2 * (1.0 + self.gate_scale * gate_cond)

        # -------------------------------------------------
        # 【修改点 7】：逆变换输出改为处理 C
        # -------------------------------------------------
        Cri2 = self.unpatch_cond(C_tok3, hw_p)
        C2c = self._to_complex(Cri2)
        C2c = torch.fft.ifftshift(C2c, dim=(-2, -1))
        cond_out = torch.fft.ifft2(C2c, dim=(-2, -1), norm="ortho").real
        
        return cond_out

# =========================================================
# Encoder / Decoder stages
# =========================================================
class EncoderStageFiLM(nn.Module):
    """Time-conditioned EncoderStage (flow branch)"""
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
    """Standard EncoderStage (cond branch)"""
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
    """Single-branch decoder with FiLM (flow branch)"""
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
    """Standard decoder (cond branch)"""
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
        faa_patch: int = 8,
        faa_attn_dropout: float = 0.0,
        faa_token_dim: Optional[int] = None,
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

        self.faa_patch = int(faa_patch)
        self.faa_attn_dropout = float(faa_attn_dropout)
        self.faa_token_dim = faa_token_dim
        self.faa_gate_scale = float(faa_gate_scale)
        self.faa_gate_R = int(faa_gate_R)

        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

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

            self.flow_enc.append(
                EncoderStageFiLM(in_ch, out_ch, self.time_dim, self.num_res_blocks, down, self.dropout)
            )
            self.cond_enc.append(
                EncoderStage(in_ch, out_ch, self.num_res_blocks, down, self.dropout)
            )

        # flow decoder
        self.dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.chs[i]
            in_ch = self.chs[i + 1] if i < (n_levels - 1) else self.chs[i]
            out_ch = self.chs[i]
            up = i < (n_levels - 1)
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

        def make_faa_list(ch: int) -> nn.ModuleList:
            mods = nn.ModuleList()
            for _ in range(self.faa_stacks):
                mods.append(
                    FAAModule(
                        ch=ch,
                        time_ch=self.time_dim,
                        heads=self.attn_heads,
                        head_dim=self.attn_head_dim,
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

        # DBSA now uses cond decoder final feature -> flow encoder first feature
        self.dbsa = DBSA(
            ch=self.chs[0],
            low_gauss_ks=usa_low_gauss_ks,
            high_gauss_ks=usa_high_gauss_ks,
            low_gauss_sigma=usa_low_gauss_sigma,
            high_gauss_sigma=usa_high_gauss_sigma,
            init_bias=usa_init_bias,
        )

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
        # Stage 1: Cond branch
        # ====================================================
        cond_enc_feats: List[torch.Tensor] = []
        cond = cond0
        for enc_c in self.cond_enc:
            c_feat = enc_c(cond)
            cond_enc_feats.append(c_feat)
            cond = enc_c.down(c_feat) if enc_c.down is not None else c_feat

        cond_bot = cond

        c_h = cond_bot
        n_levels = len(self.chs)
        cond_dec_feats: List[torch.Tensor] = [None] * n_levels  # type: ignore
        for cdec_idx, (cdec, c_skip) in enumerate(zip(self.cond_dec, reversed(cond_enc_feats))):
            c_h = cdec(c_h, c_skip)
            level = (n_levels - 1) - cdec_idx
            cond_dec_feats[level] = c_h

        seg_logits = self.cond_seg_head(cond_dec_feats[0])

        # ====================================================
        # Stage 2: Flow branch
        # ====================================================
        flow_skips: List[torch.Tensor] = []
        flow = flow0

        for level, enc_flow in enumerate(self.flow_enc):
            f_skip = enc_flow(flow, t_emb)

            # DBSA: cond decoder final feature -> flow encoder first feature
            if level == 0:
                f_skip = self.dbsa(f_cond=cond_dec_feats[0], f_flow=f_skip)

            flow_skips.append(f_skip)
            f_main = f_skip

            # FAA uses encoder features only
            tmp = cond_enc_feats[level]
            
            if level in self.attn_level_set:
                faa_list: nn.ModuleList = self.couple_modules[str(level)]
                for faa in faa_list:
                    tmp = faa(
                        flow=f_main,
                        cond=tmp,
                        t_emb=t_emb,
                    )
            f_main = tmp

            flow = enc_flow.down(f_main) if enc_flow.down is not None else f_main

        # Bottleneck FAA
        if self.bottleneck_coupler is not None:
            tmp = cond_bot
            for faa in self.bottleneck_coupler:
                tmp = faa(
                    flow=flow,
                    cond=tmp,
                    t_emb=t_emb,
                )
            flow = tmp

        # Flow decoder
        h = flow
        for dec_idx, (dec, f_skip) in enumerate(zip(self.dec, reversed(flow_skips))):
            h = dec(h, f_skip, t_emb)

        h = F.silu(h)
        return self.out_v(h), seg_logits


# =========================================================
# build_model
# =========================================================
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
    freq_attn_depth: int = 0,
    use_bottleneck_coupling: bool = True,
    faa_stacks: int = 4,
    faa_patch: int = 2,
    faa_attn_dropout: float = 0.0,
    faa_token_dim: Optional[int] = None,
    faa_gate_scale: float = 1.0,
    faa_gate_R: int = 2,
    usa_low_gauss_ks: int = 3,
    usa_high_gauss_ks: int = 5,
    usa_low_gauss_sigma: float = 1.5,
    usa_high_gauss_sigma: float = 1.5,
    usa_init_bias: float = -2.0,
) -> MainNet:
    return MainNet(
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
        faa_patch=faa_patch,
        faa_attn_dropout=faa_attn_dropout,
        faa_token_dim=faa_token_dim,
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
        faa_patch=8,
        faa_attn_dropout=0.0,
        faa_token_dim=None,
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

    print("✅ Build Successfully with Token Discrepancy Preparation!")
    print(f"Input x_t shape : {x_t.shape}")
    print(f"Output v shape  : {v.shape}")
    print(f"Seg Logits shape: {seg_logits.shape}")