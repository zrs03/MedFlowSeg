# -*- coding: utf-8 -*-
"""
✅ 完整可运行版本：把 HOG 从“频域乘性滤波”改成“额外 token 分支拼接后做同样 attention”

核心改动（只在 FAModule 内）：
1) 仍然在 token-grid 频域上算 local soft-HOG：hog_local (B, D, Hp, Wp, K)
2) 用 1x1 conv(K->1) 把每个频点的 K-bin HOG 压成 hog_map (B, D, Hp, Wp)
3) 把 hog_map 当作“额外的频域 token 通道”（complex，imag=0），和原频谱拼接：
      Fz_aug = cat([Fz, complex(hog_map,0)], dim=1) -> (B, 2D, Hp, Wp)
      Bz_aug / Cz_aug 同理（必须一致，否则 cross-attn 维度对不上）
4) 频域 RI tokens 的维度从 2D 变成 4D，cross-attn 的 embed_dim 相应变成 4D
5) iFFT 后得到的 map 通道是 2D，把它 fuse 回 D 再走原来的 token MLP + unpatch：
      fuse: Linear(2D -> D)，初始化为 [I, 0]，保证初始完全忽略 HOG 分支（更稳）

其余保持不变：
- 频域 cross-attn (F<-B, F<-C, B<-C) 在 RI tokens 上
- iFFT -> token -> MLP -> unpatch
- X 注入仍然只在 encoder level0，严格按 TriSAFuse
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Time embedding + FiLM (conv)
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


# =========================================================
# Control processor (optional, for cond features)
# =========================================================
class ControlResProc(nn.Module):
    def __init__(self, ch: int, time_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        self.num_blocks = int(num_blocks)
        if self.num_blocks <= 0:
            self.blocks = nn.ModuleList([])
        else:
            self.blocks = nn.ModuleList(
                [ResBlockFiLM(ch, ch, time_ch=time_ch, dropout=dropout) for _ in range(self.num_blocks)]
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# =========================================================
# ViT-style patchify & unpatchify
# =========================================================
class PatchEmbed2D(nn.Module):
    """
    (B,C,H,W) -> (B,N,D), N=(H/P)*(W/P)
    """
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
            raise ValueError(f"PatchEmbed2D requires H,W divisible by patch={p}, got {(h,w)}")
        x = self.proj(x)  # (B, D, Hp, Wp)
        hp, wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
        return x, (hp, wp)


class Unpatch2D(nn.Module):
    """
    (B,N,D) -> (B,C,H,W)
    """
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
            raise ValueError(f"Unpatch2D got N={n} but Hp*Wp={hp*wp}")

        x = self.to_patch(tokens)  # (B, N, C*P*P)
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
# HOG local on token-grid FFT: Z(B,D,Hp,Wp) -> hog_local(B,D,Hp,Wp,K)
# =========================================================
class HOGFromTokenFFT(nn.Module):
    """
    在 token-grid 的频域上算 soft-HOG（不做 Hp,Wp 聚合）
    输出 hog_local: (B, D, Hp, Wp, K)
    """
    def __init__(
        self,
        dim: int,
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        eps: float = 1e-8,
        mag_weight: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.eps = float(eps)
        self.mag_weight = bool(mag_weight)

        k = torch.arange(self.hog_bins).float()
        theta = -math.pi + (2.0 * math.pi) * k / float(self.hog_bins)
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: complex (B, D, Hp, Wp)
        if Z.dim() != 4 or Z.shape[1] != self.dim:
            raise ValueError(f"expects Z (B,{self.dim},Hp,Wp), got {tuple(Z.shape)}")

        mag = Z.abs().clamp_min(self.eps)        # (B,D,Hp,Wp)
        phi = torch.angle(Z)                     # (B,D,Hp,Wp)

        theta = self.theta.view(1, 1, 1, 1, self.hog_bins).to(phi.device, phi.dtype)
        delta = (phi.unsqueeze(-1) - theta + math.pi) % (2.0 * math.pi) - math.pi

        logits = torch.cos(delta) / max(self.hog_tau, 1e-6)
        p = F.softmax(logits, dim=-1)            # (B,D,Hp,Wp,K)

        if self.mag_weight:
            p = p * mag.unsqueeze(-1)

        return p  # (B,D,Hp,Wp,K)


# =========================================================
# Learnable Gaussian (per-channel depthwise conv)
# =========================================================
class LearnableGaussianBlur2d(nn.Module):
    """
    kGauss：逐通道 depthwise（groups=C）
    - w = softplus(raw) 保证非负
    - per-channel normalize: sum=1
    """
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
        g = torch.exp(-(xx**2 + yy**2) / (2.0 * (sigma ** 2)))
        g = g / (g.sum() + 1e-8)
        raw = torch.log(g.clamp_min(1e-12))
        raw = raw.view(1, 1, k, k).repeat(self.channels, 1, 1, 1)
        with torch.no_grad():
            self.raw.copy_(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.channels:
            raise ValueError(f"LearnableGaussianBlur2d expects (B,{self.channels},H,W), got {tuple(x.shape)}")
        w = F.softplus(self.raw)  # non-negative
        w = w / (w.sum(dim=(-1, -2), keepdim=True) + self.eps)  # per-channel sum=1
        pad = self.kernel_size // 2
        return F.conv2d(x, w, bias=None, stride=1, padding=pad, groups=self.channels)


# =========================================================
# Strict U-SA (用于 X 注入)
# =========================================================
class TriSAFuse(nn.Module):
    """
    ✅ 按图（kGauss + 3个独立的1x1 kConv）

    f_anc = Max( f_cond*(1-kGauss), f_cond )
         = max( f_cond - Gauss(f_cond), f_cond )

    g_fore  = Sigmoid( kConv_fore(|f_anc|) )   # (B,1,H,W)
    g_back  = Sigmoid( kConv_back(|f_anc|) )   # (B,1,H,W)
    g_cross = Sigmoid( kConv_cross(|f_anc|) )  # (B,1,H,W)

    f_fore^0 = f_fore^0 + f_fore^0 · g_fore
    f_back^0 = f_back^0 + f_back^0 · g_back + f_fore^0 · g_cross
    """
    def __init__(
        self,
        ch: int,
        gauss_ks: int = 7,
        gauss_sigma_init: float = 1.5,
        init_bias: float = -2.0,
    ):
        super().__init__()
        self.ch = int(ch)
        self.gauss = LearnableGaussianBlur2d(
            channels=self.ch, kernel_size=gauss_ks, sigma_init=gauss_sigma_init
        )

        self.k_fore  = nn.Conv2d(self.ch, 1, 1, bias=True)
        self.k_back  = nn.Conv2d(self.ch, 1, 1, bias=True)
        self.k_cross = nn.Conv2d(self.ch, 1, 1, bias=True)

        nn.init.zeros_(self.k_fore.weight);  nn.init.constant_(self.k_fore.bias,  init_bias)
        nn.init.zeros_(self.k_back.weight);  nn.init.constant_(self.k_back.bias,  init_bias)
        nn.init.zeros_(self.k_cross.weight); nn.init.constant_(self.k_cross.bias, init_bias)

    def forward(self, f_cond: torch.Tensor, f_back: torch.Tensor, f_fore: torch.Tensor):
        if f_cond.shape != f_back.shape or f_back.shape != f_fore.shape:
            raise ValueError("TriSAFuse requires f_cond/f_back/f_fore same shape (B,C,H,W)")

        f_blur = self.gauss(f_cond)
        f_hp = f_cond - f_blur
        f_anc = torch.maximum(f_hp, f_cond)

        a = f_anc.abs()
        g_fore  = torch.sigmoid(self.k_fore(a))
        g_back  = torch.sigmoid(self.k_back(a))
        g_cross = torch.sigmoid(self.k_cross(a))

        f_fore0 = f_fore
        f_fore = f_fore0 + f_fore0 * g_fore
        f_back = f_back + f_back * g_back + f_fore0 * g_cross
        return f_back, f_fore


# =========================================================
# ✅ FA module: HOG -> extra spectrum channels (concat) -> RI tokens cross-attn
# =========================================================
class FAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        hog_bins: int,
        hog_tau: float = 0.35,
        hog_eps: float = 1e-8,
        hog_mag_weight: bool = True,
        fa_patch: int = 8,
        fa_attn_dropout: float = 0.0,
        token_dim: Optional[int] = None,
    ):
        super().__init__()
        self.ch = int(ch)
        self.fa_patch = int(fa_patch)
        if token_dim is None:
            token_dim = int(heads * head_dim)
        self.token_dim = int(token_dim)

        C = self.ch
        D = self.token_dim
        self.time_ch = int(time_ch)
        self.hog_bins = int(hog_bins)

        self.patch_f = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)
        self.patch_b = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)
        self.patch_c = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)

        self.unpatch_f = Unpatch2D(embed_dim=D, out_ch=C, patch=self.fa_patch)
        self.unpatch_b = Unpatch2D(embed_dim=D, out_ch=C, patch=self.fa_patch)

        self.hog_from_fft = HOGFromTokenFFT(
            dim=D,
            hog_bins=hog_bins,
            hog_tau=hog_tau,
            eps=hog_eps,
            mag_weight=hog_mag_weight,
        )

        self.hog_k_to_2_f = nn.Sequential(
                nn.Conv2d(hog_bins, hog_bins, 1, bias=True),  
                nn.SiLU(),
                nn.Conv2d(hog_bins, 2, 1, bias=True),
            )
        self.hog_k_to_2_b = nn.Sequential(
                nn.Conv2d(hog_bins, hog_bins, 1, bias=True),  
                nn.SiLU(),
                nn.Conv2d(hog_bins, 2, 1, bias=True),
            )
        self.hog_k_to_2_c = nn.Sequential(
                nn.Conv2d(hog_bins, hog_bins, 1, bias=True),  
                nn.SiLU(),
                nn.Conv2d(hog_bins, 2, 1, bias=True),
            )

        # GN
        self.hog_norm_f_r = GroupNorm32(D)
        self.hog_norm_f_i = GroupNorm32(D)
        self.hog_norm_b_r = GroupNorm32(D)
        self.hog_norm_b_i = GroupNorm32(D)
        self.hog_norm_c_r = GroupNorm32(D)
        self.hog_norm_c_i = GroupNorm32(D)

        # ✅ 拼接后频谱通道变成 2D，RI token 维度变成 4D
        self.D_aug = 2 * D
        self.RI_dim = 2 * self.D_aug  # = 4D

        # cross-attn 都在 RI token 上
        self.attn_f_b = TokenCrossAttention(dim=self.RI_dim, heads=heads, dropout=fa_attn_dropout)
        self.attn_f_c = TokenCrossAttention(dim=self.RI_dim, heads=heads, dropout=fa_attn_dropout)
        self.attn_b_c = TokenCrossAttention(dim=self.RI_dim, heads=heads, dropout=fa_attn_dropout)

        # iFFT 后 map 通道是 2D，需要 fuse 回 D 才能走原来的 token-MLP + unpatch
        self.fuse_f = nn.Linear(self.D_aug, D, bias=True)
        self.fuse_b = nn.Linear(self.D_aug, D, bias=True)
        self._init_fuse_identity(self.fuse_f, D)
        self._init_fuse_identity(self.fuse_b, D)

        self.mlp_f = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 4 * D),
            nn.SiLU(),
            nn.Linear(4 * D, D),
        )
        self.mlp_b = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 4 * D),
            nn.SiLU(),
            nn.Linear(4 * D, D),
        )

    @staticmethod
    def _init_fuse_identity(fuse: nn.Linear, D: int):
        # fuse: (2D -> D), 初始化成 [I, 0]，bias=0
        with torch.no_grad():
            fuse.weight.zero_()
            fuse.bias.zero_()
            fuse.weight[:, :D] = torch.eye(D, device=fuse.weight.device, dtype=fuse.weight.dtype)

    @staticmethod
    def _tok_to_map(tok: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        b, n, d = tok.shape
        if n != hp * wp:
            raise ValueError(f"_tok_to_map: N={n} but Hp*Wp={hp*wp}")
        return tok.transpose(1, 2).contiguous().view(b, d, hp, wp)

    @staticmethod
    def _map_to_tok(m: torch.Tensor) -> torch.Tensor:
        return m.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _complex_to_ri_tokens(z: torch.Tensor) -> torch.Tensor:
        # z: (B, Dch, Hp, Wp) complex -> ri: (B,2*Dch,Hp,Wp) -> tokens (B,N,2*Dch)
        ri = torch.cat([z.real, z.imag], dim=1)
        return ri.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _ri_tokens_to_complex(tok_ri: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        b, n, d2 = tok_ri.shape
        if n != hp * wp:
            raise ValueError(f"_ri_tokens_to_complex: N={n} but Hp*Wp={hp*wp}")
        d = d2 // 2
        ri = tok_ri.transpose(1, 2).contiguous().view(b, d2, hp, wp)
        return torch.complex(ri[:, :d], ri[:, d:])

    def _hog_ri_from_loc(self, hog_loc, conv):
        B, D, Hp, Wp, K = hog_loc.shape
        x = hog_loc.permute(0,1,4,2,3).contiguous().view(B*D, K, Hp, Wp)   # (B*D,K,Hp,Wp)
        y = conv(x)  # (B*D,2,Hp,Wp)
        y = y.view(B, D, 2, Hp, Wp)  # (B,D,2,Hp,Wp)
        real = y[:, :, 0]  # (B,D,Hp,Wp)
        imag = y[:, :, 1]  # (B,D,Hp,Wp)
        return real, imag

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # patchify -> tokens
        F_tok, hw_p = self.patch_f(fore)
        B_tok, _ = self.patch_b(back)
        C_tok, _ = self.patch_c(cond)
        Hp, Wp = hw_p

        # tokens -> maps (B,D,Hp,Wp)
        F_map = self._tok_to_map(F_tok, Hp, Wp)
        B_map = self._tok_to_map(B_tok, Hp, Wp)
        C_map = self._tok_to_map(C_tok, Hp, Wp)

        # FFT
        Fz = torch.fft.fftshift(torch.fft.fft2(F_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(B_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(C_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        # HOG local (on original D channels)
        Hf_loc = self.hog_from_fft(Fz)  # (B,D,Hp,Wp,K)
        Hb_loc = self.hog_from_fft(Bz)
        Hc_loc = self.hog_from_fft(Cz)

        Hf_r, Hf_i = self._hog_ri_from_loc(Hf_loc, self.hog_k_to_2_f)
        Hb_r, Hb_i = self._hog_ri_from_loc(Hb_loc, self.hog_k_to_2_b)
        Hc_r, Hc_i = self._hog_ri_from_loc(Hc_loc, self.hog_k_to_2_c)

        Hf_r = self.hog_norm_f_r(Hf_r); Hf_i = self.hog_norm_f_i(Hf_i)
        Hb_r = self.hog_norm_b_r(Hb_r); Hb_i = self.hog_norm_b_i(Hb_i)
        Hc_r = self.hog_norm_c_r(Hc_r); Hc_i = self.hog_norm_c_i(Hc_i)

        Hf_z = torch.complex(Hf_r, Hf_i)  # (B,D,Hp,Wp) complex
        Hb_z = torch.complex(Hb_r, Hb_i)
        Hc_z = torch.complex(Hc_r, Hc_i)

        Fz_aug = torch.cat([Fz, Hf_z], dim=1)  # (B,2D,Hp,Wp)
        Bz_aug = torch.cat([Bz, Hb_z], dim=1)
        Cz_aug = torch.cat([Cz, Hc_z], dim=1)

        # spectra -> RI tokens (B,N,4D)
        Ff = self._complex_to_ri_tokens(Fz_aug)
        Bf = self._complex_to_ri_tokens(Bz_aug)
        Cf = self._complex_to_ri_tokens(Cz_aug)

        # three-edge cross-attn
        d_fb = self.attn_f_b(Ff, Bf)
        d_fc = self.attn_f_c(Ff, Cf)
        Ff2 = Ff + d_fb + d_fc

        d_bc = self.attn_b_c(Bf, Cf)
        Bf2 = Bf + d_bc

        # RI -> complex spectra (B,2D,Hp,Wp)
        Fz2 = self._ri_tokens_to_complex(Ff2, Hp, Wp)
        Bz2 = self._ri_tokens_to_complex(Bf2, Hp, Wp)

        # iFFT -> maps (real) (B,2D,Hp,Wp)
        Fz2 = torch.fft.ifftshift(Fz2, dim=(-2, -1))
        Bz2 = torch.fft.ifftshift(Bz2, dim=(-2, -1))
        F_map2 = torch.fft.ifft2(Fz2, dim=(-2, -1), norm="ortho").real
        B_map2 = torch.fft.ifft2(Bz2, dim=(-2, -1), norm="ortho").real

        # maps -> tokens (B,N,2D)
        F_tok2_aug = self._map_to_tok(F_map2)
        B_tok2_aug = self._map_to_tok(B_map2)

        # ✅ fuse(2D->D)，初始相当于只用前 D（原 token），忽略 HOG 分支
        F_tok2 = self.fuse_f(F_tok2_aug)
        B_tok2 = self.fuse_b(B_tok2_aug)

        # token MLP
        F_tok2 = F_tok2 + self.mlp_f(F_tok2)
        B_tok2 = B_tok2 + self.mlp_b(B_tok2)

        # unpatch
        fore_out = self.unpatch_f(F_tok2, hw_p)
        back_out = self.unpatch_b(B_tok2, hw_p)
        return fore_out, back_out


# =========================================================
# Encoder/Decoder stages
# =========================================================
class EncoderStageFiLM(nn.Module):
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


class DecoderStageDirect(nn.Module):
    """
    主干 decoder：输入是 concat(back,fore) 所以通道是 2*chs
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
        for _ in range(int(num_blocks)):
            blocks.append(ResBlockFiLM(ch, out_ch, time_ch=time_ch, dropout=dropout))
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.fuse_norm = GroupNorm32(in_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)
        x = x + self.skip_to_x(skip)
        x = F.silu(self.fuse_norm(x))
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


class DecoderStageSingle(nn.Module):
    """
    cond decoder：单分支（通道是 chs）
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


# =========================================================
# Main UNet (X injection via strict U-SA ONLY at encoder level0)
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
        fa_stacks: int,
        # HOG-on-token-grid
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        hog_mag_weight: bool = True,
        hog_eps: float = 1e-8,
        # Control proc depth
        control_resblocks_per_level: int = 1,
        # FA
        fa_patch: int = 8,
        fa_attn_dropout: float = 0.0,
        fa_token_dim: Optional[int] = None,
        # U-SA strict gauss
        usa_gauss_ks: int = 7,
        usa_gauss_sigma_init: float = 1.5,
        usa_init_bias: float = -10.0,
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

        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.hog_eps = float(hog_eps)
        self.hog_mag_weight = bool(hog_mag_weight)

        self.control_resblocks_per_level = int(control_resblocks_per_level)

        self.fa_patch = int(fa_patch)
        self.fa_attn_dropout = float(fa_attn_dropout)
        self.fa_token_dim = fa_token_dim

        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

        back_in_ch = 1
        self.fore_in_ch = self.xt_in_ch - 1
        cond_in_ch = self.img_in_ch

        # stems
        self.back_stem = nn.Conv2d(back_in_ch, self.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(self.fore_in_ch, self.base_ch, 3, padding=1)
        self.cond_stem = nn.Conv2d(cond_in_ch, self.base_ch, 3, padding=1)

        # channels per level
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

        # main decoder (fb concat)
        self.dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = 2 * self.chs[i]
            in_ch = (2 * self.chs[i + 1]) if i < (n_levels - 1) else (2 * self.chs[i])
            out_ch = 2 * self.chs[i]
            up = i < (n_levels - 1)
            self.dec.append(
                DecoderStageDirect(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    time_ch=self.time_dim,
                    num_blocks=self.num_res_blocks,
                    up=up,
                    dropout=self.dropout,
                )
            )

        # cond decoder（单分支，给 c_dec0）
        self.cond_dec = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_ch = self.chs[i]
            in_ch = (self.chs[i + 1]) if i < (n_levels - 1) else (self.chs[i])
            out_ch = self.chs[i]
            up = i < (n_levels - 1)
            self.cond_dec.append(
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

        self.out_v = nn.Conv2d(2 * self.base_ch, self.xt_in_ch, 3, padding=1)

        # FA levels: last K
        K_levels = min(self.freq_attn_depth, n_levels)
        self.attn_level_set = set(range(n_levels - K_levels, n_levels))

        def make_fa_list(ch: int) -> nn.ModuleList:
            mods = nn.ModuleList()
            for _ in range(self.fa_stacks):
                mods.append(
                    FAModule(
                        ch=ch,
                        time_ch=self.time_dim,
                        heads=self.attn_heads,
                        head_dim=self.attn_head_dim,
                        hog_bins=self.hog_bins,
                        hog_tau=self.hog_tau,
                        hog_eps=self.hog_eps,
                        hog_mag_weight=self.hog_mag_weight,
                        fa_patch=self.fa_patch,
                        fa_attn_dropout=self.fa_attn_dropout,
                        token_dim=self.fa_token_dim,
                    )
                )
            return mods

        self.couple_modules = nn.ModuleDict()
        for i in sorted(self.attn_level_set):
            self.couple_modules[str(i)] = make_fa_list(self.chs[i])

        self.bottleneck_coupler = make_fa_list(self.chs[-1]) if self.use_bottleneck_coupling else None

        # control proc (for cond skips)
        self.ctrl_proc_levels = nn.ModuleList(
            [
                ControlResProc(self.chs[i], time_ch=self.time_dim, num_blocks=self.control_resblocks_per_level, dropout=self.dropout)
                for i in range(n_levels)
            ]
        )

        # X 注入：严格 U-SA，仅 encoder level0
        self.usa_x = TriSAFuse(
            ch=self.chs[0],
            gauss_ks=usa_gauss_ks,
            gauss_sigma_init=usa_gauss_sigma_init,
            init_bias=usa_init_bias,
        )

    def forward(self, x_t: torch.Tensor, image: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
        if isinstance(t, float):
            t = torch.tensor(t, device=x_t.device)
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.expand(x_t.shape[0])
        elif isinstance(t, torch.Tensor) and t.dim() == 1 and t.shape[0] == 1 and x_t.shape[0] > 1:
            t = t.expand(x_t.shape[0])

        t_emb = self.time_mlp(t)

        # split xt
        x_back = x_t[:, 0:1]
        x_fore = x_t[:, 1:]

        # stems
        back0 = self.back_stem(x_back)
        fore0 = self.fore_stem(x_fore)
        cond0 = self.cond_stem(image)

        # (1) cond encoder -> cond_skips (processed)
        cond_skips: List[torch.Tensor] = []
        cond = cond0
        for level, enc_c in enumerate(self.cond_enc):
            c_skip = enc_c(cond, t_emb)
            c_skip_proc = self.ctrl_proc_levels[level](c_skip, t_emb)
            cond_skips.append(c_skip_proc)
            cond = enc_c.down(c_skip_proc) if enc_c.down is not None else c_skip_proc

        # (2) cond decoder -> c_dec0 (full-res)
        c_h = cond
        for cdec, c_skip in zip(self.cond_dec, reversed(cond_skips)):
            c_h = cdec(c_h, c_skip, t_emb)
        c_dec0 = c_h  # (B, chs[0], H, W)

        # (3) fore/back encoder with X injection ONLY at level0 via U-SA
        fore_skips: List[torch.Tensor] = []
        back_skips: List[torch.Tensor] = []

        back = back0
        fore = fore0

        for level, (enc_b, enc_f) in enumerate(zip(self.back_enc, self.fore_enc)):
            b_skip = enc_b(back, t_emb)
            f_skip = enc_f(fore, t_emb)

            if level == 0:
                b_skip, f_skip = self.usa_x(f_cond=c_dec0, f_back=b_skip, f_fore=f_skip)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)

            b_main, f_main = b_skip, f_skip
            c_main = cond_skips[level]

            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_main, b_main = fa(fore=f_main, back=b_main, cond=c_main, t_emb=t_emb)

            back = enc_b.down(b_main) if enc_b.down is not None else b_main
            fore = enc_f.down(f_main) if enc_f.down is not None else f_main

        # bottleneck coupling
        if self.bottleneck_coupler is not None:
            cond_bot = cond_skips[-1]
            for fa in self.bottleneck_coupler:
                fore, back = fa(fore=fore, back=back, cond=cond_bot, t_emb=t_emb)

        # (4) main decoder
        h = torch.cat([back, fore], dim=1)
        for dec, f_skip, b_skip in zip(self.dec, reversed(fore_skips), reversed(back_skips)):
            skip_fb = torch.cat([b_skip, f_skip], dim=1)
            h = dec(h, skip_fb, t_emb)

        h = F.silu(h)
        return self.out_v(h)


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
    freq_attn_depth: int = 1,
    use_bottleneck_coupling: bool = True,
    fa_stacks: int = 4,
    hog_bins: int = 8,
    hog_tau: float = 0.35,
    hog_mag_weight: bool = True,
    hog_eps: float = 1e-8,
    control_resblocks_per_level: int = 1,
    fa_patch: int = 8,
    fa_attn_dropout: float = 0.0,
    fa_token_dim: Optional[int] = None,
    usa_gauss_ks: int = 7,
    usa_gauss_sigma_init: float = 1.5,
    usa_init_bias: float = -10.0,
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
        fa_stacks=fa_stacks,
        hog_bins=hog_bins,
        hog_tau=hog_tau,
        hog_mag_weight=hog_mag_weight,
        hog_eps=hog_eps,
        control_resblocks_per_level=control_resblocks_per_level,
        fa_patch=fa_patch,
        fa_attn_dropout=fa_attn_dropout,
        fa_token_dim=fa_token_dim,
        usa_gauss_ks=usa_gauss_ks,
        usa_gauss_sigma_init=usa_gauss_sigma_init,
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
        freq_attn_depth=2,
        use_bottleneck_coupling=True,
        fa_stacks=2,
        hog_bins=8,
        hog_tau=0.35,
        hog_mag_weight=True,
        hog_eps=1e-8,
        control_resblocks_per_level=1,
        fa_patch=8,
        fa_attn_dropout=0.0,
        fa_token_dim=None,
        usa_gauss_ks=7,
        usa_gauss_sigma_init=1.5,
        usa_init_bias=-10.0,
    )

    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v = net(x_t, img, t)
    print("v:", v.shape)