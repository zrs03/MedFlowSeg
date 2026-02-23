# -*- coding: utf-8 -*-
"""
✅ 修改版：把 X 注入改为【cond encoder 首层(level0)通过 TriSAFuse 注入到 back/fore】，
decoder 仍然通过 zero conv 注入（ctrl_zero_dec）——其余都不变：
- FA：token 频域 + 方案A + 自token+自hog token拼接 + 三条边 cross-attn + 只回写前N 不变
- Token-Time-FiLM：feature patchify 与 hog patchify 两边都加（不变）
- ControlNet/UNet 主干不变（除了 encoder 注入逻辑这处改动）

建议：build_model 里设置
  control_inject_to_encoder=False
  control_inject_to_bottleneck=False
  control_inject_to_decoder=True
"""

from __future__ import annotations

import math
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
# ZeroConv (ControlNet 核心)
# -----------------------------
class ZeroConv2d(nn.Module):
    """1x1 conv with ZERO init (weight/bias = 0)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# -----------------------------
# Control feature processor: N x ResBlockFiLM per level
# -----------------------------
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
# ✅ Learnable Gaussian + TriSAFuse (X 注入)
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
        w = F.softplus(self.raw)
        w = w / (w.sum(dim=(-1, -2), keepdim=True) + self.eps)
        pad = self.kernel_size // 2
        return F.conv2d(x, w, bias=None, stride=1, padding=pad, groups=self.channels)


class TriSAFuse(nn.Module):
    """
    ✅ Strict U-SA：用 cond 的高频/锚点生成三个 gate，对 fore/back 做 X 注入
    """
    def __init__(
        self,
        ch: int,
        gauss_ks: int = 7,
        gauss_sigma_init: float = 1.5,
        init_bias: float = -10.0,
    ):
        super().__init__()
        self.ch = int(ch)
        self.gauss = LearnableGaussianBlur2d(self.ch, kernel_size=gauss_ks, sigma_init=gauss_sigma_init)

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
# ✅ ViT-style patchify & unpatchify
# =========================================================
class PatchEmbed2D(nn.Module):
    """
    ViT-style patchify + linear projection by Conv2d(stride=patch).
    Input : (B, C, H, W)
    Output: (B, N, D) where N=(H/P)*(W/P)
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
        x = self.proj(x)  # (B, D, H/P, W/P)
        hp, wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, D)
        return x, (hp, wp)


class Unpatch2D(nn.Module):
    """
    token -> (B, C, H, W) 反 patchify
    方式：Linear(D -> C*P*P) 后 reshape 回原图（非重叠）
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
        x = x.view(b, hp, wp, self.out_ch, p, p)  # (B, Hp, Wp, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, Hp, p, Wp, p)
        x = x.view(b, self.out_ch, hp * p, wp * p)  # (B, C, H, W)
        return x


# =========================================================
# ✅ Token Time-FiLM (PatchEmbed 后对 tokens 做调制)
# =========================================================
class TokenTimeFiLM(nn.Module):
    """
    对 tokens 做 time conditioning:
      x: (B,N,D)
      t_emb: (B,time_ch)
      out = (1+gamma)*x + beta
    """
    def __init__(self, dim: int, time_ch: int):
        super().__init__()
        self.dim = int(dim)
        self.to_gb = nn.Sequential(nn.SiLU(), nn.Linear(time_ch, 2 * self.dim))
        # ✅ identity init
        nn.init.zeros_(self.to_gb[-1].weight)
        nn.init.zeros_(self.to_gb[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gb(t_emb).chunk(2, dim=1)  # (B,D)
        gamma = gamma[:, None, :]
        beta  = beta[:, None, :]
        return (1.0 + gamma) * x + beta


# =========================================================
# ✅ Token Cross-Attention (ViT-style)
# =========================================================
class TokenCrossAttention(nn.Module):
    """
    Cross-attention in token space:
      q_tokens: (B, Nq, D)
      kv_tokens: (B, Nk, D)
    Output:
      (B, Nq, D)
    """
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


# -----------------------------
# Dense frequency-domain "HOG" + magnitude
# -----------------------------
def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


class FreqDenseHOG(nn.Module):
    """
    输入 x: (B,C,H,W)
    输出 hog_map: (B, C*K, H, W)  —— 不改变空间分辨率
    """
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
# ✅ FA module: PatchEmbed → FFT → (freq cross-attn) → iFFT → MLP → Unpatch
#   + 方案A：hog_fore / hog_back / hog_cond
#   + 拼接与三条 cross-attn 边
#   + ✅ 新增：feature tokens + hog tokens 都加 t 的 Token-Time-FiLM
# =========================================================
class FAModule(nn.Module):
    """
    输入：
      fore/back/cond: (B, C, H, W)
      hog_fore/hog_back/hog_cond: (B, C*K, H, W)

    关键：
      - attention 在频域：FFT 后转 RI tokens (B,N,2D) 做 cross-attn
      - iFFT 后再做 MLP：在空间 tokens (B,N,D) 上做 FFN
      - HOG（方案A）：HOG patchify 得 (B,N,D)，再 Linear 投到 (B,N,2D)
      - 拼接规则：谁当 Q/KV，就只拼接自己 feature+自己 hog
      - 三条边：
          d_fb = attn(FQ <- BK)
          d_fc = attn(FQ <- CK)
          d_bc = attn(BK <- CK)
      - 回写频谱只取前 N（feature RI tokens），hog tokens 仅作条件
    """
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        hog_bins: int,
        noise_strength: float = 1.0,
        fa_patch: int = 4,
        fa_attn_dropout: float = 0.0,
        token_dim: Optional[int] = None,
    ):
        super().__init__()
        self.ch = int(ch)
        self.hog_bins = int(hog_bins)
        self.noise_strength = float(noise_strength)

        self.fa_patch = int(fa_patch)
        if token_dim is None:
            token_dim = int(heads * head_dim)
        self.token_dim = int(token_dim)

        C = self.ch
        D = self.token_dim
        hog_ch = C * self.hog_bins

        # 1) patchify 主特征：C -> D
        self.patch_f = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)
        self.patch_b = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)
        self.patch_c = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)

        # 2) patchify HOG：C*K -> D（空间域 tokens）
        self.patch_hf = PatchEmbed2D(in_ch=hog_ch, embed_dim=D, patch=self.fa_patch)
        self.patch_hb = PatchEmbed2D(in_ch=hog_ch, embed_dim=D, patch=self.fa_patch)
        self.patch_hc = PatchEmbed2D(in_ch=hog_ch, embed_dim=D, patch=self.fa_patch)

        # ✅ Time-FiLM on patch tokens (feature)
        self.time_film_f = TokenTimeFiLM(D, time_ch)
        self.time_film_b = TokenTimeFiLM(D, time_ch)
        self.time_film_c = TokenTimeFiLM(D, time_ch)

        # ✅ Time-FiLM on patch tokens (hog)
        self.time_film_hf = TokenTimeFiLM(D, time_ch)
        self.time_film_hb = TokenTimeFiLM(D, time_ch)
        self.time_film_hc = TokenTimeFiLM(D, time_ch)

        # 3) HOG tokens -> RI tokens (2D)
        self.hog_to_ri_f = nn.Linear(D, 2 * D, bias=True)
        self.hog_to_ri_b = nn.Linear(D, 2 * D, bias=True)
        self.hog_to_ri_c = nn.Linear(D, 2 * D, bias=True)

        # 4) 频域 cross-attn：embed_dim = 2D
        self.attn_f_freq = TokenCrossAttention(dim=2 * D, heads=heads, dropout=fa_attn_dropout)
        self.attn_b_freq = TokenCrossAttention(dim=2 * D, heads=heads, dropout=fa_attn_dropout)

        # 5) iFFT 后 MLP：embed_dim = D
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

        # 6) unpatch：D -> C
        self.unpatch_f = Unpatch2D(embed_dim=D, out_ch=C, patch=self.fa_patch)
        self.unpatch_b = Unpatch2D(embed_dim=D, out_ch=C, patch=self.fa_patch)

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
        ri = torch.cat([z.real, z.imag], dim=1)           # (B, 2D, Hp, Wp)
        return ri.flatten(2).transpose(1, 2).contiguous() # (B, N, 2D)

    @staticmethod
    def _ri_tokens_to_complex(tok_ri: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        b, n, d2 = tok_ri.shape
        if n != hp * wp:
            raise ValueError(f"_ri_tokens_to_complex: N={n} but Hp*Wp={hp*wp}")
        d = d2 // 2
        ri = tok_ri.transpose(1, 2).contiguous().view(b, d2, hp, wp)  # (B, 2D, Hp, Wp)
        return torch.complex(ri[:, :d], ri[:, d:])                     # (B, D, Hp, Wp) complex

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        hog_fore: torch.Tensor,
        hog_back: torch.Tensor,
        hog_cond: torch.Tensor,
        t_emb: torch.Tensor,
    ):
        # 1) PatchEmbed: (B,C,H,W) -> (B,N,D)
        F_tok, hw_p = self.patch_f(fore)
        B_tok, _    = self.patch_b(back)
        C_tok, _    = self.patch_c(cond)

        # ✅ Time-FiLM on feature tokens
        F_tok = self.time_film_f(F_tok, t_emb)
        B_tok = self.time_film_b(B_tok, t_emb)
        C_tok = self.time_film_c(C_tok, t_emb)

        Hp, Wp = hw_p
        N = F_tok.shape[1]

        # tokens -> token-grid maps: (B,D,Hp,Wp)
        F_map = self._tok_to_map(F_tok, Hp, Wp)
        B_map = self._tok_to_map(B_tok, Hp, Wp)
        C_map = self._tok_to_map(C_tok, Hp, Wp)

        # 2) FFT on token-grid
        Fz = torch.fft.fftshift(torch.fft.fft2(F_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(B_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(C_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        # 3) complex -> RI tokens
        Ff_tok = self._complex_to_ri_tokens(Fz)  # (B,N,2D)
        Bf_tok = self._complex_to_ri_tokens(Bz)  # (B,N,2D)
        Cf_tok = self._complex_to_ri_tokens(Cz)  # (B,N,2D)

        # 4) HOG tokens
        Hf_tok, _ = self.patch_hf(hog_fore)  # (B,N,D)
        Hb_tok, _ = self.patch_hb(hog_back)  # (B,N,D)
        Hc_tok, _ = self.patch_hc(hog_cond)  # (B,N,D)

        # ✅ Time-FiLM on hog tokens
        Hf_tok = self.time_film_hf(Hf_tok, t_emb)
        Hb_tok = self.time_film_hb(Hb_tok, t_emb)
        Hc_tok = self.time_film_hc(Hc_tok, t_emb)

        Hf2_tok = self.hog_to_ri_f(Hf_tok)   # (B,N,2D)
        Hb2_tok = self.hog_to_ri_b(Hb_tok)   # (B,N,2D)
        Hc2_tok = self.hog_to_ri_c(Hc_tok)   # (B,N,2D)

        # 5) 拼接规则 + 三条边
        FQ = torch.cat([Ff_tok, Hf2_tok], dim=1)  # (B,2N,2D)
        BK = torch.cat([Bf_tok, Hb2_tok], dim=1)  # (B,2N,2D)
        CK = torch.cat([Cf_tok, Hc2_tok], dim=1)  # (B,2N,2D)

        d_fb = self.attn_f_freq(FQ, BK)
        d_fc = self.attn_f_freq(FQ, CK)
        FQ2 = FQ + d_fb + d_fc

        d_bc = self.attn_b_freq(BK, CK)
        BK2 = BK + d_bc

        # 只回写前 N
        Ff_tok2 = FQ2[:, :N, :]
        Bf_tok2 = BK2[:, :N, :]

        # 6) iFFT
        Fz2 = self._ri_tokens_to_complex(Ff_tok2, Hp, Wp)
        Bz2 = self._ri_tokens_to_complex(Bf_tok2, Hp, Wp)

        Fz2 = torch.fft.ifftshift(Fz2, dim=(-2, -1))
        Bz2 = torch.fft.ifftshift(Bz2, dim=(-2, -1))

        F_map2 = torch.fft.ifft2(Fz2, dim=(-2, -1), norm="ortho").real
        B_map2 = torch.fft.ifft2(Bz2, dim=(-2, -1), norm="ortho").real

        # map -> tokens
        F_tok2 = self._map_to_tok(F_map2)
        B_tok2 = self._map_to_tok(B_map2)

        # 7) MLP
        F_tok2 = F_tok2 + self.mlp_f(F_tok2)
        B_tok2 = B_tok2 + self.mlp_b(B_tok2)

        # 8) Unpatch
        fore_out = self.unpatch_f(F_tok2, hw_p)
        back_out = self.unpatch_b(B_tok2, hw_p)

        return fore_out, back_out


# -----------------------------
# Encoder stage (blocks only; downsample handled outside)
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

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# -----------------------------
# Decoder stage
# -----------------------------
class DecoderStageDirect(nn.Module):
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

        self.fuse_norm = GroupNorm32(in_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            x = self.up(x)

        x = x + self.skip_to_x(skip)
        x = F.silu(self.fuse_norm(x))

        for blk in self.blocks:
            x = blk(x, t_emb)
        return x


# -----------------------------
# Main Network
# -----------------------------
class FMForeBackCondDeepKTriFFTSingleDecoderResUNet_ControlNet(nn.Module):
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
        # --- HOG config (dense) ---
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        hog_mag_weight: bool = True,
        hog_eps: float = 1e-8,
        # --- ControlNet injections ---
        control_inject_to_encoder: bool = False,      # ✅ 默认关掉（encoder改用TriSAFuse）
        control_inject_to_decoder: bool = True,       # ✅ decoder zero conv 注入保留
        control_inject_to_bottleneck: bool = False,   # ✅ 默认关掉（按你要求“其他都不变”，这里只建议关）
        control_resblocks_per_level: int = 1,
        # --- FA patch/token settings ---
        fa_patch: int = 4,
        fa_attn_dropout: float = 0.0,
        fa_token_dim: Optional[int] = None,
        # --- X injection (TriSAFuse) ---
        usa_gauss_ks: int = 7,
        usa_gauss_sigma_init: float = 1.5,
        usa_init_bias: float = -10.0,
    ):
        super().__init__()

        if xt_in_ch < 2:
            raise ValueError("xt_in_ch must be >=2 (channel0 background, remaining foreground).")
        if freq_attn_depth < 0:
            raise ValueError("freq_attn_depth must be >= 0.")
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

        self.control_inject_to_encoder = bool(control_inject_to_encoder)
        self.control_inject_to_decoder = bool(control_inject_to_decoder)
        self.control_inject_to_bottleneck = bool(control_inject_to_bottleneck)
        self.control_resblocks_per_level = int(control_resblocks_per_level)

        # dense HOG config
        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.hog_eps = float(hog_eps)

        # FA config
        self.fa_patch = int(fa_patch)
        self.fa_attn_dropout = float(fa_attn_dropout)
        self.fa_token_dim = fa_token_dim

        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

        back_in_ch = 1
        self.fore_in_ch = self.xt_in_ch - 1
        cond_in_ch = self.img_in_ch

        # stems (主干)
        self.back_stem = nn.Conv2d(back_in_ch, self.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(self.fore_in_ch, self.base_ch, 3, padding=1)

        # control stem
        self.cond_stem = nn.Conv2d(cond_in_ch, self.base_ch, 3, padding=1)

        # Dense HOG extractor
        self.freq_hog = FreqDenseHOG(
            hog_bins=self.hog_bins,
            hog_tau=self.hog_tau,
            eps=self.hog_eps,
            mag_weight=hog_mag_weight,
        )

        self.chs = [self.base_ch * m for m in self.ch_mult]
        n_levels = len(self.chs)

        # ✅ X 注入：cond encoder 首层(level0)用 TriSAFuse
        self.usa_x = TriSAFuse(
            ch=self.chs[0],
            gauss_ks=usa_gauss_ks,
            gauss_sigma_init=usa_gauss_sigma_init,
            init_bias=usa_init_bias,
        )

        # FA levels: last K
        K_levels = min(self.freq_attn_depth, n_levels)
        self.attn_level_set = set(range(n_levels - K_levels, n_levels))

        self.hog_norm_levels = nn.ModuleList([
            GroupNorm32(num_channels=self.chs[i] * self.hog_bins) if (i in self.attn_level_set) else nn.Identity()
            for i in range(n_levels)
        ])
        self.hog_norm_mid = GroupNorm32(num_channels=self.chs[-1] * self.hog_bins) if self.use_bottleneck_coupling else nn.Identity()

        # encoders: 主干 back/fore + Control cond
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
                        noise_strength=self.noise_strength,
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

        # decoder (shared, only skip_fb)
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

        self.out_v = nn.Conv2d(2 * self.base_ch, self.xt_in_ch, 3, padding=1)

        # per-level c_skip processing
        self.ctrl_proc_levels = nn.ModuleList(
            [
                ControlResProc(
                    self.chs[i],
                    time_ch=self.time_dim,
                    num_blocks=self.control_resblocks_per_level,
                    dropout=self.dropout,
                )
                for i in range(n_levels)
            ]
        )
        self.ctrl_proc_mid = ControlResProc(
            self.chs[-1],
            time_ch=self.time_dim,
            num_blocks=self.control_resblocks_per_level,
            dropout=self.dropout,
        )

        # ControlNet ZeroConv injections（encoder/bottleneck保留模块，但默认不用）
        self.ctrl_zero_b_enc = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])
        self.ctrl_zero_f_enc = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])
        self.ctrl_zero_b_mid = ZeroConv2d(self.chs[-1], self.chs[-1])
        self.ctrl_zero_f_mid = ZeroConv2d(self.chs[-1], self.chs[-1])

        # ✅ decoder 注入保留（你要求：decoder 通过 zero conv 注入）
        self.ctrl_zero_dec = nn.ModuleList([ZeroConv2d(self.chs[i], 2 * self.chs[i]) for i in range(n_levels)])

    def forward(self, x_t: torch.Tensor, image: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
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

        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            b_skip = enc_b(back, t_emb)   # (B, C, H, W)
            f_skip = enc_f(fore, t_emb)   # (B, C, H, W)
            c_skip = enc_c(cond, t_emb)   # (B, C, H, W)

            c_skip_proc = self.ctrl_proc_levels[level](c_skip, t_emb)

            # ✅ X 注入改为：cond encoder 首层(level0)通过 TriSAFuse 注入到 b/f
            if level == 0:
                b_skip, f_skip = self.usa_x(f_cond=c_skip_proc, f_back=b_skip, f_fore=f_skip)

            # ❌ encoder zero conv 注入默认不使用（如你想保留开关，可把下面取消注释）
            if self.control_inject_to_encoder:
                b_skip = b_skip + self.ctrl_zero_b_enc[level](c_skip_proc)
                f_skip = f_skip + self.ctrl_zero_f_enc[level](c_skip_proc)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c_skip_proc)

            b_main, f_main, c_main = b_skip, f_skip, c_skip_proc

            # ✅ 在进入 FA 前：dense HOG + GN
            if level in self.attn_level_set:
                hog_f = self.freq_hog(f_main)  # (B, C*K, H, W)
                hog_b = self.freq_hog(b_main)  # (B, C*K, H, W)
                hog_c = self.freq_hog(c_main)  # (B, C*K, H, W)

                hog_f = self.hog_norm_levels[level](hog_f)
                hog_b = self.hog_norm_levels[level](hog_b)
                hog_c = self.hog_norm_levels[level](hog_c)

                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_main, b_main = fa(
                        f_main, b_main, c_main,
                        hog_fore=hog_f, hog_back=hog_b, hog_cond=hog_c,
                        t_emb=t_emb
                    )

            back = enc_b.down(b_main) if enc_b.down is not None else b_main
            fore = enc_f.down(f_main) if enc_f.down is not None else f_main
            cond = enc_c.down(c_main) if enc_c.down is not None else c_main

        cond_mid = self.ctrl_proc_mid(cond, t_emb)

        if self.control_inject_to_bottleneck:
            back = back + self.ctrl_zero_b_mid(cond_mid)
            fore = fore + self.ctrl_zero_f_mid(cond_mid)

        # ✅ bottleneck 同理：dense HOG + GN（方案A：fore/back/cond）
        if self.bottleneck_coupler is not None:
            hog_f = self.freq_hog(fore)
            hog_b = self.freq_hog(back)
            hog_c = self.freq_hog(cond_mid)

            hog_f = self.hog_norm_mid(hog_f)
            hog_b = self.hog_norm_mid(hog_b)
            hog_c = self.hog_norm_mid(hog_c)

            for fa in self.bottleneck_coupler:
                fore, back = fa(
                    fore, back, cond_mid,
                    hog_fore=hog_f, hog_back=hog_b, hog_cond=hog_c,
                    t_emb=t_emb
                )

        h = torch.cat([back, fore], dim=1)

        for dec_idx, (dec, f_skip, b_skip, c_skip_proc) in enumerate(
            zip(self.dec, reversed(fore_skips), reversed(back_skips), reversed(cond_skips))
        ):
            skip_fb = torch.cat([b_skip, f_skip], dim=1)
            h = dec(h, skip_fb, t_emb)

            # ✅ decoder 通过 zero conv 注入（保留）
            if self.control_inject_to_decoder:
                level = (len(self.chs) - 1) - dec_idx
                h = h + self.ctrl_zero_dec[level](c_skip_proc)

        h = F.silu(h)
        v = self.out_v(h)
        return v


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
    fa_stacks: int = 4,
    noise_strength: float = 1.0,
    # dense HOG
    hog_bins: int = 8,
    hog_tau: float = 0.35,
    hog_mag_weight: bool = True,
    # control
    control_inject_to_encoder: bool = False,      # ✅ encoder用TriSAFuse，默认关
    control_inject_to_decoder: bool = True,       # ✅ decoder zero conv 注入开
    control_inject_to_bottleneck: bool = False,   # ✅ 默认关
    control_resblocks_per_level: int = 1,
    # FA token
    fa_patch: int = 4,
    fa_attn_dropout: float = 0.0,
    fa_token_dim: Optional[int] = None,
    # X injection (TriSAFuse)
    usa_gauss_ks: int = 7,
    usa_gauss_sigma_init: float = 1.5,
    usa_init_bias: float = -10.0,
) -> FMForeBackCondDeepKTriFFTSingleDecoderResUNet_ControlNet:
    return FMForeBackCondDeepKTriFFTSingleDecoderResUNet_ControlNet(
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
        hog_eps=1e-8,
        control_inject_to_encoder=control_inject_to_encoder,
        control_inject_to_decoder=control_inject_to_decoder,
        control_inject_to_bottleneck=control_inject_to_bottleneck,
        control_resblocks_per_level=control_resblocks_per_level,
        fa_patch=fa_patch,
        fa_attn_dropout=fa_attn_dropout,
        fa_token_dim=fa_token_dim,
        usa_gauss_ks=usa_gauss_ks,
        usa_gauss_sigma_init=usa_gauss_sigma_init,
        usa_init_bias=usa_init_bias,
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
        control_inject_to_encoder=False,     # ✅ encoder不用zero conv
        control_inject_to_decoder=True,      # ✅ decoder保留zero conv注入
        control_inject_to_bottleneck=False,  # ✅ bottleneck默认不注入
        control_resblocks_per_level=1,
        fa_patch=4,
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