# -*- coding: utf-8 -*-
"""
✅ 你要的改法（完整代码）：

核心改动：
1) HOG 不再拼接到 feature RI tokens 里；
2) 先在 token-grid 上做 FFT 得到频谱 Z:(B,D,Hp,Wp)；
3) 从同一个 Z 上做 soft-HOG，得到 hog_tok:(B,N,D)，与 feature_tok:(B,N,D) 完全对齐（N=Hp*Wp）；
4) 用 hog_tok 生成每个 token 的 FiLM 参数 (gamma,beta)，直接调控 feature_tok：
      feat_tok := (1+gamma)*feat_tok + beta
   —— 这一步在 FFT 之前做（也可以选在 iFFT 后再做一次，这里我只做一次，最干净）
5) cross-attn 仍是三条边（F<-B, F<-C, B<-C），但只对“feature 频域 RI tokens”做（不再有 hog 拼接序列）
6) ControlNet 只在 encoder 第一层、decoder 最后一层注入（bottleneck 不注入）

超参：
- fa_patch 固定默认 8（build_model 里默认就是 8），不再有 hog_patch

注意：
- 因为不再拼接 hog，attention 序列长度从 2N 回到 N，显存会更省。
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
# ControlNet: ZeroConv + optional processor
# =========================================================
class ZeroConv2d(nn.Module):
    """1x1 conv with ZERO init (weight/bias = 0)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


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
# Helpers: angle wrap
# =========================================================
def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


# =========================================================
# ✅ HOG-from-token-grid FFT: Z(B,D,Hp,Wp) -> hog_tok(B,N,D)
# =========================================================
class HOGFromTokenFFT(nn.Module):
    """
    在 token-grid 频谱上做 soft-HOG：
      输入：Z complex (B, D, Hp, Wp)  (已经 fftshift)
      输出：hog_tok (B, N, D), N=Hp*Wp
    做法：
      - 取 phi = angle(Z), mag = |Z|
      - 对每个像素(频率点)做 K-bin soft assignment
      - 统计（sum over K）得到每个频率点的“方向聚合响应”
      - 输出一个 D 通道的 HOG token：hog_d(u,v) = sum_k p_k(u,v)  (可加 mag 加权)
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
        theta = -math.pi + (2.0 * math.pi) * k / float(self.hog_bins)
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (B, D, Hp, Wp) complex, shifted
        mag = Z.abs().clamp_min(self.eps)                 # (B,D,Hp,Wp)
        phi = torch.angle(Z)                               # (B,D,Hp,Wp)

        theta = self.theta.view(1, 1, 1, 1, self.hog_bins).to(device=phi.device, dtype=phi.dtype)
        delta = _wrap_pi(phi.unsqueeze(-1) - theta)       # (B,D,Hp,Wp,K)

        logits = torch.cos(delta) / max(self.hog_tau, 1e-6)
        p = F.softmax(logits, dim=-1)                     # (B,D,Hp,Wp,K)
        if self.mag_weight:
            p = p * mag.unsqueeze(-1)

        # 聚合 K -> 1（每个频率点一个标量响应），保持 D 通道
        hog_map = p.sum(dim=-1)                            # (B,D,Hp,Wp)

        # map -> tokens
        hog_tok = hog_map.flatten(2).transpose(1, 2).contiguous()  # (B,N,D)
        return hog_tok


# =========================================================
# ✅ 用 hog_tok 调控 feat_tok 的 FiLM（逐 token）
# =========================================================
class HogTokenFiLM(nn.Module):
    """
    feat_tok: (B,N,D)
    hog_tok : (B,N,D)  (已对齐)
    t_emb   : (B,T)

    输出：feat_tok' = (1+gamma)*feat_tok + beta
    其中 gamma,beta 来自 concat([hog_tok, t_emb]) 的 MLP，逐 token 生成。
    """
    def __init__(self, dim: int, time_ch: int):
        super().__init__()
        self.dim = int(dim)
        self.time_ch = int(time_ch)
        self.to_gb = nn.Sequential(
            nn.LayerNorm(self.dim + self.time_ch),
            nn.Linear(self.dim + self.time_ch, 2 * self.dim),
        )
        # identity init：开始时不改动 feature
        nn.init.zeros_(self.to_gb[-1].weight)
        nn.init.zeros_(self.to_gb[-1].bias)

    def forward(self, feat_tok: torch.Tensor, hog_tok: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, n, d = feat_tok.shape
        if hog_tok.shape != (b, n, d):
            raise ValueError(f"HogTokenFiLM requires hog_tok shape {(b,n,d)}, got {tuple(hog_tok.shape)}")
        if t_emb.dim() != 2 or t_emb.shape[0] != b:
            raise ValueError("t_emb must be (B,time_ch)")

        t = t_emb[:, None, :].expand(b, n, t_emb.shape[1])
        cond = torch.cat([hog_tok, t], dim=-1)  # (B,N,D+T)
        gamma, beta = self.to_gb(cond).chunk(2, dim=-1)  # (B,N,D)
        return (1.0 + gamma) * feat_tok + beta


# =========================================================
# ✅ FA module: (B,D,Hp,Wp) FFT + HOG(F) FiLM(feature) + freq cross-attn
# =========================================================
class FAModule(nn.Module):
    """
    - PatchEmbed: (B,C,H,W) -> feat_tok (B,N,D)
    - reshape -> (B,D,Hp,Wp) -> FFT2 on (Hp,Wp)
    - 从同一个 Z 上提取 hog_tok(B,N,D)
    - hog_tok 通过 FiLM 调控 feat_tok（在 FFT 前）
    - 再 FFT -> RI tokens (B,N,2D)
    - 三条边 cross-attn 仅在 feature RI 上做：
        F <- B, F <- C,  B <- C
    - iFFT -> feat_tok2 -> FFN -> Unpatch -> (B,C,H,W)
    """
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

        self.patch_f = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)
        self.patch_b = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)
        self.patch_c = PatchEmbed2D(in_ch=C, embed_dim=D, patch=self.fa_patch)

        self.unpatch_f = Unpatch2D(embed_dim=D, out_ch=C, patch=self.fa_patch)
        self.unpatch_b = Unpatch2D(embed_dim=D, out_ch=C, patch=self.fa_patch)

        # 从 token-grid FFT 直接提 hog_tok (B,N,D)
        self.hog_from_fft = HOGFromTokenFFT(
            hog_bins=hog_bins,
            hog_tau=hog_tau,
            eps=hog_eps,
            mag_weight=hog_mag_weight,
        )

        # hog_tok -> FiLM(feature_tok)
        self.film_f = HogTokenFiLM(dim=D, time_ch=time_ch)
        self.film_b = HogTokenFiLM(dim=D, time_ch=time_ch)
        self.film_c = HogTokenFiLM(dim=D, time_ch=time_ch)

        # freq cross-attn on RI tokens, dim=2D
        self.attn_f_b = TokenCrossAttention(dim=2 * D, heads=heads, dropout=fa_attn_dropout)
        self.attn_f_c = TokenCrossAttention(dim=2 * D, heads=heads, dropout=fa_attn_dropout)
        self.attn_b_c = TokenCrossAttention(dim=2 * D, heads=heads, dropout=fa_attn_dropout)

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
    def _tok_to_map(tok: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        # (B,N,D) -> (B,D,Hp,Wp)
        b, n, d = tok.shape
        if n != hp * wp:
            raise ValueError(f"_tok_to_map: N={n} but Hp*Wp={hp*wp}")
        return tok.transpose(1, 2).contiguous().view(b, d, hp, wp)

    @staticmethod
    def _map_to_tok(m: torch.Tensor) -> torch.Tensor:
        # (B,D,Hp,Wp) -> (B,N,D)
        return m.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _complex_to_ri_tokens(z: torch.Tensor) -> torch.Tensor:
        # z: (B,D,Hp,Wp) complex -> (B,N,2D)
        ri = torch.cat([z.real, z.imag], dim=1)                 # (B,2D,Hp,Wp)
        return ri.flatten(2).transpose(1, 2).contiguous()       # (B,N,2D)

    @staticmethod
    def _ri_tokens_to_complex(tok_ri: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        # tok_ri: (B,N,2D) -> complex z: (B,D,Hp,Wp)
        b, n, d2 = tok_ri.shape
        if n != hp * wp:
            raise ValueError(f"_ri_tokens_to_complex: N={n} but Hp*Wp={hp*wp}")
        d = d2 // 2
        ri = tok_ri.transpose(1, 2).contiguous().view(b, d2, hp, wp)  # (B,2D,Hp,Wp)
        return torch.complex(ri[:, :d], ri[:, d:])                    # (B,D,Hp,Wp)

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ----------------------------
        # 1) PatchEmbed -> tokens
        # ----------------------------
        F_tok, hw_p = self.patch_f(fore)  # (B,N,D)
        B_tok, _ = self.patch_b(back)
        C_tok, _ = self.patch_c(cond)

        Hp, Wp = hw_p
        N = F_tok.shape[1]
        D = F_tok.shape[2]

        # ----------------------------
        # 2) token-grid maps -> FFT（先算一次，为了拿 hog_tok）
        # ----------------------------
        F_map0 = self._tok_to_map(F_tok, Hp, Wp)  # (B,D,Hp,Wp)
        B_map0 = self._tok_to_map(B_tok, Hp, Wp)
        C_map0 = self._tok_to_map(C_tok, Hp, Wp)

        Fz0 = torch.fft.fftshift(torch.fft.fft2(F_map0, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz0 = torch.fft.fftshift(torch.fft.fft2(B_map0, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz0 = torch.fft.fftshift(torch.fft.fft2(C_map0, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        # ----------------------------
        # 3) 从频谱提 hog_tok（与 N 对齐）
        # ----------------------------
        Hf_tok = self.hog_from_fft(Fz0)  # (B,N,D)
        Hb_tok = self.hog_from_fft(Bz0)
        Hc_tok = self.hog_from_fft(Cz0)

        # ----------------------------
        # 4) hog_tok -> FiLM 调控 feature_tok（不拼接）
        # ----------------------------
        F_tok = self.film_f(F_tok, Hf_tok, t_emb)
        B_tok = self.film_b(B_tok, Hb_tok, t_emb)
        C_tok = self.film_c(C_tok, Hc_tok, t_emb)

        # ----------------------------
        # 5) 调控后再做 FFT（用于 cross-attn）
        # ----------------------------
        F_map = self._tok_to_map(F_tok, Hp, Wp)
        B_map = self._tok_to_map(B_tok, Hp, Wp)
        C_map = self._tok_to_map(C_tok, Hp, Wp)

        Fz = torch.fft.fftshift(torch.fft.fft2(F_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(B_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(C_map, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        Ff = self._complex_to_ri_tokens(Fz)  # (B,N,2D)
        Bf = self._complex_to_ri_tokens(Bz)
        Cf = self._complex_to_ri_tokens(Cz)

        # ----------------------------
        # 6) 三条边 cross-attn（只对 feature RI）
        # ----------------------------
        d_fb = self.attn_f_b(Ff, Bf)
        d_fc = self.attn_f_c(Ff, Cf)
        Ff2 = Ff + d_fb + d_fc

        d_bc = self.attn_b_c(Bf, Cf)
        Bf2 = Bf + d_bc

        # ----------------------------
        # 7) iFFT -> tokens -> FFN -> unpatch
        # ----------------------------
        Fz2 = self._ri_tokens_to_complex(Ff2, Hp, Wp)
        Bz2 = self._ri_tokens_to_complex(Bf2, Hp, Wp)

        Fz2 = torch.fft.ifftshift(Fz2, dim=(-2, -1))
        Bz2 = torch.fft.ifftshift(Bz2, dim=(-2, -1))

        F_map2 = torch.fft.ifft2(Fz2, dim=(-2, -1), norm="ortho").real  # (B,D,Hp,Wp)
        B_map2 = torch.fft.ifft2(Bz2, dim=(-2, -1), norm="ortho").real

        F_tok2 = self._map_to_tok(F_map2)  # (B,N,D)
        B_tok2 = self._map_to_tok(B_map2)

        F_tok2 = F_tok2 + self.mlp_f(F_tok2)
        B_tok2 = B_tok2 + self.mlp_b(B_tok2)

        fore_out = self.unpatch_f(F_tok2, hw_p)  # (B,C,H,W)
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


# =========================================================
# Main UNet
# =========================================================
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
        # HOG-on-token-grid
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        hog_mag_weight: bool = True,
        hog_eps: float = 1e-8,
        # Control
        control_resblocks_per_level: int = 1,
        control_inject_to_encoder: bool = True,
        control_inject_to_decoder: bool = True,
        # FA
        fa_patch: int = 8,
        fa_attn_dropout: float = 0.0,
        fa_token_dim: Optional[int] = None,
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
        self.control_inject_to_encoder = bool(control_inject_to_encoder)
        self.control_inject_to_decoder = bool(control_inject_to_decoder)

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

        # decoder
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

        # Control proc
        self.ctrl_proc_levels = nn.ModuleList(
            [
                ControlResProc(self.chs[i], time_ch=self.time_dim, num_blocks=self.control_resblocks_per_level, dropout=self.dropout)
                for i in range(n_levels)
            ]
        )

        # Control ZeroConv (we only *use* level0 enc + level0 dec in forward)
        self.ctrl_zero_b_enc = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])
        self.ctrl_zero_f_enc = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])
        self.ctrl_zero_dec = nn.ModuleList([ZeroConv2d(self.chs[i], 2 * self.chs[i]) for i in range(n_levels)])

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
        back = self.back_stem(x_back)
        fore = self.fore_stem(x_fore)
        cond = self.cond_stem(image)

        fore_skips: List[torch.Tensor] = []
        back_skips: List[torch.Tensor] = []
        cond_skips: List[torch.Tensor] = []

        # =========================
        # Encoder
        # =========================
        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            b_skip = enc_b(back, t_emb)
            f_skip = enc_f(fore, t_emb)
            c_skip = enc_c(cond, t_emb)

            c_skip_proc = self.ctrl_proc_levels[level](c_skip, t_emb)

            # ✅ ONLY encoder first level inject
            if self.control_inject_to_encoder and level == 0:
                b_skip = b_skip + self.ctrl_zero_b_enc[level](c_skip_proc)
                f_skip = f_skip + self.ctrl_zero_f_enc[level](c_skip_proc)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c_skip_proc)

            b_main, f_main, c_main = b_skip, f_skip, c_skip_proc

            # FA coupling on selected levels
            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_main, b_main = fa(
                        fore=f_main,
                        back=b_main,
                        cond=c_main,
                        t_emb=t_emb,
                    )

            back = enc_b.down(b_main) if enc_b.down is not None else b_main
            fore = enc_f.down(f_main) if enc_f.down is not None else f_main
            cond = enc_c.down(c_main) if enc_c.down is not None else c_main

        # bottleneck coupling（不做 control 注入）
        if self.bottleneck_coupler is not None:
            for fa in self.bottleneck_coupler:
                fore, back = fa(fore=fore, back=back, cond=cond, t_emb=t_emb)

        h = torch.cat([back, fore], dim=1)

        # =========================
        # Decoder
        # =========================
        for dec_idx, (dec, f_skip, b_skip, c_skip_proc) in enumerate(
            zip(self.dec, reversed(fore_skips), reversed(back_skips), reversed(cond_skips))
        ):
            skip_fb = torch.cat([b_skip, f_skip], dim=1)
            h = dec(h, skip_fb, t_emb)

            level = (len(self.chs) - 1) - dec_idx
            # ✅ ONLY decoder last stage inject (level==0)
            if self.control_inject_to_decoder and level == 0:
                h = h + self.ctrl_zero_dec[level](c_skip_proc)

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
    freq_attn_depth: int = 2,
    use_bottleneck_coupling: bool = True,
    fa_stacks: int = 4,
    # HOG-on-token-grid
    hog_bins: int = 8,
    hog_tau: float = 0.35,
    hog_mag_weight: bool = True,
    # Control
    control_resblocks_per_level: int = 1,
    control_inject_to_encoder: bool = True,
    control_inject_to_decoder: bool = True,
    # FA
    fa_patch: int = 8,  # ✅ 你指定：统一为 8
    fa_attn_dropout: float = 0.0,
    fa_token_dim: Optional[int] = None,
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
        hog_bins=hog_bins,
        hog_tau=hog_tau,
        hog_mag_weight=hog_mag_weight,
        hog_eps=1e-8,
        control_resblocks_per_level=control_resblocks_per_level,
        control_inject_to_encoder=control_inject_to_encoder,
        control_inject_to_decoder=control_inject_to_decoder,
        fa_patch=fa_patch,
        fa_attn_dropout=fa_attn_dropout,
        fa_token_dim=fa_token_dim,
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
        control_inject_to_encoder=True,
        control_inject_to_decoder=True,
        control_resblocks_per_level=1,
        fa_patch=8,  # ✅ 固定 8
        fa_attn_dropout=0.0,
        fa_token_dim=None,
    )

    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v = net(x_t, img, t)
    print("v:", v.shape)