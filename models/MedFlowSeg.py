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
# Dense frequency-domain "HOG" + magnitude
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


# -----------------------------
# (方案B) Frequency-domain direct head
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
        return out


# -----------------------------
# FA module (one-pass) + direct freq update
# cond: as KV only
# -----------------------------
class FAModule(nn.Module):
    def __init__(
        self,
        ch: int,
        time_ch: int,
        heads: int,
        head_dim: int,
        noise_strength: float = 1.0,
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

    def forward(self, fore: torch.Tensor, back: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor):
        Fz = torch.fft.fftshift(torch.fft.fft2(fore, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(back, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Cz = torch.fft.fftshift(torch.fft.fft2(cond, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        Fri = self._to_ri(Fz)
        Bri = self._to_ri(Bz)
        Cri = self._to_ri(Cz)

        f_in = torch.cat([Fri, Cri], dim=1)
        b_in = torch.cat([Bri, Cri], dim=1)

        Fri += self.direct_head_f(f_in, t_emb)
        Bri += self.direct_head_b(b_in, t_emb)

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

        return fore_out, back_out


# -----------------------------
# Encoder stage (blocks only; downsample handled outside) ✅
# -----------------------------
class EncoderStageFiLM(nn.Module):
    """
    ✅ 只负责 ResBlocks，返回 skip（同分辨率特征）
    ✅ downsample 放到外层做：x = stage.down(skip)
    """
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
        return x  # skip (same resolution)


# -----------------------------
# Decoder stage (NO cond direct add; ControlNet uses zero-injection instead)
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
# Main Network (ControlNet style injection + optional per-level control ResBlocks)
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
        # --- HOG routing params ---
        hog_bins: int = 8,
        hog_tau: float = 0.35,
        hog_mag_weight: bool = True,
        hog_eps: float = 1e-8,
        hog_softmax_tau: float = 0.10,
        # --- 方案E: bin 权重 head ---
        hog_bin_weight_hidden: int = 0,
        # --- ControlNet injections ---
        control_inject_to_encoder: bool = True,
        control_inject_to_decoder: bool = True,
        control_inject_to_bottleneck: bool = True,
        # --- NEW: per-level control processing depth ---
        control_resblocks_per_level: int = 1,
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

        self.control_inject_to_encoder = bool(control_inject_to_encoder)
        self.control_inject_to_decoder = bool(control_inject_to_decoder)
        self.control_inject_to_bottleneck = bool(control_inject_to_bottleneck)
        self.control_resblocks_per_level = int(control_resblocks_per_level)

        # HOG routing config
        self.hog_bins = int(hog_bins)
        self.hog_tau = float(hog_tau)
        self.hog_eps = float(hog_eps)
        self.hog_softmax_tau = float(hog_softmax_tau)

        self.time_mlp = TimeMLP(time_dim=self.time_dim, hidden_dim=self.time_dim)

        back_in_ch = 1
        self.fore_in_ch = self.xt_in_ch - 1
        cond_in_ch = self.img_in_ch

        # stems (主干)
        self.back_stem = nn.Conv2d(back_in_ch, self.base_ch, 3, padding=1)
        self.fore_stem = nn.Conv2d(self.fore_in_ch, self.base_ch, 3, padding=1)

        # control stem
        self.cond_stem = nn.Conv2d(cond_in_ch, self.base_ch, 3, padding=1)

        # HOG extractor
        self.freq_hog = FreqDenseHOG(
            hog_bins=self.hog_bins,
            hog_tau=self.hog_tau,
            eps=self.hog_eps,
            mag_weight=hog_mag_weight,
        )

        # 方案E：ctx只用 t_emb -> wk
        out_dim = self.fore_in_ch * self.hog_bins
        if hog_bin_weight_hidden and hog_bin_weight_hidden > 0:
            self.hog_bin_weight_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_dim, int(hog_bin_weight_hidden)),
                nn.SiLU(),
                nn.Linear(int(hog_bin_weight_hidden), out_dim),
            )
        else:
            self.hog_bin_weight_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_dim, out_dim),
            )

        self.chs = [self.base_ch * m for m in self.ch_mult]
        n_levels = len(self.chs)

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

        # per-level c_skip processing (ResBlockFiLM x N)
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

        # ControlNet ZeroConv injections
        self.ctrl_zero_b_enc = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])
        self.ctrl_zero_f_enc = nn.ModuleList([ZeroConv2d(self.chs[i], self.chs[i]) for i in range(n_levels)])

        self.ctrl_zero_b_mid = ZeroConv2d(self.chs[-1], self.chs[-1])
        self.ctrl_zero_f_mid = ZeroConv2d(self.chs[-1], self.chs[-1])

        self.ctrl_zero_dec = nn.ModuleList([ZeroConv2d(self.chs[i], 2 * self.chs[i]) for i in range(n_levels)])

    # -----------------------------
    # Scheme-1 HOG routing (方案E: ctx只用 t_emb)
    # ✅ 修改：只对 back 做 share，fore 不变
    # -----------------------------
    def _hog_routing_fuse(self, x_back: torch.Tensor, x_fore: torch.Tensor, t_emb: torch.Tensor):
        b, cf, h, w = x_fore.shape
        k = self.hog_bins
        eps = self.hog_eps

        if cf != self.fore_in_ch:
            raise ValueError(f"Unexpected fore channel count: got {cf}, expected {self.fore_in_ch}")

        hog_back_map = self.freq_hog(x_back)
        hog_fore_map = self.freq_hog(x_fore)

        pb = hog_back_map.view(b, 1, k, h, w).permute(0, 1, 3, 4, 2).contiguous()
        pf = hog_fore_map.view(b, cf, k, h, w).permute(0, 1, 3, 4, 2).contiguous()

        pb = pb / (pb.sum(dim=-1, keepdim=True) + eps)
        pf = pf / (pf.sum(dim=-1, keepdim=True) + eps)

        wk_logits = self.hog_bin_weight_head(t_emb).view(b, cf, k)
        wk = torch.softmax(wk_logits, dim=-1).view(b, cf, 1, 1, k)

        sim_c = (pf * pb * wk).sum(dim=-1)
        alpha_c = torch.clamp(sim_c, 0.0, 1.0)

        # FFT
        Fz = torch.fft.fftshift(torch.fft.fft2(x_fore, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        Bz = torch.fft.fftshift(torch.fft.fft2(x_back, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

        # channel weight for background share strength
        tau = max(self.hog_softmax_tau, 1e-6)
        w_ch = torch.softmax(alpha_c / tau, dim=1)
        alpha_back = torch.clamp((w_ch * alpha_c).sum(dim=1, keepdim=True), 0.0, 1.0)

        # ✅ 只更新背景频域：B 共享一部分来自 F_mix；F 完全保持不变
        F_mix = (w_ch.to(Fz.dtype) * Fz).sum(dim=1, keepdim=True)  # (B,1,H,W)
        B_shared = 0.5 * (Bz + F_mix)  # (B,1,H,W)
        Bz2 = (1.0 - alpha_back).to(Bz.dtype) * Bz + alpha_back.to(Bz.dtype) * B_shared

        # iFFT
        Fz2 = torch.fft.ifftshift(Fz, dim=(-2, -1))          # fore unchanged
        Bz2 = torch.fft.ifftshift(Bz2, dim=(-2, -1))         # back updated

        x_fore2 = torch.fft.ifft2(Fz2, dim=(-2, -1), norm="ortho").real
        x_back2 = torch.fft.ifft2(Bz2, dim=(-2, -1), norm="ortho").real
        return x_back2, x_fore2

    def freeze_base(self):
        base_modules = [
            self.back_stem, self.fore_stem,
            self.back_enc, self.fore_enc,
            self.dec, self.out_v,
            self.couple_modules,
            self.bottleneck_coupler,
        ]
        for m in base_modules:
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = False

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

        # HOG routing (fore unchanged; back updated)
        x_back, x_fore = self._hog_routing_fuse(x_back, x_fore, t_emb=t_emb)

        back = self.back_stem(x_back)
        fore = self.fore_stem(x_fore)

        cond = self.cond_stem(image)

        fore_skips: List[torch.Tensor] = []
        back_skips: List[torch.Tensor] = []
        cond_skips: List[torch.Tensor] = []

        for level, (enc_b, enc_f, enc_c) in enumerate(zip(self.back_enc, self.fore_enc, self.cond_enc)):
            b_skip = enc_b(back, t_emb)
            f_skip = enc_f(fore, t_emb)
            c_skip = enc_c(cond, t_emb)

            c_skip_proc = self.ctrl_proc_levels[level](c_skip, t_emb)

            if self.control_inject_to_encoder:
                b_skip = b_skip + self.ctrl_zero_b_enc[level](c_skip_proc)
                f_skip = f_skip + self.ctrl_zero_f_enc[level](c_skip_proc)

            back_skips.append(b_skip)
            fore_skips.append(f_skip)
            cond_skips.append(c_skip_proc)

            b_main, f_main, c_main = b_skip, f_skip, c_skip_proc

            if level in self.attn_level_set:
                fa_list: nn.ModuleList = self.couple_modules[str(level)]
                for fa in fa_list:
                    f_main, b_main = fa(f_main, b_main, c_main, t_emb=t_emb)

            back = enc_b.down(b_main) if enc_b.down is not None else b_main
            fore = enc_f.down(f_main) if enc_f.down is not None else f_main
            cond = enc_c.down(c_main) if enc_c.down is not None else c_main

        cond_mid = self.ctrl_proc_mid(cond, t_emb)

        if self.control_inject_to_bottleneck:
            back = back + self.ctrl_zero_b_mid(cond_mid)
            fore = fore + self.ctrl_zero_f_mid(cond_mid)

        if self.bottleneck_coupler is not None:
            for fa in self.bottleneck_coupler:
                fore, back = fa(fore, back, cond_mid, t_emb=t_emb)

        h = torch.cat([back, fore], dim=1)

        for dec_idx, (dec, f_skip, b_skip, c_skip_proc) in enumerate(
            zip(self.dec, reversed(fore_skips), reversed(back_skips), reversed(cond_skips))
        ):
            skip_fb = torch.cat([b_skip, f_skip], dim=1)
            h = dec(h, skip_fb, t_emb)

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
    hog_bins: int = 8,
    hog_tau: float = 0.35,
    hog_mag_weight: bool = True,
    hog_softmax_tau: float = 0.10,
    hog_bin_weight_hidden: int = 0,
    control_inject_to_encoder: bool = True,
    control_inject_to_decoder: bool = True,
    control_inject_to_bottleneck: bool = True,
    control_resblocks_per_level: int = 1,
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
        hog_softmax_tau=hog_softmax_tau,
        hog_bin_weight_hidden=hog_bin_weight_hidden,
        control_inject_to_encoder=control_inject_to_encoder,
        control_inject_to_decoder=control_inject_to_decoder,
        control_inject_to_bottleneck=control_inject_to_bottleneck,
        control_resblocks_per_level=control_resblocks_per_level,
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
        hog_bin_weight_hidden=0,
        control_inject_to_encoder=True,
        control_inject_to_decoder=True,
        control_inject_to_bottleneck=True,
        control_resblocks_per_level=1,
    )

    x_t = torch.randn(2, 4, 256, 256)
    img = torch.randn(2, 1, 256, 256)
    t = torch.rand(2)

    v = net(x_t, img, t)
    print("v:", v.shape)