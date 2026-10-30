# SPDX-License-Identifier: GPL-3.0-or-later
"""Dequantization network (Kim et al., IEEE SPL 2022, Sec. II-C, Fig. 1, Eq. 3).

bits [B, T, S] -> per-frame embedding c_FFN [B, 56, T'].

- Multi-stream CNN: 2-D convolutions over (frame, bit) with kernels 3x5, 5x5, 7x5
  (j = 2 layers, 8 -> 16 channels) for local embeddings, and a 1-D convolution over time
  with all S bits as channels (kernel 5) for a global embedding.
- Multi-stream self-attention: every stream gets its own MHSA (k = 3 heads, d_head = 4,
  d_model = 16, d_ff = 16) with sinusoidal positional encoding and post-LayerNorm.
- The four streams are concatenated per frame and mapped by FFN1 (1024) and FFN2 (56).

Additions (all off by default, so the default is the paper's network):
- group_frames: treat g consecutive frames as one unit of g*S bits. With an interleaver
  spanning g frames this puts bits that belong together back into one token set.
- bit_order: a fixed reordering of the unit's bits (e.g. from analysis.bit_stats), so
  the 5-bit-wide kernels see related bits next to each other again.
- soft_perm: a learnable permutation (Sinkhorn) in front of the CNN.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(dim // 2, device=device, dtype=torch.float32)[None, :]
    ang = pos / (10000 ** (2 * i / dim))
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # [n, dim]


class MHSA(nn.Module):
    """Multi-head attention with d_head * heads != d_model allowed (3 x 4 = 12 vs 16)."""

    def __init__(self, d_model=16, heads=3, d_head=4):
        super().__init__()
        self.h, self.dh = heads, d_head
        self.qkv = nn.Linear(d_model, 3 * heads * d_head)
        self.out = nn.Linear(heads * d_head, d_model)

    def forward(self, x):  # [N, L, d]
        N, L, _ = x.shape
        q, k, v = self.qkv(x).view(N, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(N, L, self.h * self.dh))


class AttnBlock(nn.Module):
    """Eq. 3: c~ = MHSA(PE(c)); c' = LN(c + c~); c'' = LN(c' + FFN(c'))."""

    def __init__(self, d_model=16, heads=3, d_head=4, d_ff=16):
        super().__init__()
        self.attn = MHSA(d_model, heads, d_head)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, c, pe):
        c1 = self.ln1(c + self.attn(c + pe))
        return self.ln2(c1 + self.ffn(c1))


def pe2d(T: int, S: int, d: int, device) -> torch.Tensor:
    """[T, S, d]: first half encodes the frame index, second half the bit index."""
    return torch.cat([sinusoid(T, d // 2, device)[:, None, :].expand(T, S, d // 2),
                      sinusoid(S, d // 2, device)[None, :, :].expand(T, S, d // 2)], dim=-1)


class SoftPermutation(nn.Module):
    def __init__(self, n: int, iters: int = 20, tau: float = 0.1):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(n, n) * 0.01)
        self.iters, self.tau = iters, tau

    def matrix(self):
        z = self.logits / self.tau
        for _ in range(self.iters):
            z = z - z.logsumexp(dim=1, keepdim=True)
            z = z - z.logsumexp(dim=0, keepdim=True)
        return z.exp()

    def forward(self, x):  # [..., n] -> [..., n]
        return x @ self.matrix()


class DequantNet(nn.Module):
    def __init__(self, n_bits: int, group_frames: int = 1, kernels=((3, 5), (5, 5), (7, 5)),
                 channels=(8, 16), kernel_1d: int = 5, heads: int = 3, d_head: int = 4,
                 d_ff: int = 16, ffn_dims=(1024, 56), attn_axis: str = "bit",
                 bit_order=None, soft_perm: bool = False, aux_mels: int = 0):
        super().__init__()
        self.g = group_frames
        S = n_bits * group_frames
        self.S = S
        d = channels[-1]
        self.attn_axis = attn_axis
        if bit_order is not None:
            assert sorted(bit_order) == list(range(S)), "bit_order must be a permutation of the unit's bits"
        self.register_buffer("bit_order", torch.as_tensor(bit_order if bit_order is not None else range(S)),
                             persistent=True)
        self.soft_perm = SoftPermutation(S) if soft_perm else None

        def conv2d_stack(k):
            layers, cin = [], 1
            for c in channels:
                layers += [nn.Conv2d(cin, c, k, padding=(k[0] // 2, k[1] // 2)), nn.LeakyReLU(0.2)]
                cin = c
            return nn.Sequential(*layers)

        self.local = nn.ModuleList(conv2d_stack(tuple(k)) for k in kernels)
        g_layers, cin = [], S
        for c in channels:
            g_layers += [nn.Conv1d(cin, c, kernel_1d, padding=kernel_1d // 2), nn.LeakyReLU(0.2)]
            cin = c
        self.globl = nn.Sequential(*g_layers)
        self.attn = nn.ModuleList(AttnBlock(d, heads, d_head, d_ff) for _ in range(len(kernels) + 1))
        self.ffn1 = nn.Linear(len(kernels) * S * d + d, ffn_dims[0])
        self.ffn2 = nn.Linear(ffn_dims[0], ffn_dims[1])
        self.out_dim = ffn_dims[1]
        # Optional auxiliary log-mel head on the FFN1 features (generic speech target,
        # computed from the clean source; never a codec parameter).
        self.aux_mels = aux_mels
        self.aux = nn.Linear(ffn_dims[0], aux_mels) if aux_mels else None

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return self.ffn2(self.features(bits)).transpose(1, 2)  # [B, 56, Tg]

    def forward_with_aux(self, bits: torch.Tensor):
        if self.aux is None:
            raise ValueError("This frontend has no auxiliary mel head (aux_mels=0)")
        h = self.features(bits)
        return self.ffn2(h).transpose(1, 2), self.aux(h).transpose(1, 2)

    def features(self, bits: torch.Tensor) -> torch.Tensor:  # [B, T, S] -> [B, Tg, ffn_dims[0]]
        B, T, n = bits.shape
        assert T % self.g == 0, f"crop length {T} must be a multiple of group_frames={self.g}"
        x = bits.reshape(B, T // self.g, n * self.g)[..., self.bit_order]
        if self.soft_perm is not None:
            x = self.soft_perm(x)
        Tg, S = x.shape[1], x.shape[2]
        x = x + 1.0  # paper: 1 is added to the input bits
        d = self.attn[0].ln1.normalized_shape[0]

        feats = []
        pe_ts = pe2d(Tg, S, d, x.device)
        for conv, blk in zip(self.local, self.attn[:-1]):
            c = conv(x[:, None])  # [B, d, Tg, S]
            c = c.permute(0, 2, 3, 1)  # [B, Tg, S, d]
            if self.attn_axis == "bit":
                c = blk(c.reshape(B * Tg, S, d), pe_ts.reshape(1, Tg, S, d).expand(B, -1, -1, -1).reshape(B * Tg, S, d))
            else:  # full: all (frame, bit) tokens of the crop
                c = blk(c.reshape(B, Tg * S, d), pe_ts.reshape(1, Tg * S, d))
            feats.append(c.reshape(B, Tg, S * d))
        cg = self.globl(x.transpose(1, 2)).transpose(1, 2)  # [B, Tg, d]
        feats.append(self.attn[-1](cg, sinusoid(Tg, d, x.device)[None]))
        return F.leaky_relu(self.ffn1(torch.cat(feats, dim=-1)), 0.2)
