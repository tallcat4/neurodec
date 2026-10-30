# SPDX-License-Identifier: GPL-3.0-or-later
"""Codec-agnostic bit-token frontend (a modern replacement for the paper's dequantizer).

bits [B, T, S] -> conditioning [B, out_dim, T] (+ optional auxiliary log-mel [B, aux_mels, T]).

- Every bit position s gets its own learned embedding for value 0 and value 1 ([S, 2, d]),
  so nothing assumes that neighbouring bit positions are related (an interleaver makes
  them arbitrary) and no bit ordering is needed.
- Bits of one frame interact through a small pre-LN Transformer over the S tokens
  ("transformer" mixer) or a 2-layer MLP over the raw bit vector ("mlp" mixer); the
  frame vector keeps position identity through a concat + linear read-out.
- A burst-position embedding (frame index modulo the public burst size) tells the model
  where a frame sits inside a burst; crops and decoding chunks start on burst boundaries.
- Temporal context comes from ConvNeXt-style 1-D blocks over frames (kernel 7 each), i.e.
  about +-3*layers frames, without any codec-specific parameter layout.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .istft import SpectralBlock


class BitTokenFrontend(nn.Module):
    def __init__(self, n_bits: int, block_frames: int = 1, d_token: int = 64, token_layers: int = 2,
                 token_heads: int = 4, width: int = 256, layers: int = 6, expansion: int = 3,
                 out_dim: int = 56, aux_mels: int = 0, mixer: str = "transformer", mlp_hidden: int = 1024,
                 dropout: float = 0.0, f0_head: bool = False, ap_bands: int = 0):
        super().__init__()
        if mixer not in ("transformer", "mlp"):
            raise ValueError(f"Unknown mixer: {mixer}")
        if n_bits < 1 or block_frames < 1 or layers < 1 or width < 1:
            raise ValueError("n_bits, block_frames, layers and width must be positive")
        self.n_bits, self.block_frames, self.mixer_kind = n_bits, block_frames, mixer
        self.out_dim, self.aux_mels = out_dim, aux_mels
        if mixer == "transformer":
            if d_token % token_heads:
                raise ValueError("d_token must be divisible by token_heads")
            self.embed = nn.Parameter(torch.randn(n_bits, 2, d_token) * 0.02)
            layer = nn.TransformerEncoderLayer(d_token, token_heads, 4 * d_token, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.tokens = nn.TransformerEncoder(layer, token_layers, enable_nested_tensor=False)
            self.read = nn.Linear(n_bits * d_token, width)
        else:
            self.embed = None
            self.tokens = None
            self.read = nn.Sequential(nn.Linear(n_bits, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, width))
        self.burst = nn.Parameter(torch.zeros(block_frames, width))
        self.blocks = nn.Sequential(*(SpectralBlock(width, expansion, 1 / layers) for _ in range(layers)))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.out = nn.Linear(width, out_dim)
        self.aux = nn.Linear(width, aux_mels) if aux_mels else None
        # Optional periodicity head: [log F0 (Hz), voicing logit] per frame. The teacher is
        # pyworld F0 of the clean source (scripts.precompute_f0); inference never needs it.
        # Channels: [log F0, voicing logit, band-aperiodicity logits x ap_bands].
        self.ap_bands = ap_bands if f0_head else 0
        self.f0 = nn.Linear(width, 2 + self.ap_bands) if f0_head else None
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, bits: torch.Tensor) -> torch.Tensor:  # [B, T, S] -> [B, T, width]
        B, T, S = bits.shape
        if S != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} bits per frame, got {S}")
        if self.mixer_kind == "transformer":
            value = (bits > 0.5).long()  # 0/1 (flipped bits from the BER augmentation stay 0/1)
            tok = self.embed[torch.arange(S, device=bits.device)[None, None, :], value]  # [B, T, S, d]
            tok = self.tokens(tok.reshape(B * T, S, -1))
            h = self.read(tok.reshape(B, T, -1))
        else:
            h = self.read(bits * 2.0 - 1.0)
        h = h + self.burst[torch.arange(T, device=bits.device) % self.block_frames]
        h = self.blocks(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(bits)).transpose(1, 2)

    def forward_with_aux(self, bits: torch.Tensor, with_f0: bool = False):
        if self.aux is None:
            raise ValueError("This frontend has no auxiliary mel head (aux_mels=0)")
        f = self.features(bits)
        cond, mel = self.out(f).transpose(1, 2), self.aux(f).transpose(1, 2)
        if with_f0:
            if self.f0 is None:
                raise ValueError("This frontend has no F0 head (f0_head=false)")
            return cond, mel, self.f0(f).transpose(1, 2)  # [B, 2 + ap_bands, T]
        return cond, mel
