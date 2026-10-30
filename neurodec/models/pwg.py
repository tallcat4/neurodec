# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel WaveGAN (Yamamoto et al. 2020) generator / discriminator, plus the paper's
up-sampling layer and gated activation unit (Eq. 4) between the frame embedding and the
generator."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Upsample(nn.Module):
    """PWG's conv-in + stretch/conv2d upsampling, scales multiply to the hop size."""

    def __init__(self, channels: int, scales=(4, 4, 5, 2), conv_in_kernel: int = 5):
        super().__init__()
        self.conv_in = nn.Conv1d(channels, channels, conv_in_kernel, padding=conv_in_kernel // 2, bias=False)
        self.scales = list(scales)
        self.convs = nn.ModuleList()
        for s in scales:
            conv = nn.Conv2d(1, 1, (1, 2 * s + 1), padding=(0, s), bias=False)
            nn.init.constant_(conv.weight, 1.0 / (2 * s + 1))
            self.convs.append(conv)

    def forward(self, c):  # [B, C, T] -> [B, C, T * prod(scales)]
        c = self.conv_in(c)[:, None]
        for s, conv in zip(self.scales, self.convs):
            c = conv(F.interpolate(c, scale_factor=(1, s), mode="nearest"))
        return c[:, 0]


class GAU(nn.Module):
    """Eq. 4: c = tanh(W_f * c_up) . sigmoid(W_g * c_up)."""

    def __init__(self, cin: int, cout: int, kernel: int = 1):
        super().__init__()
        self.f = nn.Conv1d(cin, cout, kernel, padding=kernel // 2)
        self.g = nn.Conv1d(cin, cout, kernel, padding=kernel // 2)

    def forward(self, c):
        return torch.tanh(self.f(c)) * torch.sigmoid(self.g(c))


class ResBlock(nn.Module):
    def __init__(self, res, gate, skip, aux, kernel, dilation):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(res, gate, kernel, padding=(kernel - 1) // 2 * dilation, dilation=dilation))
        self.aux = weight_norm(nn.Conv1d(aux, gate, 1, bias=False))
        self.out = weight_norm(nn.Conv1d(gate // 2, res, 1))
        self.skip = weight_norm(nn.Conv1d(gate // 2, skip, 1))

    def forward(self, x, c):
        h = self.conv(x) + self.aux(c)
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        return (x + self.out(h)) * math.sqrt(0.5), self.skip(h)


class PWGGenerator(nn.Module):
    def __init__(self, aux=80, layers=30, stacks=3, res=64, gate=128, skip=64, kernel=3):
        super().__init__()
        per = layers // stacks
        self.first = weight_norm(nn.Conv1d(1, res, 1))
        self.blocks = nn.ModuleList(ResBlock(res, gate, skip, aux, kernel, 2 ** (i % per)) for i in range(layers))
        self.last = nn.Sequential(nn.ReLU(), weight_norm(nn.Conv1d(skip, skip, 1)), nn.ReLU(),
                                  weight_norm(nn.Conv1d(skip, 1, 1)))

    def forward(self, z, c):  # z [B, 1, N], c [B, aux, N]
        x, skips = self.first(z), 0.0
        for blk in self.blocks:
            x, s = blk(x, c)
            skips = skips + s
        return self.last(skips * math.sqrt(1.0 / len(self.blocks)))[:, 0]


class PWGDiscriminator(nn.Module):
    def __init__(self, layers=10, channels=64, kernel=3, initialization="torch_default"):
        super().__init__()
        if initialization not in ("torch_default", "pwg"):
            raise ValueError(f"Unknown discriminator initialization: {initialization}")
        mods, cin = [], 1
        for i in range(layers - 1):
            dil = 1 if i == 0 else i
            mods += [weight_norm(nn.Conv1d(cin, channels, kernel, padding=(kernel - 1) // 2 * dil, dilation=dil)),
                     nn.LeakyReLU(0.2)]
            cin = channels
        mods.append(weight_norm(nn.Conv1d(cin, 1, kernel, padding=(kernel - 1) // 2)))
        self.net = nn.Sequential(*mods)
        if initialization == "pwg":
            # PWG's public Conv1d uses Kaiming-normal weights and zero biases.
            # Initialize both weight-norm tensors so the effective weight equals
            # the sampled weight. Preserve the RNG stream used for G's noise.
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                for layer in self.net:
                    if isinstance(layer, nn.Conv1d):
                        v = layer.parametrizations.weight.original1
                        nn.init.kaiming_normal_(v, nonlinearity="relu")
                        layer.parametrizations.weight.original0.copy_(v.norm(dim=(1, 2), keepdim=True))
                        layer.bias.zero_()

    def forward(self, x):  # [B, N] -> [B, N] logits
        return self.net(x[:, None])[:, 0]
