# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact periodic-waveform and complex-STFT critics for generic speech.

Uses the MPD/MRD principles of HiFi-GAN/Vocos, with reduced channel widths and
whole-band spectral critics rather than Vocos's separate frequency subbands.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .pwg import PWGDiscriminator


def convolution(cin, cout, kernel, stride=(1, 1), padding=(0, 0)):
    layer = nn.Conv2d(cin, cout, kernel, stride, padding)
    nn.init.kaiming_normal_(layer.weight, a=.1, nonlinearity='leaky_relu')
    nn.init.zeros_(layer.bias)
    return weight_norm(layer)


class PeriodCritic(nn.Module):
    def __init__(self, period, channels):
        super().__init__()
        if period < 2 or not channels:
            raise ValueError('Periods must be >= 2 and channels nonempty')
        self.period = period
        widths = [1, *channels]
        self.layers = nn.ModuleList(convolution(a, b, (5, 1),
                                   (3 if i < len(channels)-1 else 1, 1), (2, 0))
                                   for i, (a, b) in enumerate(zip(widths, widths[1:])))
        self.score = convolution(widths[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, wav):
        h = F.pad(wav[:, None], (0, (-wav.shape[-1]) % self.period), mode='replicate')
        h = h.reshape(h.shape[0], 1, -1, self.period)
        features = []
        for layer in self.layers:
            h = F.leaky_relu(layer(h), .1)
            features.append(h)
        return self.score(h), features


class SpectrumCritic(nn.Module):
    def __init__(self, n_fft, channels):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        self.layers = nn.ModuleList([
            convolution(2, channels, (3, 9), padding=(1, 4)),
            *(convolution(channels, channels, (3, 9), (1, 2), (1, 4)) for _ in range(3)),
            convolution(channels, channels, (3, 3), padding=(1, 1)),
        ])
        self.score = convolution(channels, 1, (3, 3), padding=(1, 1))

    def forward(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = wav.float() - wav.float().mean(dim=-1, keepdim=True)
            wav = .8 * wav / wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
            spectrum = torch.stft(wav, self.n_fft, self.n_fft // 4, window=self.window.float(),
                                  center=True, pad_mode='constant', return_complex=True)
            h = torch.view_as_real(spectrum).permute(0, 3, 2, 1).contiguous()
        features = []
        for layer in self.layers:
            h = F.leaky_relu(layer(h), .1)
            features.append(h)
        return self.score(h), features


class MultiDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), period_channels=(16, 64, 128, 256, 256),
                 ffts=(256, 512, 1024), spectrum_channels=32):
        super().__init__()
        if not periods and not ffts:
            raise ValueError('At least one critic is required')
        self.critics = nn.ModuleList([*(PeriodCritic(p, period_channels) for p in periods),
                                      *(SpectrumCritic(n, spectrum_channels) for n in ffts)])

    def forward(self, wav, return_features=False):
        results = [critic(wav) for critic in self.critics]
        scores, features = map(list, zip(*results))
        return (scores, features) if return_features else scores


def build_discriminator(model_cfg):
    kind = model_cfg.get('discriminator_kind', 'pwg')
    if kind == 'pwg':
        return PWGDiscriminator(**model_cfg['discriminator'])
    if kind == 'multi':
        return MultiDiscriminator(**model_cfg.get('multi_discriminator', {}))
    raise ValueError(f'Unknown discriminator kind: {kind}')


class MelCritic(nn.Module):
    """Small 2-D patch critic on log-mel spectrograms [B, n_mels, T] (generic speech).

    Used to sharpen predicted log-mel (against regression to the mean) and to judge
    frame-to-frame structure; it never sees bits or codec parameters."""

    def __init__(self, channels=(32, 64, 128), kernel=(3, 5)):
        super().__init__()
        layers, cin = [], 1
        for i, c in enumerate(channels):
            layers.append(convolution(cin, c, kernel, (1 if i == 0 else 2, 1 if i == 0 else 2),
                                      (kernel[0] // 2, kernel[1] // 2)))
            cin = c
        self.layers = nn.ModuleList(layers)
        self.score = convolution(cin, 1, (3, 3), padding=(1, 1))

    def forward(self, mel, return_features=False):
        h = mel[:, None].float()
        h = (h + 5.0) / 5.0  # natural-log power mel is roughly in [-11, 5]
        features = []
        for layer in self.layers:
            h = F.leaky_relu(layer(h), .1)
            features.append(h)
        score = self.score(h)
        return (score, features) if return_features else score
