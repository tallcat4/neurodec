# SPDX-License-Identifier: GPL-3.0-or-later
"""Small ConvNeXt/iSTFT vocoder inspired by Siuzdak's Vocos (2023).

Operates on learned frame features, with no codec-specific acoustic parameters.
This is an 8 kHz adaptation, not the published Vocos model or pretrained weights.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .excitation import HarmonicExcitation


class CausalConv1d(nn.Conv1d):
    """Conv1d that sees only the current and past steps (left padding); same parameters as nn.Conv1d."""

    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, groups=groups)

    def forward(self, x):
        return super().forward(F.pad(x, (self.kernel_size[0] - 1, 0)))


def conv1d(in_channels, out_channels, kernel_size, causal=False, groups=1):
    if causal:
        return CausalConv1d(in_channels, out_channels, kernel_size, groups=groups)
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, groups=groups)


class _CausalTime2d(nn.Conv2d):
    """Conv2d over (frequency, time): centred in frequency, causal in time (the last axis)."""

    def forward(self, x):
        return super().forward(F.pad(x, (self.kernel_size[1] - 1, 0)))


class SpectralBlock(nn.Module):
    def __init__(self, width, expansion, scale, causal=False):
        super().__init__()
        self.temporal = conv1d(width, width, 7, causal, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.expand = nn.Linear(width, width * expansion)
        self.project = nn.Linear(width * expansion, width)
        self.scale = nn.Parameter(torch.full((width,), scale))

    def forward(self, x):
        h = self.norm(self.temporal(x).transpose(1, 2))
        h = self.project(F.gelu(self.expand(h))) * self.scale
        return x + h.transpose(1, 2)


class ExcitationProjector(nn.Module):
    """[B, C * F, T] excitation features -> 2D convs over (F, T) -> [B, width, T]."""

    def __init__(self, channels, bins, width, hidden=8, causal=False):
        super().__init__()
        self.channels, self.bins = channels, bins
        if causal:
            conv = lambda i, o: _CausalTime2d(i, o, (7, 3), padding=(3, 0))
        else:
            conv = lambda i, o: nn.Conv2d(i, o, (7, 3), padding=(3, 1))
        self.conv = nn.Sequential(conv(channels, hidden), nn.GELU(), conv(hidden, hidden), nn.GELU(), conv(hidden, 2))
        self.out = nn.Conv1d(2 * bins, width, 1)

    def forward(self, feats):
        B, _, T = feats.shape
        h = self.conv(feats.view(B, self.channels, self.bins, T))
        return self.out(h.reshape(B, 2 * self.bins, T))


class ISTFTGenerator(nn.Module):
    def __init__(self, input_channels, samples_per_frame=160, width=192,
                 layers=6, expansion=3, n_fft=512, hop=80, upsample='linear', excitation=False,
                 excitation_mode='harmonic', excitation_proj='linear', excitation_harmonics=40,
                 source_filter=False, gate_floor=1.0, causal=False, lookahead=0):
        """causal: every step depends only on the current and earlier frames, so the decoder can run
        frame by frame; the only look-ahead left is the overlap-add of the centred iSTFT (n_fft / 2 samples).
        lookahead (causal only; 0 or 1): frame t may also use frame t + 1, through the upsampler and the
        excitation's F0 / aperiodicity interpolation (centred between frames), to smooth the frame joins."""
        super().__init__()
        self.causal = bool(causal)
        if lookahead not in (0, 1) or (lookahead and not self.causal):
            raise ValueError('lookahead must be 0 or 1, and applies to the causal generator only')
        self.lookahead = int(lookahead)
        self.centred = centred = not self.causal or bool(self.lookahead)  # interpolation between frame centres (needs frame t + 1)
        if layers < 1 or n_fft < 4 or n_fft % 2 or not 0 < hop < n_fft // 2:
            raise ValueError('Require layers >= 1, even n_fft, and 0 < hop < n_fft/2')
        if samples_per_frame % hop:
            raise ValueError('Synthesis hop must divide samples_per_frame')
        if upsample not in ('linear', 'learned'):
            raise ValueError(f'Unknown upsample: {upsample}')
        self.samples_per_frame, self.n_fft, self.hop = samples_per_frame, n_fft, hop
        ratio = samples_per_frame // hop
        self.upsampler = None
        if upsample == 'learned' and ratio > 1:
            # Learned depthwise interpolation of the frame-rate condition, initialised to
            # linear interpolation, so the model can smooth the 20 ms steps that otherwise
            # leak into the waveform as frame-rate amplitude buzz.
            # Kernel 3*ratio spans three frames (room to learn a wider smoother than linear).
            # Causal: no padding, so output step j = t * ratio + p sees frames t, t-1, t-2 only (cropped in forward);
            # centred: frames t - 1, t, t + 1.
            self.upsampler = nn.ConvTranspose1d(input_channels, input_channels, 3 * ratio, ratio,
                                                padding=ratio if centred else 0, groups=input_channels)
        self.input = conv1d(input_channels, width, 7, self.causal)
        # Optional harmonic-plus-noise excitation features (from F0 / voicing) added at the input.
        self.excitation = HarmonicExcitation(n_fft, hop, samples_per_frame, mode=excitation_mode,
                                             max_harmonics=excitation_harmonics, causal=self.causal,
                                             lookahead=self.lookahead) if excitation else None
        # Source-filter synthesis: the network outputs a complex filter (log gain + phase shift) that is
        # applied to the excitation spectrum, so periodicity at the given F0 holds by construction
        # (MBE / NSF principle) and the F0 prior cannot be ignored. Noise enters only through the
        # excitation's unvoiced / aperiodic components.
        self.source_filter = bool(source_filter)
        if self.source_filter and not excitation:
            raise ValueError('source_filter needs excitation: true')
        # Soft source-filter (gate_floor < 1): per-bin gate g in [gate_floor, 1] mixes the filtered
        # excitation (weight g) with a freely generated unit-magnitude phasor (weight 1 - g) under the
        # same predicted magnitude. The floor keeps the excitation from being ignored (the failure of
        # additive injection), while the free path can fill energy between harmonics.
        if not 0.0 < gate_floor <= 1.0:
            raise ValueError('gate_floor must be in (0, 1]')
        self.gate_floor = float(gate_floor)
        self.gate = nn.Linear(width, n_fft // 2 + 1) if self.source_filter and gate_floor < 1.0 else None
        if excitation_proj not in ('linear', 'conv2d'):
            raise ValueError(f'Unknown excitation_proj: {excitation_proj}')
        self.exc_proj = None
        if excitation:
            if excitation_proj == 'conv2d':
                # Learnable 2D conv stack over the prior's (channel, frequency, time) image before the
                # backbone: every harmonic-prior vocoder in the literature injects through trained
                # convs (SiFi-GAN 'Direct' / 1-2 layer Harmonic-Net failed); kernel 7 along frequency
                # spans neighbouring harmonics of low F0 (3.8 bins apart at 60 Hz).
                self.exc_proj = ExcitationProjector(self.excitation.out_channels // (n_fft // 2 + 1),
                                                    n_fft // 2 + 1, width, causal=self.causal)
            else:
                self.exc_proj = nn.Conv1d(self.excitation.out_channels, width, 1)
        self.input_norm = nn.LayerNorm(width, eps=1e-6)
        self.blocks = nn.Sequential(*(SpectralBlock(width, expansion, 1 / layers, self.causal) for _ in range(layers)))
        self.output_norm = nn.LayerNorm(width, eps=1e-6)
        self.spectrum = nn.Linear(width, 2 * (n_fft // 2 + 1))
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)
        if self.upsampler is not None:
            with torch.no_grad():  # start as exact linear interpolation (trunc_normal above is overridden)
                if not centred:  # from the previous frame (taps r..2r-1) to the current one (taps 0..r-1)
                    k = torch.arange(3 * ratio, dtype=torch.float32)
                    taps = torch.where(k < ratio, (k + 1) / ratio, (1 - (k - ratio + 1) / ratio).clamp_min(0))
                else:
                    taps = (1 - (torch.arange(3 * ratio) + 0.5 - 1.5 * ratio).abs() / ratio).clamp_min(0)
                self.upsampler.weight.copy_(taps.view(1, 1, -1).expand(input_channels, 1, -1))
                self.upsampler.bias.zero_()

    def forward(self, condition, f0=None, voiced=None, aperiodicity=None):
        """condition [B, C, T]; with excitation enabled, f0 (Hz) and voiced [B, T] are required."""
        length = condition.shape[-1] * self.samples_per_frame
        # The extra boundary frame lets centered iSTFT return the exact length,
        # including short and grouped-frame inputs, without zero padding audio.
        if self.upsampler is not None:
            h = self.upsampler(condition)[..., : length // self.hop]
        elif not self.centred:  # hold each frame (linear interpolation would need the next one)
            h = condition.repeat_interleave(self.samples_per_frame // self.hop, dim=-1)
        else:
            h = F.interpolate(condition, size=length // self.hop, mode='linear', align_corners=False)
        h = F.pad(h, (0, 1), mode='replicate')
        h = self.input(h)
        exc_spec = None
        if self.excitation is not None:
            if f0 is None or voiced is None:
                raise ValueError('This generator uses harmonic excitation and needs f0 and voiced per frame')
            if self.source_filter:
                feats, exc_spec = self.excitation(f0, voiced, h.shape[-1], aperiodicity, return_spec=True)
            else:
                feats = self.excitation(f0, voiced, h.shape[-1], aperiodicity)
            h = h + self.exc_proj(feats.to(h.dtype))
        h = self.input_norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.output_norm(self.blocks(h).transpose(1, 2))
        coefficients = self.spectrum(h).transpose(1, 2).float()
        gate = self.gate(h).transpose(1, 2).float() if self.gate is not None else None
        with torch.autocast(device_type=condition.device.type, enabled=False):
            log_magnitude, phase = coefficients.chunk(2, dim=1)
            magnitude = log_magnitude.clamp(-12, 4.6).exp()
            if exc_spec is not None:
                # unit-RMS excitation spectrum per frame, shaped by the predicted gain and phase shift
                e = exc_spec.to(torch.complex64)  # keep the imaginary part (float() would drop it)
                e = e / (e.abs().pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-5)
                if gate is not None:
                    g = self.gate_floor + (1 - self.gate_floor) * torch.sigmoid(gate)
                    shifted = torch.complex(phase.cos(), phase.sin())
                    mixed = g * e * shifted + (1 - g) * shifted
                    spectrum = torch.complex(magnitude * mixed.real, magnitude * mixed.imag)
                    return torch.istft(spectrum, self.n_fft, self.hop, window=self.window.float(),
                                       center=True, length=length)
                magnitude = magnitude * e.abs()
                phase = phase + e.angle()
            spectrum = torch.complex(magnitude * phase.cos(), magnitude * phase.sin())
            return torch.istft(spectrum, self.n_fft, self.hop, window=self.window.float(),
                               center=True, length=length)
