# SPDX-License-Identifier: GPL-3.0-or-later
"""Harmonic-plus-noise excitation features for the iSTFT vocoder (generic speech prior).

Given per-frame F0 (Hz, 20 ms frames) and a voicing flag, a sum of harmonics with
continuous phase is synthesised at the sample rate (noise where unvoiced), and its STFT
(same n_fft / hop as the generator) is returned as log-magnitude plus cos/sin of the phase.
The generator receives these as extra input features, so periodicity is given by
construction instead of being inferred from a blurred spectral envelope (NSF / Wavehax idea).
Training uses F0 extracted from the clean source; inference uses the frontend's F0 head.
"""
import math

import torch
from torch import nn


class HarmonicExcitation(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 80, samples_per_frame: int = 160, sr: int = 8000,
                 max_harmonics: int = 40, noise_level: float = 0.03, mode: str = "harmonic",
                 cyclic_decay: float = 0.5, causal: bool = False, lookahead: int = 0):
        super().__init__()
        # causal: F0 is interpolated from the previous to the current frame, and analysis frame m uses
        # the excitation up to the end of its own hop only (window ending at (m + 1) * hop);
        # with lookahead 1, F0 / voicing / aperiodicity are interpolated between frame centres (the second
        # half of frame t uses frame t + 1), the analysis window stays causal
        self.causal, self.lookahead = causal, int(lookahead)
        if mode not in ("harmonic", "cyclic", "wavehax"):
            raise ValueError(f"Unknown excitation mode: {mode}")
        self.n_fft, self.hop, self.spf, self.sr = n_fft, hop, samples_per_frame, sr
        self.max_harmonics, self.noise_level, self.mode = max_harmonics, noise_level, mode
        self.cyclic_decay = cyclic_decay  # noise-burst decay as a fraction of the pitch period
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        # wavehax: real/imag of the harmonic STFT plus real/imag of a separate noise STFT
        self.out_channels = (4 if mode == "wavehax" else 3) * (n_fft // 2 + 1)

    @torch.no_grad()
    def waveform(self, f0_hz: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
        """f0_hz, voiced: [B, T] per 20 ms frame -> excitation [B, T * samples_per_frame]."""
        B, T = f0_hz.shape
        n = T * self.spf
        f0 = f0_hz.clamp_min(1.0).float()
        v = voiced.float()
        # sample-rate F0 and voicing by linear interpolation between frame centres
        if self.causal and not self.lookahead:  # sample i of frame t lies between F0[t-1] (frame start) and F0[t] (end)
            def ramp(x):
                x = torch.cat([x[:, :1], x], 1)[:, None]
                return torch.nn.functional.interpolate(x, size=n + 1, mode="linear", align_corners=True)[:, 0, 1:]
            f0_s, v_s = ramp(f0), ramp(v)
        else:
            f0_s = torch.nn.functional.interpolate(f0[:, None], size=n, mode="linear", align_corners=False)[:, 0]
            v_s = torch.nn.functional.interpolate(v[:, None], size=n, mode="linear", align_corners=False)[:, 0]
        phase = torch.cumsum(2 * math.pi * f0_s / self.sr, dim=1)
        noise = torch.randn(B, n, device=f0.device)
        if self.mode == "wavehax":
            # Harmonic prior of Wavehax (Yoneyama et al., TASLP 2025, App. A): every harmonic up to
            # Nyquist (K = floor(f_nyq / f0), 66 at 60 Hz, so low male voices keep a periodic cue in
            # 3.2-4 kHz too), constant power g_k = 0.1 sqrt(2 / K), and a per-utterance linear
            # initial phase phi_k = k theta. Voicing is applied in forward() as a smooth gain.
            k_max = int(self.sr / 2 / 60.0)
            k = torch.arange(1, k_max + 1, device=f0.device, dtype=torch.float32)
            theta = (torch.rand(B, 1, 1, device=f0.device) * 2 - 1) * math.pi
            wrapped = torch.remainder(phase, 2 * math.pi)  # keeps k * phase precise for long inputs
            harm = torch.sin(wrapped[:, :, None] * k + theta * k)
            band = (f0_s[:, :, None] * k < self.sr / 2).float()
            n_active = band.sum(dim=2).clamp_min(1.0)
            periodic = (harm * band).sum(dim=2) * 0.1 * torch.sqrt(2.0 / n_active)
            return v_s * periodic + 0.01 * noise, periodic, noise
        if self.mode == "cyclic":
            # Cyclic noise (Wang & Yamagishi, Interspeech 2020): a pulse train at the pitch period
            # convolved with one static, exponentially decaying noise burst per utterance. The
            # excitation keeps the same spectral statistics at 100 Hz and at 250 Hz, unlike a
            # fixed sum of sinusoids.
            cycle = torch.floor(phase / (2 * math.pi))
            pulses = torch.cat([torch.ones_like(cycle[:, :1]), (cycle[:, 1:] > cycle[:, :-1]).float()], dim=1)
            burst = int(self.sr / 60 * 1.5)  # covers the longest period (60 Hz) with margin
            t = torch.arange(burst, device=f0.device, dtype=torch.float32)
            mean_f0 = (f0_s * v_s).sum(dim=1) / v_s.sum(dim=1).clamp_min(1.0)  # per item
            tau = (self.sr / mean_f0.clamp_min(60.0)) * self.cyclic_decay  # samples
            kernel = torch.randn(B, burst, device=f0.device) * torch.exp(-t[None, :] / tau[:, None])
            periodic = torch.nn.functional.conv1d(
                torch.nn.functional.pad(pulses, (burst - 1, 0))[None], kernel.flip(1)[:, None, :], groups=B)[0]
            periodic = periodic / (periodic.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-5)
        else:
            k = torch.arange(1, self.max_harmonics + 1, device=f0.device, dtype=torch.float32)
            harm = torch.sin(phase[:, :, None] * k)  # [B, n, K]
            band = (f0_s[:, :, None] * k < self.sr / 2).float()  # keep harmonics below Nyquist
            n_active = band.sum(dim=2, keepdim=True).clamp_min(1.0)
            periodic = (harm * band).sum(dim=2) / n_active.sqrt().squeeze(2)
        return v_s * periodic + (1 - v_s) * self.noise_level * noise * 10 + v_s * self.noise_level * noise, periodic, noise

    def forward(self, f0_hz: torch.Tensor, voiced: torch.Tensor, frames_out: int,
                aperiodicity: torch.Tensor | None = None, return_spec: bool = False):
        """-> features [B, 3 * (n_fft/2 + 1), frames_out] at the generator's hop rate
        (with return_spec also the mixed complex excitation spectrum [B, n_fft/2 + 1, frames_out]).
        aperiodicity [B, T, bands] in [0, 1] (optional) mixes noise into the harmonic spectrum per
        frequency band (multiband excitation; Griffin & Lim 1988), instead of one voicing flag."""
        with torch.autocast(device_type=f0_hz.device.type, enabled=False):
            e, periodic, noise = self.waveform(f0_hz, voiced)
            if self.causal:
                stft = lambda x: torch.stft(torch.nn.functional.pad(x, (self.n_fft - self.hop, 0)), self.n_fft,
                                            self.hop, window=self.window.float(), center=False, return_complex=True)
            else:
                stft = lambda x: torch.stft(x, self.n_fft, self.hop, window=self.window.float(), center=True,
                                            return_complex=True)
            if self.mode == "wavehax":
                if return_spec:
                    raise ValueError("source-filter synthesis needs the harmonic or cyclic excitation mode")
                # Real/imag of the voiced harmonic prior and of an independent noise channel; the
                # generator mixes them itself (PeriodNet: sine + separate noise; Wavehax: RI > log-mag).
                # Per-band aperiodicity is deliberately not applied here (no (1 - a) harmonic scaling).
                P, N = stft(e), stft(noise * 0.1)
                scale = 4.0 / self.n_fft
                feats = torch.cat([P.real, P.imag, N.real, N.imag], dim=1) * scale
                if feats.shape[-1] < frames_out:
                    feats = torch.nn.functional.pad(feats, (0, frames_out - feats.shape[-1]), mode="replicate")
                return feats[..., :frames_out]
            if aperiodicity is None:
                spec = stft(e)
            else:
                P, N = stft(periodic), stft(noise)
                bins, frames = P.shape[1], P.shape[2]
                a = aperiodicity.float().clamp(0, 1).transpose(1, 2)  # [B, bands, T]
                if self.causal and not self.lookahead:
                    a = a.repeat_interleave(self.spf // self.hop, dim=2)
                    a = torch.nn.functional.pad(a, (0, max(frames - a.shape[2], 0)), mode="replicate")[..., :frames]
                else:
                    a = torch.nn.functional.interpolate(a, size=frames, mode="linear", align_corners=False)
                a = torch.nn.functional.interpolate(a.transpose(1, 2), size=bins, mode="nearest").transpose(1, 2)
                spec = P * (1 - a) + N * a * 0.5
            mag = torch.log(spec.abs() + 1e-5)
            feats = torch.cat([mag, torch.cos(spec.angle()), torch.sin(spec.angle())], dim=1)
        if feats.shape[-1] < frames_out:
            pad = frames_out - feats.shape[-1]
            feats = torch.nn.functional.pad(feats, (0, pad), mode="replicate")
            spec = torch.cat([spec, spec[..., -1:].expand(-1, -1, pad)], dim=-1)
        if return_spec:
            return feats[..., :frames_out], spec[..., :frames_out]
        return feats[..., :frames_out]
