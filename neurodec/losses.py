# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-resolution STFT loss (PWG, sizes halved for 8 kHz as in the paper) and GAN losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    def __init__(self, n_fft, hop, win):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    def mag(self, x):
        s = torch.stft(x, self.n_fft, self.hop, self.win, self.window, return_complex=True)
        return torch.sqrt(torch.clamp(s.real ** 2 + s.imag ** 2, min=1e-7))

    def forward(self, y_hat, y):
        a, b = self.mag(y_hat), self.mag(y)
        sc = torch.linalg.norm(b - a, dim=(1, 2)) / (torch.linalg.norm(b, dim=(1, 2)) + 1e-7)
        logmag = F.l1_loss(torch.log(a), torch.log(b))
        return sc.mean(), logmag


class MultiResSTFTLoss(nn.Module):
    def __init__(self, ffts=(512, 1024, 256), hops=(60, 120, 25), wins=(300, 600, 120)):
        super().__init__()
        self.losses = nn.ModuleList(STFTLoss(*p) for p in zip(ffts, hops, wins))

    def forward(self, y_hat, y):
        sc = mag = 0.0
        for l in self.losses:
            a, b = l(y_hat, y)
            sc, mag = sc + a, mag + b
        return sc / len(self.losses), mag / len(self.losses)


def d_loss(real_logits, fake_logits, kind="lsgan"):
    """Eq. 1. 'bce' is the paper's log form, 'lsgan' the PWG reference implementation."""
    if isinstance(real_logits, (list, tuple)):
        if not real_logits or len(real_logits) != len(fake_logits):
            raise ValueError('Mismatched or empty critic outputs')
        return sum(d_loss(r, f, kind) for r, f in zip(real_logits, fake_logits)) / len(real_logits)
    if kind == "bce":
        return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    return F.mse_loss(real_logits, torch.ones_like(real_logits)) + F.mse_loss(fake_logits, torch.zeros_like(fake_logits))


def g_adv_loss(fake_logits, kind="lsgan"):
    if isinstance(fake_logits, (list, tuple)):
        if not fake_logits:
            raise ValueError('Empty critic outputs')
        return sum(g_adv_loss(f, kind) for f in fake_logits) / len(fake_logits)
    if kind == "bce":  # non-saturating form of log(1 - D(G))
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


def feature_matching(real_features, fake_features):
    if not real_features or len(real_features) != len(fake_features):
        raise ValueError('Mismatched or empty critic feature lists')
    losses = []
    for real, fake in zip(real_features, fake_features):
        if not real or len(real) != len(fake):
            raise ValueError('Mismatched or empty layer feature lists')
        losses.append(sum(F.l1_loss(f, r.detach()) for r, f in zip(real, fake)) / len(real))
    return sum(losses) / len(losses)


class PhaseLoss(nn.Module):
    """Phase-aware spectral losses (APNet2-style, generic speech): anti-wrapped differences of
    the instantaneous phase advance over time (instantaneous frequency) and over frequency
    (group delay) between the generated and the target STFT, weighted by the target's
    magnitude so that the noise floor does not dominate."""

    def __init__(self, n_fft=512, hop=80, floor_db=-50.0):
        super().__init__()
        self.n_fft, self.hop, self.floor_db = n_fft, hop, floor_db
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    @staticmethod
    def _wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, y_hat, y):
        with torch.autocast(device_type=y.device.type, enabled=False):
            a = torch.stft(y_hat.float(), self.n_fft, self.hop, window=self.window, return_complex=True)
            b = torch.stft(y.float(), self.n_fft, self.hop, window=self.window, return_complex=True)
            mag = b.abs()
            weight = (20 * torch.log10(mag + 1e-6) > mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6).log10() * 20 + self.floor_db).float()
            pa, pb = torch.angle(a), torch.angle(b)
            # instantaneous frequency: phase advance between consecutive frames
            iaf = self._wrap(self._wrap(pa[..., 1:] - pa[..., :-1]) - self._wrap(pb[..., 1:] - pb[..., :-1]))
            w_t = weight[..., 1:] * weight[..., :-1]
            # group delay: phase difference between neighbouring bins
            gd = self._wrap(self._wrap(pa[:, 1:] - pa[:, :-1]) - self._wrap(pb[:, 1:] - pb[:, :-1]))
            w_f = weight[:, 1:] * weight[:, :-1]
            l_if = (iaf.abs() * w_t).sum() / w_t.sum().clamp_min(1.0)
            l_gd = (gd.abs() * w_f).sum() / w_f.sum().clamp_min(1.0)
            return l_if, l_gd


def periodicity_loss(pred, f0_hz, f0_floor: float = 60.0):
    """pred [B, 2(+bands), T] = (log F0, voicing logit, band-aperiodicity logits);
    f0_hz [B, T] teacher (0 = unvoiced) or [B, T, 1 + bands] with the aperiodicity teacher.
    Returns (L1 on log F0 over voiced frames, BCE on voicing[, BCE on aperiodicity over voiced frames])."""
    if f0_hz.dim() == 3:
        ap_teacher, f0_hz = f0_hz[..., 1:], f0_hz[..., 0]
    else:
        ap_teacher = None
    voiced = (f0_hz > 0).float()
    log_f0 = torch.log(f0_hz.clamp_min(f0_floor))
    l_f0 = ((pred[:, 0] - log_f0).abs() * voiced).sum() / voiced.sum().clamp_min(1.0)
    l_vuv = F.binary_cross_entropy_with_logits(pred[:, 1], voiced)
    if ap_teacher is not None and pred.shape[1] > 2:
        n = min(pred.shape[1] - 2, ap_teacher.shape[-1])
        l_ap = F.binary_cross_entropy_with_logits(pred[:, 2:2 + n].transpose(1, 2), ap_teacher[..., :n],
                                                  reduction="none").mean(dim=-1)
        l_ap = (l_ap * voiced).sum() / voiced.sum().clamp_min(1.0)
        return l_f0, l_vuv + l_ap
    return l_f0, l_vuv
