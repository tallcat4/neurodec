# SPDX-License-Identifier: GPL-3.0-or-later
"""Periodicity teachers computed from clean speech only (training aids, never codec parameters)."""
import numpy as np

SR = 8000
FRAME = 160
N_BANDS = 5


def frame_f0(wav: np.ndarray, frames: int) -> np.ndarray:
    """pyworld F0 (Hz, 0 = unvoiced) per 20 ms frame."""
    import pyworld

    x = wav.astype(np.float64)
    f0, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=FRAME * 1000 / SR)
    f0 = pyworld.stonemask(x, f0, t, SR)[:frames]
    if len(f0) < frames:
        f0 = np.pad(f0, (0, frames - len(f0)))
    return f0.astype(np.float32)


def band_aperiodicity(wav: np.ndarray, f0: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    """[frames, N_BANDS] in [0, 1]: fraction of band energy between the harmonics of F0
    (1 = noise-like; unvoiced frames are 1). Generic harmonic-vs-noise measure."""
    from scipy import signal

    frames = len(f0)
    f, t, Z = signal.stft(wav.astype(np.float64), SR, nperseg=n_fft, noverlap=n_fft - FRAME, padded=False, boundary=None)
    P = np.abs(Z) ** 2
    edges = np.linspace(0, SR / 2, N_BANDS + 1)
    ap = np.ones((frames, N_BANDS), np.float32)
    bin_hz = SR / n_fft
    for j in range(min(frames, P.shape[1])):
        if f0[j] <= 0:
            continue
        harm = np.arange(1, int((SR / 2) / f0[j])) * f0[j]
        mid = harm[:-1] + f0[j] / 2
        hb = np.rint(harm / bin_hz).astype(int)
        mb = np.rint(mid / bin_hz).astype(int)
        for b in range(N_BANDS):
            lo, hi = edges[b], edges[b + 1]
            hs = hb[(harm >= lo) & (harm < hi)]
            ms = mb[(mid >= lo) & (mid < hi)]
            if len(hs) == 0 or len(ms) == 0:
                continue
            h, m = P[hs, j].mean(), P[ms, j].mean()
            ap[j, b] = float(m / (h + m + 1e-12))
    return ap
