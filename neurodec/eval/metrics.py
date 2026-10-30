# SPDX-License-Identifier: GPL-3.0-or-later
"""Objective metrics at 8 kHz: PESQ-NB, STOI, log-mel distance, F0 RMSE, V/UV error,
plus two listening-oriented indicators (harmonic-to-noise ratio, frame-rate buzz)."""
import numpy as np
import pyworld
from pesq import pesq
from pystoi import stoi
from scipy import signal

SR = 8000


def _envelope(x: np.ndarray, hop: int = 8) -> np.ndarray:
    """Log RMS envelope at 1 ms resolution (20-sample window); zero mean."""
    n = len(x) // hop * hop
    e = np.convolve(x.astype(np.float64) ** 2, np.ones(20) / 20, mode="same")[:n].reshape(-1, hop).mean(1)
    e = np.log(e + 1e-7)
    return e - e.mean()


def pooled_lag(refs, degs, max_lag: int = 2000, hop: int = 8) -> int:
    """Constant delay of a decoder: deg[n + k] ~ ref[n]. Parametric vocoders do not keep
    the waveform phase, so the log-energy envelopes are correlated (summed over all
    pairs), at 1 ms resolution."""
    m = max_lag // hop
    lags = np.arange(-m, m + 1)
    acc = np.zeros(len(lags))
    for ref, deg in zip(refs, degs):
        r, d = _envelope(ref, hop), _envelope(deg, hop)
        n = len(r) + len(d)
        nfft = 1 << (n - 1).bit_length()
        cc = np.fft.irfft(np.fft.rfft(d, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)
        acc += cc[lags % nfft] / (np.linalg.norm(r) * np.linalg.norm(d) + 1e-9)
    return int(lags[np.argmax(acc)] * hop)


def shift(x: np.ndarray, k: int, n: int) -> np.ndarray:
    """y[i] = x[i + k], length n, zero padded."""
    y = np.zeros(n, np.float32)
    lo, hi = max(0, -k), min(n, len(x) - k)
    if hi > lo:
        y[lo:hi] = x[lo + k : hi + k]
    return y


def _logmel(x, n_mels=40):
    f, t, Z = signal.stft(x, SR, nperseg=256, noverlap=256 - 80)
    P = np.abs(Z) ** 2
    hz2mel = lambda h: 2595 * np.log10(1 + h / 700)
    edges = 700 * (10 ** (np.linspace(hz2mel(50), hz2mel(3800), n_mels + 2) / 2595) - 1)
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        l, c, r = edges[i : i + 3]
        fb[i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    return 10 * np.log10(fb @ P + 1e-8)


def logmel_distance(ref, deg) -> float:
    """RMS dB difference over frames where the reference is within 40 dB of its peak."""
    a, b = _logmel(ref), _logmel(deg)
    e = a.max(axis=0)
    keep = e > e.max() - 40
    return float(np.sqrt(np.mean((a[:, keep] - b[:, keep]) ** 2)))


def f0_metrics(ref, deg) -> tuple[float, float]:
    """(F0 RMSE in cents over frames voiced in both, V/UV disagreement rate)."""
    def f0(x):
        x = x.astype(np.float64)
        f, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=10)
        return pyworld.stonemask(x, f, t, SR)
    a, b = f0(ref), f0(deg)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    va, vb = a > 0, b > 0
    both = va & vb
    rmse = float(np.sqrt(np.mean((1200 * np.log2(b[both] / a[both])) ** 2))) if both.any() else float("nan")
    return rmse, float(np.mean(va != vb))


def hnr_db(x: np.ndarray, f0_ref: np.ndarray | None = None, hop_s: float = 0.01) -> float:
    """Harmonic-to-noise ratio (dB) over voiced frames: energy at harmonics of the reference
    F0 versus energy midway between them (up to 3.5 kHz). Lower = rougher / buzzier."""
    x = x.astype(np.float64)
    if f0_ref is None:
        f, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=hop_s * 1000)
        f0_ref = pyworld.stonemask(x, f, t, SR)
    freqs, times, Z = signal.stft(x, SR, nperseg=1024, noverlap=1024 - int(hop_s * SR), padded=False)
    P = np.abs(Z) ** 2
    vals = []
    for i, ti in enumerate(times):
        k = min(int(round(ti / hop_s)), len(f0_ref) - 1)
        if f0_ref[k] <= 0:
            continue
        harm = np.arange(1, int(3500 / f0_ref[k]) + 1) * f0_ref[k]
        if len(harm) < 3:
            continue
        hi = np.rint(harm / (SR / 1024)).astype(int)
        mid = np.rint((harm[:-1] + f0_ref[k] / 2) / (SR / 1024)).astype(int)
        vals.append(10 * np.log10(P[hi, i].sum() / (P[mid, i].sum() + 1e-12) + 1e-12))
    return float(np.mean(vals)) if vals else float("nan")


def frame_buzz_db(x: np.ndarray, rate_hz: float = 50.0, hop: int = 8) -> float:
    """Prominence (dB above the 30-300 Hz median) of the log-energy-envelope modulation
    spectrum at `rate_hz` (50 Hz = one 20 ms codec frame). Large values mean amplitude
    buzz locked to the frame rate."""
    env = _envelope(x, hop)
    if len(env) < 256:
        return float("nan")
    fs = SR / hop
    f, pxx = signal.welch(env, fs, nperseg=min(1024, len(env)))
    p_db = 10 * np.log10(pxx + 1e-12)
    floor = np.median(p_db[(f > 30) & (f < 300)])
    return float(p_db[np.argmin(np.abs(f - rate_hz))] - floor)


def all_metrics(ref: np.ndarray, deg: np.ndarray) -> dict:
    n = min(len(ref), len(deg))
    ref, deg = ref[:n].astype(np.float32), deg[:n].astype(np.float32)
    out = {"logmel_db": logmel_distance(ref, deg)}
    try:  # listening-oriented indicators; never fail the evaluation
        out["hnr_db"] = hnr_db(deg)
        out["hnr_ref_db"] = hnr_db(ref)
        out["buzz50_db"] = frame_buzz_db(deg)
        out["buzz50_ref_db"] = frame_buzz_db(ref)
    except Exception:
        out.update(hnr_db=float("nan"), hnr_ref_db=float("nan"), buzz50_db=float("nan"), buzz50_ref_db=float("nan"))
    try:
        out["pesq_nb"] = float(pesq(SR, ref, deg, "nb"))
    except Exception:  # e.g. no utterance detected
        out["pesq_nb"] = float("nan")
    out["stoi"] = float(stoi(ref, deg, SR, extended=False))
    out["f0_rmse_cent"], out["vuv_err"] = f0_metrics(ref, deg)
    return out
