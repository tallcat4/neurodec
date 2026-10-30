# SPDX-License-Identifier: GPL-3.0-or-later
"""Speaker -> air -> radio microphone path.

The encoder sees the simulated signal; the training target stays the clean source.
All randomness comes from the passed Generator so a session is reproducible.
"""
import numpy as np
from scipy import signal

SR = 8000


def _peaking(f0: float, gain_db: float, q: float):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return np.array(b) / den[0], np.array(den) / den[0]


def random_eq(x: np.ndarray, rng: np.random.Generator, hp=(100, 400), lp=(2800, 3800),
              n_peaks=3, peak_db=6.0) -> np.ndarray:
    """Band limiting plus a few random resonances (small speaker / electret mic)."""
    sos = signal.butter(2, rng.uniform(*hp), "highpass", fs=SR, output="sos")
    x = signal.sosfilt(sos, x)
    sos = signal.butter(4, rng.uniform(*lp), "lowpass", fs=SR, output="sos")
    x = signal.sosfilt(sos, x)
    for _ in range(rng.integers(0, n_peaks + 1)):
        b, a = _peaking(rng.uniform(200, 3500), rng.uniform(-peak_db, peak_db), rng.uniform(0.7, 4.0))
        x = signal.lfilter(b, a, x)
    return x


def synthetic_rir(rng: np.random.Generator, rt60=(0.05, 0.5), drr_db=(0, 15)) -> np.ndarray:
    """Direct path plus exponentially decaying noise tail."""
    t60 = rng.uniform(*rt60)
    n = int(t60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / t60)
    pre = int(rng.uniform(0.0005, 0.003) * SR)  # 0.5-3 ms until the first reflections
    tail[:pre] = 0.0
    tail /= np.sqrt(np.sum(tail ** 2)) + 1e-12
    h = tail * 10 ** (-rng.uniform(*drr_db) / 20)
    h[0] += 1.0
    return h


def colored_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """White, pink or brown noise, unit RMS."""
    beta = rng.choice([0.0, 1.0, 2.0])
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec /= f ** (beta / 2)
    x = np.fft.irfft(spec, n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def clock_drift(x: np.ndarray, ppm: float) -> np.ndarray:
    """Playback sound card and radio ADC clocks differ by a few tens of ppm."""
    n_out = int(round(len(x) * (1 + ppm * 1e-6)))
    t = np.linspace(0, len(x) - 1, n_out)
    return np.interp(t, np.arange(len(x)), x)


def agc(x: np.ndarray, target_db: float, attack=0.01, release=0.5, max_gain_db=20.0) -> np.ndarray:
    """Slow feed-forward AGC on a 1 ms envelope (radio mic preamps usually have one)."""
    hop = SR // 1000
    n = len(x) // hop * hop
    env = np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1) + 1e-10)
    a_att, a_rel = np.exp(-1 / (attack * 1000)), np.exp(-1 / (release * 1000))
    level = np.empty_like(env)
    cur = env[0]
    for i, e in enumerate(env):
        a = a_att if e > cur else a_rel
        cur = a * cur + (1 - a) * e
        level[i] = cur
    gain_db = np.clip(target_db - 20 * np.log10(level + 1e-10), -max_gain_db, max_gain_db)
    gain = np.repeat(10 ** (gain_db / 20), hop)
    gain = np.concatenate([gain, np.full(len(x) - n, gain[-1] if len(gain) else 1.0)])
    return x * gain


class AcousticSim:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        if not c.get("enabled", True):
            return x.astype(np.float32)
        y = x.astype(np.float64)
        y = clock_drift(y, rng.uniform(-c["drift_ppm"], c["drift_ppm"]))[: len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        y = random_eq(y, rng)  # loudspeaker
        y = signal.fftconvolve(y, synthetic_rir(rng, tuple(c["rt60"]), tuple(c["drr_db"])))[: len(x)]
        y = random_eq(y, rng, hp=(80, 300), lp=(3200, 3900), n_peaks=2, peak_db=4.0)  # microphone
        level_db = rng.uniform(*c["level_db"])
        rms = np.sqrt(np.mean(y ** 2)) + 1e-12
        y *= 10 ** (level_db / 20) / rms
        snr = rng.uniform(*c["snr_db"])
        y += colored_noise(len(y), rng) * np.sqrt(np.mean(y ** 2)) * 10 ** (-snr / 20)
        if c.get("agc", True) and rng.random() < c.get("agc_prob", 0.5):
            y = agc(y, level_db)
        return np.clip(y, -1.0, 1.0).astype(np.float32)
