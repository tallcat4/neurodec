# SPDX-License-Identifier: GPL-3.0-or-later
"""Non-speech probe signals sent alongside the corpus.

They cover parameter regions that read speech rarely reaches (very high / low pitch,
long sustained vowels, pure noise, tones, silence) at a small share of the budget.
"""
import numpy as np
from scipy import signal

SR = 8000
# Rough (F1, F2, F3) of /a i u e o/.
VOWELS = [(800, 1200, 2500), (300, 2300, 3000), (350, 1300, 2400), (500, 1900, 2600), (500, 900, 2400)]


def _formants(x, formants, rng):
    for f in formants:
        f = min(f * rng.uniform(0.85, 1.15), 3700)
        b, a = signal.iirpeak(f, f / rng.uniform(5, 15), fs=SR)
        x = signal.lfilter(b, a, x) + 0.1 * x
    return x


def glottal_sweep(n, rng):
    """Pulse train with a gliding F0 (60-400 Hz) through changing vowels."""
    f0 = np.exp(np.interp(np.arange(n), np.linspace(0, n, 5), np.log(rng.uniform(60, 400, 5))))
    phase = np.cumsum(f0 / SR)
    src = np.diff(np.floor(phase), prepend=0.0) * 1.0
    src = signal.lfilter([1.0], [1.0, -0.95], src)  # glottal tilt
    src += rng.uniform(0, 0.05) * rng.standard_normal(n)  # breathiness
    seg = max(n // rng.integers(1, 5), 1)
    out = np.zeros(n)
    for s in range(0, n, seg):
        out[s : s + seg] = _formants(src[s : s + seg], VOWELS[rng.integers(len(VOWELS))], rng)
    return out


def noise_burst(n, rng):
    """Fricative-like band-passed noise bursts."""
    x = rng.standard_normal(n)
    lo = rng.uniform(1000, 2500)
    sos = signal.butter(4, [lo, min(lo + rng.uniform(500, 2500), 3900)], "bandpass", fs=SR, output="sos")
    x = signal.sosfilt(sos, x)
    env = (rng.random(n // 800 + 1) > 0.4).astype(float).repeat(800)[:n]
    return x * signal.lfilter([0.02], [1, -0.98], env)


def tones(n, rng):
    """One or two sinusoids with slow FM."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for _ in range(rng.integers(1, 3)):
        f = rng.uniform(100, 3500)
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 3) * np.sin(2 * np.pi * rng.uniform(0.1, 3) * t))
    return x


def hum(n, rng):
    t = np.arange(n) / SR
    f = rng.choice([50.0, 60.0])
    return sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 8))


def silence(n, rng):
    return rng.standard_normal(n) * 1e-4


GENERATORS = {"glottal": (glottal_sweep, 0.5), "noise": (noise_burst, 0.2), "tones": (tones, 0.15),
              "hum": (hum, 0.05), "silence": (silence, 0.1)}


def make_probe(rng: np.random.Generator, seconds=(2.0, 8.0)) -> tuple[str, np.ndarray]:
    names = list(GENERATORS)
    p = np.array([GENERATORS[k][1] for k in names])
    kind = names[rng.choice(len(names), p=p / p.sum())]
    n = int(rng.uniform(*seconds) * SR)
    x = GENERATORS[kind][0](n, rng)
    x = x / (np.max(np.abs(x)) + 1e-9) * 10 ** (rng.uniform(-30, -6) / 20)
    return kind, x.astype(np.float32)
