# SPDX-License-Identifier: GPL-3.0-or-later
"""Artifact analysis of listening WAVs: spectrograms plus simple buzz/spike indicators.

Indicators (all computed on the same utterance, no codec knowledge):
- band energy (dB) relative to the target in 0-1, 1-2, 2-3, 3-4 kHz
- frame-rate modulation: energy-envelope modulation spectrum peaks at 50/100/200 Hz
  (buzz locked to 20 ms / 10 ms / 5 ms frames)
- harmonic-to-noise ratio of voiced frames (pyworld f0 on the target)
- spectral flux: mean frame-to-frame log-spectral change (spiky/unstable spectra)

  python -m scripts.diagnostics.analyze_artifacts --dir results/listen_progress_2323 --out results/listen_progress_2323/analysis
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 8000


def logspec(x, n_fft=512, hop=40):
    f, t, z = signal.stft(x, SR, nperseg=n_fft, noverlap=n_fft - hop, padded=False)
    return f, t, 20 * np.log10(np.abs(z) + 1e-6)


def band_db(x):
    f, _, s = logspec(x)
    p = (10 ** (s / 10)).mean(axis=1)
    out = {}
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]:
        m = (f >= lo) & (f < hi)
        out[f'{lo // 1000}-{hi // 1000}k'] = float(10 * np.log10(p[m].sum() + 1e-12))
    return out


def modulation_peaks(x, hop=8):
    env = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode='valid')[::hop] + 1e-12)
    env = np.log(env) - np.log(env).mean()
    fs = SR / hop
    f, p = signal.welch(env, fs, nperseg=min(1024, len(env)))
    p_db = 10 * np.log10(p + 1e-12)
    floor = np.median(p_db[(f > 30) & (f < 300)])
    return {f'{m}Hz': float(p_db[np.argmin(np.abs(f - m))] - floor) for m in (50, 100, 200)}


def hnr_voiced(x, f0, hop_s=0.005):
    """Harmonic-to-noise ratio (dB) using a comb of harmonics vs between-harmonic energy."""
    f, t, s = logspec(x, 1024, int(hop_s * SR))
    p = 10 ** (s / 10)
    vals = []
    for i, ti in enumerate(t):
        k = min(int(ti / hop_s), len(f0) - 1)
        if f0[k] <= 0:
            continue
        harm = np.arange(1, int(3500 / f0[k]) + 1) * f0[k]
        if len(harm) < 3:
            continue
        hi = np.array([np.argmin(np.abs(f - h)) for h in harm])
        mid = np.array([np.argmin(np.abs(f - h)) for h in harm[:-1] + f0[k] / 2])
        vals.append(10 * np.log10(p[hi, i].sum() / (p[mid, i].sum() + 1e-12)))
    return float(np.mean(vals)) if vals else float('nan')


def spectral_flux(x):
    _, _, s = logspec(x)
    s = np.clip(s, -40, None)
    return float(np.abs(np.diff(s, axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(d.glob('*.wav'))
    target, _ = sf.read(d / 'target.wav', dtype='float32')
    import pyworld
    f0, _ = pyworld.harvest(target.astype(np.float64), SR, frame_period=5.0)
    rows = {}
    ref_bands = band_db(target)
    for p in files:
        x, _ = sf.read(p, dtype='float32')
        n = min(len(x), len(target))
        bands = band_db(x[:n])
        rows[p.stem] = dict(band_db_vs_target={k: round(bands[k] - ref_bands[k], 2) for k in bands},
                            frame_modulation_db=modulation_peaks(x[:n]),
                            hnr_db=round(hnr_voiced(x[:n], f0), 2), spectral_flux=round(spectral_flux(x[:n]), 3),
                            rms_db=round(float(20 * np.log10(np.sqrt(np.mean(x[:n] ** 2)) + 1e-9)), 2))
    (out / 'indicators.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for k, v in rows.items():
        print(k, json.dumps(v))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(files), 1, figsize=(12, 2.2 * len(files)), sharex=True)
        for ax, p in zip(np.atleast_1d(axes), files):
            x, _ = sf.read(p, dtype='float32')
            f, t, s = logspec(x[:len(target)])
            ax.pcolormesh(t, f, np.clip(s, -70, 0), shading='auto', cmap='magma')
            ax.set_ylabel(p.stem[:28], fontsize=7)
        fig.tight_layout()
        fig.savefig(out / 'spectrograms.png', dpi=110)
        print('wrote', out / 'spectrograms.png')
    except ImportError:
        print('matplotlib not available; indicators only')


if __name__ == '__main__':
    main()
