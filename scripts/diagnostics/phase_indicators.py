# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-coherence indicators for listening sets (codec-agnostic).

For every voiced frame of the *target* (pyworld harvest F0) and every harmonic below 3.5 kHz,
the instantaneous frequency (IF) of each system's output is estimated from the STFT phase
advance between frames 2.5 ms apart (phase-vocoder formula). Reported per system:

- harm_if_dev_cent : mean |IF - k*F0| in cents (harmonics off their expected frequency or
                     smeared by incoherent phase -> larger)
- harm_if_jitter   : mean std of IF over 20 ms windows, in cents (frame-to-frame wobble)
- hf_flux          : mean frame-to-frame log-spectral change in 1-4 kHz (spiky/unstable spectra)
- harm_energy_ratio: energy at harmonics / total energy in voiced frames (dB)

  python -m scripts.diagnostics.phase_indicators --dir results/listen_ja_wide
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
from scipy import signal

SR = 8000
N_FFT = 1024
HOP = 20  # 2.5 ms


def stft(x):
    f, t, Z = signal.stft(x.astype(np.float64), SR, nperseg=N_FFT, noverlap=N_FFT - HOP, padded=False, boundary=None)
    return f, t, Z


def indicators(x, f0, f0_hop_s=0.005):
    f, t, Z = stft(x)
    mag = np.abs(Z)
    phase = np.angle(Z)
    bin_hz = SR / N_FFT
    # instantaneous frequency per bin from the phase advance over one hop
    dphi = np.diff(phase, axis=1)
    expected = 2 * np.pi * f[:, None] * HOP / SR
    dev = np.angle(np.exp(1j * (dphi - expected)))
    inst = f[:, None] + dev * SR / (2 * np.pi * HOP)  # [bins, frames-1]
    devs, jit, ratios = [], [], []
    win = max(1, int(0.02 * SR / HOP))
    for j in range(inst.shape[1]):
        k = min(int(round(t[j] / f0_hop_s)), len(f0) - 1)
        if f0[k] <= 0:
            continue
        harm = np.arange(1, int(3500 / f0[k]) + 1) * f0[k]
        if len(harm) < 3:
            continue
        bins = np.rint(harm / bin_hz).astype(int)
        # use the strongest bin within +-1 of the nominal harmonic bin
        cand = np.stack([bins - 1, bins, bins + 1], axis=1).clip(0, len(f) - 1)
        best = cand[np.arange(len(bins)), np.argmax(mag[cand, j], axis=1)]
        est = inst[best, j]
        valid = est > 0
        if valid.sum() < 2:
            continue
        devs.append(np.mean(np.abs(1200 * np.log2(est[valid] / harm[valid]))))
        lo, hi = max(0, j - win // 2), min(inst.shape[1], j + win // 2 + 1)
        seg = inst[best[valid], lo:hi]
        seg = np.where(seg > 0, seg, np.nan)
        jit.append(np.nanmean(np.nanstd(1200 * np.log2(seg / harm[valid, None]), axis=1)))
        p = mag[:, j] ** 2
        ratios.append(10 * np.log10(p[best].sum() / (p.sum() + 1e-12) + 1e-12))
    s = 20 * np.log10(mag + 1e-6)
    band = (f >= 1000) & (f <= 4000)
    flux = float(np.abs(np.diff(np.clip(s[band], -60, None), axis=1)).mean())
    return dict(harm_if_dev_cent=float(np.nanmean(devs)) if devs else float('nan'),
                harm_if_jitter_cent=float(np.nanmean(jit)) if jit else float('nan'),
                hf_flux_db=flux,
                harm_energy_ratio_db=float(np.nanmean(ratios)) if ratios else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    groups = defaultdict(dict)
    for p in sorted(d.glob('*.wav')):
        m = re.match(r'(.+?_tx\d+)_(.+)$', p.stem)
        if not m:
            continue
        groups[m.group(1)][m.group(2)] = p
    per_system = defaultdict(list)
    for tag, files in groups.items():
        tkey = next((k for k in files if k.endswith('target')), None)
        if tkey is None:
            continue
        target, _ = sf.read(files[tkey], dtype='float32')
        f0, _ = pyworld.harvest(target.astype(np.float64), SR, frame_period=5.0)
        for k, p in files.items():
            x, _ = sf.read(p, dtype='float32')
            n = min(len(x), len(target))
            per_system[k].append(indicators(x[:n], f0))
    keys = ['harm_if_dev_cent', 'harm_if_jitter_cent', 'hf_flux_db', 'harm_energy_ratio_db']
    lines = ['| system | n | ' + ' | '.join(keys) + ' |', '|---|---|' + '---|' * len(keys)]
    for sname, rows in per_system.items():
        lines.append(f'| {sname} | {len(rows)} | ' + ' | '.join(f"{np.nanmean([r[k] for r in rows]):.2f}" for k in keys) + ' |')
    table = '\n'.join(lines) + '\n'
    print(table)
    if a.out:
        Path(a.out).write_text(table, encoding='utf-8')
        (Path(a.out).with_suffix('.json')).write_text(json.dumps(per_system, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
