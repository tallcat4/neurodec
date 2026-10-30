# SPDX-License-Identifier: GPL-3.0-or-later
"""Precompute frame-rate F0 / voicing and band-wise aperiodicity teachers for every training transmission.

The teacher is extracted from the clean source (pyworld dio + stonemask at the 20 ms model
frame rate, aligned like the training targets), so it needs no codec knowledge. Written to
<data_dir>/train/f0/<file>.npy as float32 Hz per frame (0 = unvoiced). Evaluation never
uses these files; they only supervise auxiliary heads during training.

  python -m scripts.precompute_f0 --config configs/bits49.yaml --workers 16
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neurodec.config import load_config
from neurodec.data.align import aligned_target

SR = 8000
FRAME = 160


from neurodec.data.periodicity import N_BANDS, band_aperiodicity as _band_aperiodicity


def band_aperiodicity(target: np.ndarray, f0: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    """[frames, N_BANDS] in [0, 1]: fraction of band energy *between* harmonics (1 = noise-like).

    For each voiced frame, energy at the harmonic bins of F0 (up to Nyquist) is compared with
    energy at the mid-points between harmonics, per frequency band; unvoiced frames are 1.
    This is a generic speech measure computed from the clean source, not a codec parameter."""
    from scipy import signal
    frames = len(f0)
    f, t, Z = signal.stft(target, SR, nperseg=n_fft, noverlap=n_fft - FRAME, padded=False, boundary=None)
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


def _one(args):
    npz, offset, out = args
    import pyworld
    z = np.load(npz)
    src = z['src'].astype(np.float32) / 32767
    frames = int(z['bits'].shape[0])
    target = aligned_target(src, offset, 0, frames).astype(np.float64)
    f0, t = pyworld.dio(target, SR, f0_floor=60, f0_ceil=500, frame_period=FRAME * 1000 / SR)
    f0 = pyworld.stonemask(target, f0, t, SR)
    f0 = f0[:frames]
    if len(f0) < frames:
        f0 = np.pad(f0, (0, frames - len(f0)))
    f0 = f0.astype(np.float32)
    np.save(out, f0)
    ap = band_aperiodicity(target, f0)
    np.save(out.with_name(out.stem + '_ap.npy'), ap)
    return out.name, frames, float((f0 > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = Path(cfg['data_dir']) / a.split
    offset = json.loads((Path(cfg['data_dir']) / 'alignment.json').read_text())['offset']
    out_dir = d / 'f0'
    out_dir.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (d / 'index.jsonl').read_text().splitlines()]
    jobs = [(d / r['file'], offset, out_dir / (Path(r['file']).stem + '.npy')) for r in rows]
    with ProcessPoolExecutor(a.workers) as ex:
        done = list(ex.map(_one, jobs, chunksize=4))
    voiced = np.mean([x[2] for x in done])
    print(f'{len(done)} transmissions, {sum(x[1] for x in done)} frames, voiced fraction {voiced:.3f}, -> {out_dir}')


if __name__ == '__main__':
    main()
