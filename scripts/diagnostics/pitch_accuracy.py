# SPDX-License-Identifier: GPL-3.0-or-later
"""Raw pitch accuracy (RPA) and raw chroma accuracy (RCA) per gender (Jassim et al. 2020 protocol).

For JVS test utterances of one gender, decode with each checkpoint, extract F0 of the clean
source and of each output (pyworld dio+stonemask, 10 ms), and count frames (voiced in the
reference) where the output F0 is within 50 cents of the reference (RPA) or within 50 cents
after folding octaves (RCA). RPA << RCA means octave (doubling/halving) errors. Also reports
the fraction of reference-voiced frames rendered unvoiced (dropouts).

  python -m scripts.diagnostics.pitch_accuracy --config configs/experiments/b49_bittoken_wide_bridge_f0_cuda.yaml --gender M \
      --ckpt runs/joint_b49_wide_bridge_f0/step30000.pt --utts 50 --out results/pitch_accuracy_M.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyworld
import torch

from neurodec.blackbox import Receiver
from neurodec.config import load_config
from neurodec.data.corpus import Utterance
from neurodec.data.dataset import Transmissions
from neurodec.data.session import plan_sessions
from neurodec.device import pick_device
from neurodec.eval.metrics import pooled_lag, shift
from neurodec.infer import decode
from neurodec.train import load_model

SR = 8000


def f0_track(x):
    x = x.astype(np.float64)
    f0, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=10)
    return pyworld.stonemask(x, f0, t, SR)


def accuracy(ref, deg):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    v = ref > 0
    if v.sum() == 0:
        return None
    hit_v = (deg > 0) & v
    cents = 1200 * np.log2(np.where(hit_v, deg, 1.0) / np.where(hit_v, ref, 1.0))
    rpa = float((hit_v & (np.abs(cents) <= 50)).sum() / v.sum())
    chroma = np.abs(((cents + 600) % 1200) - 600)
    rca = float((hit_v & (chroma <= 50)).sum() / v.sum())
    dropout = float(((deg <= 0) & v).sum() / v.sum())
    return dict(rpa=rpa, rca=rca, dropout=dropout, voiced_frames=int(v.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--gender', default='M', choices=['M', 'F'])
    ap.add_argument('--cond', default='clean')
    ap.add_argument('--utts', type=int, default=50)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device()
    tx = Transmissions(Path(cfg['data_dir']) / f'test_{a.cond}')
    test = [Utterance(**u) for u in json.loads((Path(cfg['data_dir']) / 'split.json').read_text(encoding='utf-8'))['test']]
    span_speakers = [[u.speaker for u in sess] for sess in plan_sessions(test, cfg['session']['seconds'])]
    root = next(c['root'] for c in cfg['corpora'] if c['type'] == 'jvs')
    table = {}
    for line in Path(root).joinpath('gender_f0range.txt').read_text(encoding='utf-8').splitlines()[1:]:
        parts = line.split(); table['jvs/' + parts[0]] = parts[1]
    rx = Receiver.from_config(cfg)
    g = cfg['blackbox']['block_frames']
    models = {Path(c).parent.name + '@' + Path(c).stem: load_model(c, device)[0] for c in a.ckpt}
    stats = {k: [] for k in ['reference', *models]}
    done = 0
    for i in range(len(tx)):
        if done >= a.utts:
            break
        wanted = [u for u, s in enumerate(span_speakers[i]) if table.get(s) == a.gender]
        if not wanted:
            continue
        bits, target = tx.bits[i], tx.target(i)
        spans = np.load(tx.dir / tx.rows[i]['file'])['spans']
        outs = {'reference': rx.receive(bits)}
        for name, G in models.items():
            torch.manual_seed(cfg['seed'] + i)
            outs[name] = decode(G, bits, device, chunk_frames=500, group=g)
        lags = {k: pooled_lag([target], [y]) for k, y in outs.items()}
        for u in wanted:
            if done >= a.utts:
                break
            s, e = spans[u] + tx.offset
            if s < 0 or e > len(target):
                continue
            ref_f0 = f0_track(target[s:e])
            for k, y in outs.items():
                y = shift(y, lags[k], len(target))
                r = accuracy(ref_f0, f0_track(y[s:e]))
                if r:
                    stats[k].append(r)
            done += 1
    lines = [f'| system ({a.gender}, n={done}) | RPA | RCA | octave err (RCA-RPA) | dropout |', '|---|---|---|---|---|']
    for k, rs in stats.items():
        m = lambda f: np.mean([r[f] for r in rs])
        lines.append(f"| {k} | {m('rpa'):.3f} | {m('rca'):.3f} | {m('rca') - m('rpa'):.3f} | {m('dropout'):.3f} |")
    text = '\n'.join(lines) + '\n'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
