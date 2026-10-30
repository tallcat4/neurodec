# SPDX-License-Identifier: GPL-3.0-or-later
"""Harmonic alignment per gender: how much of each band's energy sits between the harmonics of the
clean source's F0 (band aperiodicity: 0 = perfectly harmonic at the target's F0, 1 = noise-like).

Decodes test utterances of one gender with each bits checkpoint (and, with --mel-ckpt, the
clean-mel upper bound with teacher F0), and compares with the target and the codec's own decoder.
A system whose harmonics are blurred or sit at a slightly wrong F0 shows a higher value than the
target even when its own HNR is fine, which is the "robotic" fine-structure error mel loss misses.

  python -m scripts.diagnostics.harmonic_alignment --config configs/experiments/b49_bittoken_wide_bridge_f0_cuda.yaml --gender M \
      --ckpt runs/joint_b49_wide_bridge_f0/step30000.pt --mel-ckpt runs/mel_istft_long/last.pt --utts 50
"""
import argparse
import json
from collections import defaultdict
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
from neurodec.eval.run_eval import decode_mel
from neurodec.infer import decode
from neurodec.train import load_model
from scripts.gender_summary import gender_table
from scripts.precompute_f0 import band_aperiodicity, SR, FRAME

BANDS = ['0-0.8k', '0.8-1.6k', '1.6-2.4k', '2.4-3.2k', '3.2-4k']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--mel-ckpt', action='append', default=[])
    ap.add_argument('--gender', default='M', choices=['M', 'F'])
    ap.add_argument('--cond', default='clean')
    ap.add_argument('--utts', type=int, default=50)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device()
    tx = Transmissions(Path(cfg['data_dir']) / f'test_{a.cond}')
    test = [Utterance(**u) for u in json.loads((Path(cfg['data_dir']) / 'split.json').read_text(encoding='utf-8'))['test']]
    speakers = [[u.speaker for u in s] for s in plan_sessions(test, cfg['session']['seconds'])]
    table = gender_table(cfg)
    rx = Receiver.from_config(cfg)
    g = cfg['blackbox']['block_frames']
    models = {Path(c).parent.name + '@' + Path(c).stem: load_model(c, device)[0] for c in a.ckpt}
    mels = {'mel_ub:' + Path(c).parent.name: load_model(c, device)[0] for c in a.mel_ckpt}
    acc = defaultdict(list)
    done = 0
    for i in range(len(tx)):
        if done >= a.utts:
            break
        wanted = [u for u, s in enumerate(speakers[i]) if table.get(s) == a.gender]
        if not wanted:
            continue
        bits, target = tx.bits[i], tx.target(i)
        spans = np.load(tx.dir / tx.rows[i]['file'])['spans']
        outs = {'reference': rx.receive(bits)}
        for name, G in models.items():
            torch.manual_seed(cfg['seed'] + i)
            outs[name] = decode(G, bits, device, chunk_frames=500, group=g)
        for name, G in mels.items():
            torch.manual_seed(cfg['seed'] + i)
            outs[name] = decode_mel(G, target, device)
        lags = {k: pooled_lag([target], [y]) for k, y in outs.items()}
        outs = {k: shift(y, lags[k], len(target)) for k, y in outs.items()}
        for u in wanted:
            if done >= a.utts:
                break
            s, e = spans[u] + tx.offset
            if s < 0 or e > len(target):
                continue
            x = target[s:e].astype(np.float64)
            f0, t = pyworld.dio(x, SR, f0_floor=60, f0_ceil=500, frame_period=FRAME * 1000 / SR)
            f0 = pyworld.stonemask(x, f0, t, SR).astype(np.float32)
            voiced = f0 > 0
            if voiced.sum() < 10:
                continue
            for name, y in [('target', target), *outs.items()]:
                ap_ = band_aperiodicity(y[s:e].astype(np.float64), f0)
                acc[name].append(ap_[voiced[:len(ap_)]].mean(axis=0))
            done += 1
    lines = [f'| system ({a.gender}, n={done}) | ' + ' | '.join(BANDS) + ' | mean |', '|---|' + '---:|' * (len(BANDS) + 1)]
    for name, v in acc.items():
        m = np.mean(v, axis=0)
        lines.append(f'| {name} | ' + ' | '.join(f'{x:.3f}' for x in m) + f' | {m.mean():.3f} |')
    text = '\n'.join(lines) + '\n'
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
