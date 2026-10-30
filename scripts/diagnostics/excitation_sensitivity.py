# SPDX-License-Identifier: GPL-3.0-or-later
"""Does an excitation-conditioned vocoder actually use its F0 prior? Decode clean test speech from its
own mel (upper bound) with the teacher F0 scaled by several factors and report PESQ per gender. If the
scores barely move, the vocoder ignores the prior and the excitation experiments were not a real test.

  python -m scripts.diagnostics.excitation_sensitivity --config configs/experiments/ja_mel_pretrain_istft_excap_cuda.yaml \
      --ckpt runs/mel_istft_excap/last.pt --scales 1.0 0.8 1.25 0 --utts 30
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pesq import pesq

from neurodec.config import load_config
from neurodec.data.corpus import Utterance
from neurodec.data.dataset import Transmissions, frame_f0
from neurodec.data.session import plan_sessions
from neurodec.device import pick_device
from neurodec.eval.metrics import pooled_lag, shift
from neurodec.models.decoder import FRAME
from neurodec.train import load_model
from scripts.gender_summary import gender_table

SR = 8000


@torch.no_grad()
def decode_scaled(G, target, device, scale):
    x = torch.from_numpy(target)
    x = torch.nn.functional.pad(x, (0, (-len(x)) % FRAME))
    teacher = torch.from_numpy(frame_f0(x.numpy(), len(x) // FRAME))[None].to(device)
    f0 = teacher[..., 0] * scale
    kw = dict(f0=f0, voiced=(f0 > 0).float(), aperiodicity=teacher[..., 1:])
    return np.clip(G(x[None].to(device), **kw)[0].float().cpu().numpy()[: len(target)], -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.8, 1.25, 0.0])
    ap.add_argument('--utts', type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device()
    G = load_model(a.ckpt, device)[0]
    if not G.uses_excitation:
        raise SystemExit('checkpoint has no excitation')
    tx = Transmissions(Path(cfg['data_dir']) / 'test_clean')
    test = [Utterance(**u) for u in json.loads((Path(cfg['data_dir']) / 'split.json').read_text(encoding='utf-8'))['test']]
    speakers = [[u.speaker for u in s] for s in plan_sessions(test, cfg['session']['seconds'])]
    table = gender_table(cfg)
    acc = defaultdict(list)
    done = 0
    for i in range(len(tx)):
        if done >= a.utts:
            break
        target = tx.target(i)
        spans = np.load(tx.dir / tx.rows[i]['file'])['spans']
        outs = {}
        for sc in a.scales:
            torch.manual_seed(cfg['seed'] + i)
            y = decode_scaled(G, target, device, sc)
            outs[sc] = shift(y, pooled_lag([target], [y]), len(target))
        for u, spk in enumerate(speakers[i]):
            if done >= a.utts:
                break
            s, e = spans[u] + tx.offset
            if s < 0 or e > len(target):
                continue
            g = table.get(spk, '?')
            for sc, y in outs.items():
                try:
                    acc[(g, sc)].append(float(pesq(SR, target[s:e], y[s:e], 'nb')))
                except Exception:
                    pass
            done += 1
    print('| gender | ' + ' | '.join(f'F0 x{sc:g}' for sc in a.scales) + ' |')
    print('|---|' + '---:|' * len(a.scales))
    for g in sorted({k[0] for k in acc}):
        print(f'| {g} (n={len(acc[(g, a.scales[0])])}) | ' + ' | '.join(f'{np.mean(acc[(g, sc)]):.3f}' for sc in a.scales) + ' |')


if __name__ == '__main__':
    main()
