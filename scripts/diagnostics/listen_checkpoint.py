# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the first held-out utterance decoded by intermediate checkpoints (quick listening).

Same utterance, alignment and cropping rule as neurodec.eval.run_eval (transmission 0,
utterance 0 of test_clean), with the lag estimated on that transmission only. This is for
listening while a run is still training, not a replacement for the full evaluation.

  python -m scripts.diagnostics.listen_checkpoint --config configs/experiments/ja_check_cuda.yaml --out results/listen_x \
      --ckpt runs/joint_mlp_istft/step10000.pt [--ckpt ...]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from neurodec.config import load_config
from neurodec.data.dataset import Transmissions
from neurodec.device import pick_device
from neurodec.eval.metrics import all_metrics, pooled_lag, shift
from neurodec.infer import decode
from neurodec.train import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cond', default='clean')
    ap.add_argument('--tx', type=int, default=0)
    ap.add_argument('--utt', type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx = Transmissions(Path(cfg['data_dir']) / f'test_{a.cond}')
    bits = tx.bits[a.tx]
    target = tx.target(a.tx)
    spans = np.load(tx.dir / tx.rows[a.tx]['file'])['spans']
    s, e = spans[a.utt] + tx.offset
    sf.write(out / 'target.wav', target[s:e], 8000)
    g = cfg['blackbox']['block_frames']
    for ck in a.ckpt:
        G, meta = load_model(ck, device)
        torch.manual_seed(cfg['seed'] + a.tx)
        y = decode(G, bits, device, group=g)
        k = pooled_lag([target], [y])
        y = shift(y, k, len(target))
        name = Path(ck).parent.name + '@' + Path(ck).stem
        sf.write(out / f'{name}.wav', y[s:e], 8000)
        m = all_metrics(target[s:e], y[s:e])
        print(name, f"lag {k}", {k2: round(float(v), 3) for k2, v in m.items()}, flush=True)


if __name__ == '__main__':
    main()
