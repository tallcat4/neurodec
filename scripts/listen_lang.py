# SPDX-License-Identifier: GPL-3.0-or-later
"""Listening set for one language: source, receiver and model outputs for several utterances.

Picks the first test transmissions whose utterances are all in `--lang`, decodes them with
each checkpoint (seeded like run_eval, chunk 500), removes each system's pooled lag, and
writes utterance WAVs plus a metrics table. Evaluation-only use of the receiver.

  python -m scripts.listen_lang --config configs/experiments/b49_bittoken_wide_istft_cuda.yaml --lang ja \
      --ckpt runs/joint_b49_wide/step30000.pt --utts 6 --out results/listen_ja_wide
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from neurodec.blackbox import Receiver
from neurodec.config import load_config
from neurodec.data.dataset import Transmissions
from neurodec.device import pick_device
from neurodec.eval.metrics import all_metrics, pooled_lag, shift
from neurodec.eval.run_eval import decode_mel
from neurodec.infer import decode
from neurodec.train import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', action='append', required=True)
    ap.add_argument('--lang', default='ja')
    ap.add_argument('--cond', default='clean')
    ap.add_argument('--utts', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gender', default=None, choices=[None, 'M', 'F'],
                    help='JVS only: keep transmissions whose speakers all have this gender (gender_f0range.txt)')
    ap.add_argument('--with-upper-bound', action='store_true',
                    help='also write mel-conditioned checkpoints (diagnostic only; listening sets ship target, ambe2-c, proposed)')
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx = Transmissions(Path(cfg['data_dir']) / f'test_{a.cond}')
    picked = [i for i, r in enumerate(tx.rows) if r['langs'] == [a.lang]]
    span_speakers = None
    if a.gender:
        # Sessions are packed greedily in list order (data.session.plan_sessions), so the u-th
        # span of transmission i is the u-th utterance of session i of the test list.
        from neurodec.data.session import plan_sessions
        from neurodec.data.corpus import Utterance
        test = [Utterance(**u) for u in json.loads((Path(cfg['data_dir']) / 'split.json').read_text(encoding='utf-8'))['test']]
        span_speakers = [[u.speaker for u in sess] for sess in plan_sessions(test, cfg['session']['seconds'])]
        table = {}
        root = next(c['root'] for c in cfg['corpora'] if c['type'] == 'jvs')
        for line in Path(root).joinpath('gender_f0range.txt').read_text(encoding='utf-8').splitlines()[1:]:
            parts = line.split()
            table['jvs/' + parts[0]] = parts[1]
    rx = Receiver.from_config(cfg)
    g = cfg['blackbox']['block_frames']
    from neurodec.blackbox.encoder import FORMATS
    own_decoder = {'ambe': 'ambe2-c', 'mrl': 'mrl-celp-c'}.get(FORMATS[cfg['bits']]['codec'], 'codec-decoder')
    models = {Path(c).parent.name + '@' + Path(c).stem: load_model(c, device) for c in a.ckpt}
    rows, written = [], 0
    for i in picked:
        if written >= a.utts:
            break
        bits, target = tx.bits[i], tx.target(i)
        spans = np.load(tx.dir / tx.rows[i]['file'])['spans']
        outs = {'reference': rx.receive(bits)}
        for name, (G, meta) in models.items():
            torch.manual_seed(cfg['seed'] + i)
            if meta['frontend'] == 'mel':  # upper bound: synthesis from the clean source's own mel
                if a.with_upper_bound:
                    outs['DIAG_mel_ub(' + name + ')'] = decode_mel(G, target, device)
            else:
                outs[name] = decode(G, bits, device, chunk_frames=500, group=g)
        lags = {k: pooled_lag([target], [y]) for k, y in outs.items()}
        for u, (s, e) in enumerate(spans):
            if written >= a.utts:
                break
            if span_speakers is not None and table.get(span_speakers[i][u]) != a.gender:
                continue
            s, e = s + tx.offset, e + tx.offset
            if s < 0 or e > len(target):
                continue
            tag = f'{a.lang}{a.gender or ""}_{written:02d}_tx{i:03d}'
            sf.write(out / f'{tag}_1_target.wav', target[s:e], 8000)
            for k, y in outs.items():
                y = shift(y, lags[k], len(target))
                label = f'2_{own_decoder}' if k == 'reference' else ('3_proposed_' + k if not k.startswith('DIAG') else k)
                sf.write(out / f'{tag}_{label}.wav', y[s:e], 8000)
                m = all_metrics(target[s:e], y[s:e])
                rows.append(dict(utt=tag, system=k, speakers=tx.rows[i]['speakers'], **m))
                print(tag, f'{k:40s}', ' '.join(f"{kk} {v:.2f}" for kk, v in m.items() if kk in ('pesq_nb', 'stoi', 'logmel_db', 'hnr_db')), flush=True)
            written += 1
    (out / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    systems = sorted({r['system'] for r in rows})
    lines = ['| system | n | PESQ | STOI | log-mel dB | HNR dB |', '|---|---|---|---|---|---|']
    for sname in systems:
        rs = [r for r in rows if r['system'] == sname]
        lines.append(f"| {sname} | {len(rs)} | " + ' | '.join(f"{np.nanmean([r[k] for r in rs]):.3f}" for k in ('pesq_nb', 'stoi', 'logmel_db', 'hnr_db')) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
