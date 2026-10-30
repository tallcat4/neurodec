# SPDX-License-Identifier: GPL-3.0-or-later
"""Two-stage decoding without any training: bits -> predicted log-mel (frontend's auxiliary
head of a bits decoder) -> a mel-conditioned backend trained only on clean speech.

The mel backend is the one the user judged artifact-free from true mel; feeding it the
frontend's *predicted* mel tells whether the residual artifact lives in the prediction
(then the two-stage output is also rough) or in joint fine-tuning of the vocoder on noisy
latents (then the two-stage output is cleaner). Writes target / ambe2-c / proposed sets.

  python -m scripts.diagnostics.listen_two_stage --config configs/experiments/b49_bittoken_wide_istft_cuda.yaml --lang ja \
      --bits-ckpt runs/joint_b49_wide/step30000.pt --mel-ckpt runs/mel_istft_long/last.pt --utts 6 --out results/listen_ja_two_stage
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
from neurodec.infer import decode
from neurodec.models.decoder import FRAME
from neurodec.train import build_model, load_model, transfer_frontend


@torch.no_grad()
def decode_two_stage(G_bits, G_mel, bits, device, chunk_frames=500, context=50, group=2):
    """bits [T, S] -> wav, through predicted log-mel and the mel backend's projection."""
    T = bits.shape[0]
    pad = (-T) % group
    b = np.pad(bits, ((0, pad), (0, 0))).astype(np.float32)
    Tp = b.shape[0]
    chunk_frames -= chunk_frames % group
    context -= context % group
    out = np.zeros(Tp * FRAME, np.float32)
    for s in range(0, Tp, chunk_frames):
        a, e = max(0, s - context), min(Tp, s + chunk_frames + context)
        x = torch.from_numpy(b[a:e])[None].to(device)
        _, mel_pred = G_bits.frontend.forward_with_aux(x)  # [1, 80, frames]
        cond = G_mel.frontend.proj(mel_pred)  # the mel backend's own input projection
        y = G_mel.generator(cond)[0].float().cpu().numpy()
        out[s * FRAME : min(Tp, s + chunk_frames) * FRAME] = y[(s - a) * FRAME : (min(Tp, s + chunk_frames) - a) * FRAME]
    return np.clip(out[: T * FRAME], -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bits-ckpt', required=True)
    ap.add_argument('--mel-ckpt', required=True)
    ap.add_argument('--lang', default='ja')
    ap.add_argument('--cond', default='clean')
    ap.add_argument('--utts', type=int, default=6)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = pick_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx = Transmissions(Path(cfg['data_dir']) / f'test_{a.cond}')
    picked = [i for i, r in enumerate(tx.rows) if r['langs'] == [a.lang]]
    rx = Receiver.from_config(cfg)
    g = cfg['blackbox']['block_frames']
    ck = torch.load(a.bits_ckpt, map_location='cpu', weights_only=False)
    frontend_only = all(k.startswith('frontend.') for k in ck['G'])
    if frontend_only:  # probe checkpoint: no trained backend, so only the two-stage path is meaningful
        G_bits = build_model(ck['meta']['cfg'], 'bits')
        transfer_frontend(G_bits, ck)
        G_bits = G_bits.to(device).eval()
    else:
        G_bits, _ = load_model(a.bits_ckpt, device)
    G_mel, meta = load_model(a.mel_ckpt, device)
    assert meta['frontend'] == 'mel' and G_mel.backend_kind == 'istft'
    rows, written = [], 0
    for i in picked:
        if written >= a.utts:
            break
        bits, target = tx.bits[i], tx.target(i)
        spans = np.load(tx.dir / tx.rows[i]['file'])['spans']
        outs = {'2_ambe2-c': rx.receive(bits)}
        torch.manual_seed(cfg['seed'] + i)
        if not frontend_only:
            outs['3_proposed_joint'] = decode(G_bits, bits, device, chunk_frames=500, group=g)
        outs['3_proposed_two_stage'] = decode_two_stage(G_bits, G_mel, bits, device, group=g)
        lags = {k: pooled_lag([target], [y]) for k, y in outs.items()}
        for (s, e) in spans:
            if written >= a.utts:
                break
            s, e = s + tx.offset, e + tx.offset
            if s < 0 or e > len(target):
                continue
            tag = f'{a.lang}_{written:02d}_tx{i:03d}'
            sf.write(out / f'{tag}_1_target.wav', target[s:e], 8000)
            for k, y in outs.items():
                y = shift(y, lags[k], len(target))
                sf.write(out / f'{tag}_{k}.wav', y[s:e], 8000)
                m = all_metrics(target[s:e], y[s:e])
                rows.append(dict(utt=tag, system=k, **m))
            written += 1
    (out / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    lines = ['| system | n | PESQ | STOI | log-mel dB | HNR dB |', '|---|---|---|---|---|---|']
    for sname in sorted({r['system'] for r in rows}):
        rs = [r for r in rows if r['system'] == sname]
        lines.append(f"| {sname} | {len(rs)} | " + ' | '.join(f"{np.nanmean([r[k] for r in rs]):.3f}" for k in ('pesq_nb', 'stoi', 'logmel_db', 'hnr_db')) + ' |')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
