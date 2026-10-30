# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare on the test transmissions (speakers not in training).

Splits reused for model selection are development evaluations, not untouched tests.

  reference  the receiving radio's own decoder (ambe2wav behind the black box)
  proposed   neural decoder(s) from bits
  mel_ub     mel-conditioned PWG on the clean source (upper bound, optional)

Each utterance is scored separately against the clean source. Each system's constant
delay is removed first (cross-correlation summed over all test transmissions, +-250 ms).

  python -m neurodec.eval.run_eval --config configs/bits49.yaml --ckpt runs/b49_12h/last.pt \
      [--ckpt more.pt ...] [--mel-ckpt runs/mel/last.pt] [--ber 0 0.01 0.03] --out results/b49
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..blackbox import Receiver
from ..config import load_config
from ..data.dataset import Transmissions, flip_bits
from ..device import pick_device
from ..infer import decode
from .metrics import all_metrics, pooled_lag, shift

FRAME = 160


def evaluate(cfg, ckpts, mel_ckpt, bers, out: Path, max_utts=None, samples=4, seed=None, chunk_frames=1500):
    from ..train import load_model

    device = pick_device()
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg['seed'] if seed is None else seed
    (out / 'evaluation.json').write_text(json.dumps(dict(
        checkpoints=list(ckpts), data_dir=cfg['data_dir'], conditions=cfg['test_conditions'],
        bers=list(bers), seed=seed, noise_protocol='torch seed + transmission index, reset before each model',
        decode_chunk_frames=chunk_frames,
        torch=str(torch.__version__), device=str(device)), indent=2)+'\n', encoding='utf-8', newline='\n')
    models = {Path(c).parent.name + ("" if Path(c).stem == "last" else f"@{Path(c).stem}"): load_model(c, device)[0]
              for c in ckpts}
    mel = load_model(mel_ckpt, device)[0] if mel_ckpt else None
    rx = Receiver.from_config(cfg)
    g = cfg["blackbox"]["block_frames"]
    rows = []
    timings = []
    for cond in cfg["test_conditions"]:
        tx = Transmissions(Path(cfg["data_dir"]) / f"test_{cond}")
        idx, n_utts = [], 0
        for i in range(len(tx)):
            idx.append(i)
            n_utts += len(np.load(tx.dir / tx.rows[i]["file"])["spans"])
            if max_utts and n_utts >= max_utts:
                break
        for ber in bers:
            print(f'[eval] {cond} BER={ber:g}: decoding {len(idx)} transmissions, {len(models)} neural models', flush=True)
            outs = defaultdict(dict)  # system -> tx -> wav
            for position, i in enumerate(idx, start=1):
                bits = flip_bits(tx.bits[i], ber, np.random.default_rng(i))
                outs["reference"][i] = rx.receive(bits)
                for name, G in models.items():
                    # Model construction and checkpoint ordering must not change
                    # the noise used to score stochastic waveform generators.
                    torch.manual_seed(seed + i)
                    started = time.perf_counter()
                    outs[name][i] = decode(G, bits, device, chunk_frames=chunk_frames, group=g)
                    # decode() transfers the waveform back to CPU, so CUDA work
                    # is complete before stopping this end-to-end decode timer.
                    timings.append(dict(cond=cond, ber=ber, system=name, tx=i,
                                        audio_seconds=len(bits) * FRAME / 8000,
                                        decode_seconds=time.perf_counter() - started))
                if mel is not None and ber == 0:
                    torch.manual_seed(seed + i)
                    outs["mel_ub"][i] = decode_mel(mel, tx.target(i), device)
                if position % 4 == 0 or position == len(idx):
                    print(f'[eval] {cond} BER={ber:g}: decoded {position}/{len(idx)}', flush=True)
            for sys_name, ys in outs.items():
                print(f'[eval] {cond} BER={ber:g}: scoring {sys_name}', flush=True)
                # A decoder's delay is constant: one lag per system from all transmissions.
                k = pooled_lag([tx.target(i) for i in idx], [ys[i] for i in idx])
                for i in idx:
                    target = tx.target(i)
                    y = shift(ys[i], k, len(target))
                    spans = np.load(tx.dir / tx.rows[i]["file"])["spans"]
                    for u, (a, b) in enumerate(spans):
                        a, b = a + tx.offset, b + tx.offset  # source -> target coordinates
                        if a < 0 or b > len(target):
                            continue
                        ref, deg = target[a:b], y[a:b]
                        rows.append({"cond": cond, "ber": ber, "system": sys_name, "tx": i, "utt": u,
                                     "lag": k, **all_metrics(ref, deg)})
                        if ber == 0 and u == 0 and i < samples:
                            sf.write(out / f"{cond}_{i:03d}_{sys_name}.wav", deg, 8000)
                            sf.write(out / f"{cond}_{i:03d}_target.wav", ref, 8000)
    (out / "per_utterance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding='utf-8', newline='\n')
    (out / 'decode_timing.json').write_text(json.dumps(dict(
        note='Actual chunked decode, batch 1; includes CPU/GPU transfers; excludes model loading, receiver and metrics. First-call overhead included.',
        device=str(device), rows=timings), indent=2) + '\n', encoding='utf-8', newline='\n')
    table = summarize(rows)
    (out / "summary.md").write_text(table, encoding='utf-8', newline='\n')
    print(table)
    return rows


@torch.no_grad()
def decode_mel(G, target, device, chunk=480_000):
    out = []
    for s in range(0, len(target), chunk):
        x = torch.from_numpy(target[s : s + chunk])
        x = torch.nn.functional.pad(x, (0, (-len(x)) % FRAME))
        kw = {}
        if getattr(G, "uses_excitation", False):  # upper bound: F0 of the clean source itself
            from ..data.dataset import frame_f0
            teacher = torch.from_numpy(frame_f0(x.numpy(), len(x) // FRAME))[None].to(device)
            kw = dict(f0=teacher[..., 0], voiced=(teacher[..., 0] > 0).float(), aperiodicity=teacher[..., 1:])
        out.append(G(x[None].to(device), **kw)[0].float().cpu().numpy()[: min(chunk, len(target) - s)])
    return np.clip(np.concatenate(out), -1, 1)


def summarize(rows) -> str:
    keys = ["pesq_nb", "stoi", "logmel_db", "f0_rmse_cent", "vuv_err", "hnr_db", "buzz50_db"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cond"], r["ber"], r["system"])].append(r)
    lines = ["| cond | BER | system | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
    for (cond, ber, sysn), rs in sorted(groups.items()):
        vals = [np.nanmean([r[k] for r in rs]) for k in keys]
        lines.append(f"| {cond} | {ber:g} | {sysn} | {len(rs)} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--mel-ckpt", default=None)
    ap.add_argument("--ber", type=float, nargs="+", default=[0.0])
    ap.add_argument("--max-utts", type=int, default=None, help="per condition")
    ap.add_argument("--out", required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--decode-chunk', type=int, default=1500, help='frames per decode chunk (smaller = less VRAM)')
    a = ap.parse_args()
    evaluate(load_config(a.config), a.ckpt, a.mel_ckpt, a.ber, Path(a.out), a.max_utts, seed=a.seed, chunk_frames=a.decode_chunk)


if __name__ == "__main__":
    main()
