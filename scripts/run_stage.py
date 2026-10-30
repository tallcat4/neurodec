# SPDX-License-Identifier: GPL-3.0-or-later
"""Run one bounded training stage (step or wall-time budget), then evaluate it.

Stages: mel-conditioned backend pre-training (--frontend mel), or a bits decoder that may
start from a probe frontend (--init-frontend-from), a pre-trained backend (--init-from)
and its critics (--init-disc-from). Writes status.json like scripts.train_for_hour.

  python -m scripts.run_stage --job runs/mel_pwg_job --config configs/experiments/ja_mel_pretrain_cuda.yaml \
      --run runs/mel_pwg --frontend mel --steps 20000
  python -m scripts.run_stage --job runs/joint_bittoken_job --config configs/experiments/ja_bittoken_multidisc_cuda.yaml \
      --run runs/joint_bittoken --steps 30000 --init-frontend-from runs/probe_bittoken/last.pt \
      --init-from runs/mel_pwg/last.pt --init-disc-from runs/mel_pwg/last.pt \
      --comparison runs/ja_pwg_multi_from46000_v1/step52568.pt --mel-ckpt runs/mel_pwg/last.pt
"""
import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import torch

from neurodec.config import load_config
from neurodec.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--frontend', default='bits', choices=['bits', 'mel'])
    ap.add_argument('--steps', type=int, required=True, help='absolute final step of the run')
    ap.add_argument('--max-seconds', type=float, default=None)
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--init-from', default=None)
    ap.add_argument('--init-disc-from', default=None)
    ap.add_argument('--init-frontend-from', default=None)
    ap.add_argument('--comparison', action='append', default=[], help='evaluation-only checkpoints')
    ap.add_argument('--mel-ckpt', default=None, help='mel-conditioned upper bound to score alongside')
    ap.add_argument('--save-interval', type=int, default=5000)
    ap.add_argument('--eval-interval', type=int, default=5000)
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--disc-start', type=int, default=None, help='override train.disc_start (smoke tests)')
    ap.add_argument('--max-utts', type=int, default=None, help='limit evaluation utterances (smoke tests)')
    ap.add_argument('--decode-chunk', type=int, default=500, help='frames per decode chunk; 500 keeps PWG decoding under ~1 GB')
    ap.add_argument('--bers', type=float, nargs='*', default=[0.0], help='bit error rates to evaluate')
    a = ap.parse_args()
    job = Path(a.job)
    job.mkdir(parents=True, exist_ok=True)
    status = dict(pid=os.getpid(), started_at=time.time(), steps=a.steps, max_seconds=a.max_seconds)

    def update(stage, **fields):
        status.update(stage=stage, updated_at=time.time(), **fields)
        pending = job / 'status.json.tmp'
        pending.write_text(json.dumps(status, indent=2), encoding='utf-8')
        pending.replace(job / 'status.json')
        print(f'JOB {stage}: {fields}', flush=True)

    try:
        cfg = load_config(a.config)
        cfg['train'].update(steps=a.steps, eval_interval=a.eval_interval, save_interval=a.save_interval,
                            keep_interval=a.save_interval)
        if a.disc_start is not None:
            cfg['train']['disc_start'] = a.disc_start
        run = Path(a.run)
        for p in [a.init_from, a.init_disc_from, a.init_frontend_from, a.mel_ckpt, *a.comparison]:
            if p and not Path(p).is_file():
                raise FileNotFoundError(p)
        update('training', run=str(run), config=a.config, frontend=a.frontend, init_from=a.init_from,
               init_disc_from=a.init_disc_from, init_frontend_from=a.init_frontend_from,
               disc_start=cfg['train']['disc_start'])
        last = train(cfg, run, frontend=a.frontend, hours=a.hours, init_from=a.init_from,
                     max_seconds=a.max_seconds, init_frontend_from=a.init_frontend_from,
                     init_disc_from=a.init_disc_from)
        step = torch.load(last, map_location='cpu', weights_only=False)['step']
        final = run / f'step{step}.pt'
        if a.no_eval:
            update('completed', final_step=step, checkpoint=str(final), finished_at=time.time())
            return
        out = Path('results') / job.name
        update('evaluating', final_step=step, checkpoint=str(final), results=str(out))
        from neurodec.eval.run_eval import evaluate
        torch.manual_seed(cfg['seed'])
        if a.frontend == 'mel':
            checkpoints, mel_ckpt = list(dict.fromkeys(a.comparison)), str(final)
        else:
            checkpoints, mel_ckpt = list(dict.fromkeys([str(final), *a.comparison])), a.mel_ckpt
        evaluate(cfg, checkpoints, mel_ckpt, a.bers, out, max_utts=a.max_utts, chunk_frames=a.decode_chunk)
        listen = out / 'listen'
        listen.mkdir(exist_ok=True)
        for wav in sorted(out.glob('clean_000_*.wav')):
            shutil.copyfile(wav, listen / wav.name.removeprefix('clean_000_'))
        update('completed', summary=str(out / 'summary.md'), listening_dir=str(listen), finished_at=time.time())
    except BaseException:
        update('failed', error=traceback.format_exc(), finished_at=time.time())
        raise


if __name__ == '__main__':
    main()
