# SPDX-License-Identifier: GPL-3.0-or-later
"""One command for a new black-box codec: build the queue plan for the whole FINAL procedure and
optionally start it unattended.

Stages (each a job of scripts.overnight_queue, run one after another on the GPU):
  0. build     (optional, CPU)  simulate the recording campaign  -> data/<campaign>/{train,test_*}
  1. teachers  (CPU)            F0 / voicing / band-aperiodicity teachers from the CLEAN source
  2. probe     (GPU, ~10 min)   bits -> log-mel frontend probe with periodicity heads
  3. joint     (GPU, ~70 min)   frontend trained through the frozen clean-speech vocoder, then evaluation
  4. listen    (GPU, ~2 min)    per-gender listening sets: target / the codec's own decoder / proposed

Prerequisites: a format entry for the codec in neurodec/blackbox/encoder.py, a receiver command in
neurodec/blackbox/radio.py, a config like configs/final/ambe49_decoder.yaml, and a clean-speech vocoder
checkpoint (train one with configs/final/clean_speech_vocoder.yaml, or reuse an existing run's last.pt;
the exported inference-only checkpoints have no critics, so the critics then start fresh).

  python -m scripts.new_codec_pipeline --config configs/final/ambe49_decoder.yaml --name ambe49_demo \\
      --vocoder runs/mel_istft_excap/last.pt --build --start
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from neurodec.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='decoder config (see configs/final/)')
    ap.add_argument('--name', required=True, help='prefix for runs/<name>_* and results/listen_<name>_*')
    ap.add_argument('--vocoder', required=True, help='clean-speech vocoder checkpoint (frozen behind the mel bridge)')
    ap.add_argument('--build', action='store_true', help='also build the campaign first (CPU, hours for 12 h)')
    ap.add_argument('--build-workers', type=int, default=16)
    ap.add_argument('--hours', type=float, default=None, help='use only the first N hours of the campaign')
    ap.add_argument('--probe-steps', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bers', nargs='*', default=['0', '0.01', '0.03'])
    ap.add_argument('--listen-lang', default='ja')
    ap.add_argument('--listen-utts', type=int, default=6)
    ap.add_argument('--start', action='store_true', help='start scripts.overnight_queue detached')
    a = ap.parse_args()

    cfg = load_config(a.config)
    data_dir = cfg['data_dir']
    if not cfg['model'].get('f0_head'):
        raise SystemExit('the final recipe needs model.f0_head: true (periodicity heads); see configs/final/ambe49_decoder.yaml')
    for run in [f'runs/{a.name}_probe', f'runs/{a.name}_joint', f'runs/{a.name}_joint_job']:
        if Path(run).exists():
            raise SystemExit(f'{run} exists; choose another --name (existing runs are never overwritten)')
    vocoder = torch.load(a.vocoder, map_location='cpu', weights_only=False)
    if vocoder.get('meta', {}).get('frontend') != 'mel':
        raise SystemExit('--vocoder must be a mel-conditioned (clean-speech) checkpoint')
    has_d = 'D' in vocoder
    if not has_d:
        print('note: the vocoder checkpoint has no critics (inference-only export); the critics start fresh', file=sys.stderr)

    hours = ['--hours', str(a.hours)] if a.hours else []
    groups = []
    if a.build:
        groups.append({'name': 'build', 'max_concurrent': 1, 'jobs': [
            {'name': f'{a.name}_build', 'module': 'neurodec.data.build',
             'args': ['--config', a.config, '--workers', str(a.build_workers)]}]})
    groups.append({'name': 'teachers', 'max_concurrent': 1, 'jobs': [
        {'name': f'{a.name}_teachers', 'module': 'scripts.precompute_f0',
         'args': ['--config', a.config, '--workers', str(a.build_workers)]}]})
    groups.append({'name': 'probe', 'max_concurrent': 1, 'jobs': [
        {'name': f'{a.name}_probe_job', 'module': 'scripts.probe_frontend', 'args': [
            '--config', a.config, '--run', f'runs/{a.name}_probe', '--steps', str(a.probe_steps),
            '--val-interval', '500', '--f0', str(cfg['loss'].get('lambda_f0', 1.0)),
            '--vuv', str(cfg['loss'].get('lambda_vuv', 0.5)), '--val-dirs', f'{data_dir}/test_sim', *hours]}]})
    joint_args = ['--job', f'runs/{a.name}_joint_job', '--config', a.config, '--run', f'runs/{a.name}_joint',
                  '--steps', str(a.steps), '--init-frontend-from', f'runs/{a.name}_probe/last.pt',
                  '--init-from', a.vocoder, *(['--init-disc-from', a.vocoder] if has_d else []),
                  '--bers', *a.bers, *hours]
    groups.append({'name': 'joint', 'max_concurrent': 1, 'jobs': [
        {'name': f'{a.name}_joint_job', 'module': 'scripts.run_stage', 'args': joint_args}]})
    groups.append({'name': 'listen', 'max_concurrent': 1, 'jobs': [
        {'name': f'listen_{a.name}_{g}', 'module': 'scripts.listen_lang', 'args': [
            '--config', a.config, '--lang', a.listen_lang, '--gender', G,
            '--ckpt', f'runs/{a.name}_joint/step{a.steps}.pt', '--utts', str(a.listen_utts),
            '--out', f'results/listen_{a.name}_{g}']} for g, G in [('male', 'M'), ('female', 'F')]]})

    state = Path(f'runs/{a.name}_pipeline')
    state.mkdir(parents=True, exist_ok=True)
    plan = state / 'plan.json'
    plan.write_text(json.dumps({'groups': groups}, indent=1), encoding='utf-8')
    cmd = [sys.executable, '-m', 'scripts.overnight_queue', '--plan', str(plan), '--state', str(state)]
    print('plan:', plan)
    print('command:', ' '.join(cmd))
    if a.start:
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == 'nt' else 0
        with open(state / 'stdout.log', 'ab') as out, open(state / 'stderr.log', 'ab') as err:
            p = subprocess.Popen(cmd, stdout=out, stderr=err, creationflags=flags, cwd=os.getcwd(),
                                 env=dict(os.environ, PYTHONUTF8='1'))
        print(f'started queue pid {p.pid}; progress in {state}/queue.log and runs/{a.name}_*/status.json')


if __name__ == '__main__':
    main()
