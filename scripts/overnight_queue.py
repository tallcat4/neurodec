# SPDX-License-Identifier: GPL-3.0-or-later
"""Unattended job queue: launch groups of stage jobs with bounded concurrency.

Each group waits for the jobs of the previous group (and any `wait_for` status files)
before starting. Jobs are `python -m <module> <args>` with per-job logs under
runs/<job>/, exactly like start-stage.ps1. Existing job directories are refused.

  python -m scripts.overnight_queue --plan configs/overnight_20261028.json --state runs/overnight_20261028
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def status_done(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('stage') in ('completed', 'failed')
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--max-concurrent', type=int, default=2)
    a = ap.parse_args()
    root = Path.cwd()
    plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
    state_dir = Path(a.state)
    state_dir.mkdir(parents=True, exist_ok=True)
    state = dict(plan=a.plan, started_at=time.time(), groups=[])
    env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TEMP=str(root / '.tmp'), TMP=str(root / '.tmp'))

    def save():
        pending = state_dir / 'state.json.tmp'
        pending.write_text(json.dumps(state, indent=2), encoding='utf-8')
        pending.replace(state_dir / 'state.json')

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        (state_dir / 'queue.log').open('a', encoding='utf-8').write(line + '\n')

    for group in plan['groups']:
        entry = dict(name=group['name'], jobs={}, started_at=time.time())
        state['groups'].append(entry)
        save()
        for wait in group.get('wait_for', []):
            while not status_done(Path(wait)):
                log(f"group {group['name']}: waiting for {wait}")
                time.sleep(60)
        running = {}
        queue = list(group['jobs'])
        while queue or running:
            for name, proc in list(running.items()):
                code = proc.poll()
                if code is not None:
                    entry['jobs'][name]['exit'] = code
                    entry['jobs'][name]['finished_at'] = time.time()
                    log(f"job {name} exited with {code}")
                    del running[name]
            while queue and len(running) < group.get('max_concurrent', a.max_concurrent):
                job = queue.pop(0)
                name = job['name']
                job_dir = root / 'runs' / name
                if job_dir.exists():
                    log(f"job {name}: directory exists, skipping")
                    entry['jobs'][name] = dict(skipped=True)
                    continue
                job_dir.mkdir(parents=True)
                cmd = [sys.executable, '-u', '-m', job['module'], *job['args']]
                proc = subprocess.Popen(cmd, cwd=root, env=env,
                                        stdout=(job_dir / 'training.log').open('w', encoding='utf-8'),
                                        stderr=(job_dir / 'errors.log').open('w', encoding='utf-8'),
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                (job_dir / 'process.json').write_text(json.dumps(dict(pid=proc.pid, module=job['module'],
                                                                     arguments=job['args'],
                                                                     launched_at=time.strftime('%Y-%m-%dT%H:%M:%S'))),
                                                      encoding='utf-8')
                running[name] = proc
                entry['jobs'][name] = dict(pid=proc.pid, started_at=time.time())
                log(f"job {name}: started pid {proc.pid}")
            save()
            time.sleep(30)
        entry['finished_at'] = time.time()
        save()
        log(f"group {group['name']} finished")
    state['finished_at'] = time.time()
    save()
    log('queue finished')


if __name__ == '__main__':
    main()
