# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-gender (and per-corpus) means of an evaluation's per_utterance.jsonl.

Maps (tx, utt) to the speaker through the test split and session plan, looks the gender up in
the JVS and LibriTTS speaker tables, and prints mean PESQ / STOI / log-mel / F0 RMSE / V-UV per
system for clean and sim at BER 0.

  python -m scripts.gender_summary --config configs/bits49.yaml results/joint_b49_wide_bridge_f0cyc_job [...]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurodec.config import load_config
from neurodec.data.corpus import Utterance
from neurodec.data.session import plan_sessions


def gender_table(cfg):
    """speaker id -> gender from the JVS and LibriTTS speaker tables (corpus roots from the config,
    falling back to the standard locations next to the project)."""
    roots = [Path(c['root']) for c in cfg['corpora'] if c.get('root')]
    roots += [Path('corpora/jvs_ver1'), Path('corpora/LibriTTS_R')]
    g = {}
    for root in roots:
        f = root / 'gender_f0range.txt'
        if f.exists():
            for line in f.read_text(encoding='utf-8').splitlines()[1:]:
                p = line.split(); g['jvs/' + p[0]] = p[1]
        for sp in root.glob('*/SPEAKERS.txt'):
            for line in sp.read_text(encoding='utf-8', errors='ignore').splitlines():
                parts = [x.strip() for x in line.split('|')]
                if len(parts) >= 2 and parts[0].isdigit():
                    g['libritts_r/' + parts[0]] = parts[1]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('results', nargs='+')
    ap.add_argument('--cond', default='clean')
    a = ap.parse_args()
    cfg = load_config(a.config)
    test = [Utterance(**u) for u in json.loads((Path(cfg['data_dir']) / 'split.json').read_text(encoding='utf-8'))['test']]
    speakers = [[u.speaker for u in s] for s in plan_sessions(test, cfg['session']['seconds'])]
    table = gender_table(cfg)
    fields = ['pesq_nb', 'stoi', 'logmel_db', 'f0_rmse_cent', 'vuv_err']
    print(f'| system | group | n | ' + ' | '.join(fields) + ' |')
    print('|---|---|---:|' + '---:|' * len(fields))
    seen = set()
    for r in a.results:
        acc = defaultdict(lambda: defaultdict(list))
        for line in (Path(r) / 'per_utterance.jsonl').read_text(encoding='utf-8').splitlines():
            d = json.loads(line)
            if d['cond'] != a.cond or d['ber'] != 0:
                continue
            spk = speakers[d['tx']][d['utt']]
            grp = spk.split('/')[0] + '_' + table.get(spk, '?')
            for f in fields:
                acc[(d['system'], grp)][f].append(d[f])
        for (sys_, grp), v in sorted(acc.items()):
            if (sys_, grp) in seen:
                continue
            seen.add((sys_, grp))
            n = len(v['pesq_nb'])
            print(f'| {sys_} | {grp} | {n} | ' + ' | '.join(f'{np.mean(v[f]):.3f}' for f in fields) + ' |')


if __name__ == '__main__':
    main()
