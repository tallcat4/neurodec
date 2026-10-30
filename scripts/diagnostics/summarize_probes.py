# SPDX-License-Identifier: GPL-3.0-or-later
"""Tabulate frontend probes: val mel L1 (dB) per validation split at chosen steps.

  python -m scripts.diagnostics.summarize_probes runs/probe_dequant_20k runs/probe_bittoken_mlp_20k --steps 2000 5000 10000 20000
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--steps', type=int, nargs='*', default=[1000, 2000, 5000, 10000, 20000])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    lines = []
    header = None
    for run in a.runs:
        run = Path(run)
        rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        summary = json.loads((run / 'metrics.json').read_text(encoding='utf-8')) if (run / 'metrics.json').exists() else {}
        by_step = {r['step']: r for r in rows}
        splits = list(rows[-1].get('val', {'val': rows[-1]['val_mel_db']}).keys())
        if header is None:
            header = '| run | params | data | ' + ' | '.join(f'{s} ({sp})' for s in a.steps for sp in splits) + ' | best (split 1) | it/s |'
            lines += [header, '|' + '---|' * (header.count('|') - 1)]
        cells = []
        for s in a.steps:
            r = by_step.get(s)
            for sp in splits:
                cells.append('' if r is None else f"{(r.get('val') or {'val': r['val_mel_db']})[sp]:.2f}")
        best = min(rows, key=lambda r: r['val_mel_db'])
        lines.append(f"| {run.name} | {summary.get('params', 0) / 1e6:.2f}M | {summary.get('data_dir', '')} | "
                     + ' | '.join(cells) + f" | {best['val_mel_db']:.2f} @{best['step']} | {rows[-1]['steps_per_second']:.1f} |")
    table = '\n'.join(lines) + '\n'
    print(table)
    if a.out:
        Path(a.out).write_text(table, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
