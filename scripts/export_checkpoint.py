# SPDX-License-Identifier: GPL-3.0-or-later
"""Export inference-only checkpoints (generator + public meta, no critics / optimizers) and record SHA256s.

  python -m scripts.export_checkpoint --out-json checkpoints/final_20261029.json \
      decoder=runs/joint_b49_wide_bridge_f0ap/step30000.pt:checkpoints/final_20261029_ambe49_decoder_step30000.pt
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('items', nargs='+', help='name=source.pt:dest.pt')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    record = {'note': a.note, 'checkpoints': {}}
    for item in a.items:
        name, rest = item.split('=', 1)
        src, dst = rest.rsplit(':', 1)
        ck = torch.load(src, map_location='cpu', weights_only=False)
        slim = {'step': ck['step'], 'G': ck['G'], 'meta': ck['meta']}
        assert 'blackbox' not in slim['meta'].get('cfg', {}), 'black-box section must not be exported'
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        torch.save(slim, dst)
        record['checkpoints'][name] = dict(file=dst, source=src, step=ck['step'], sha256=sha256(Path(dst)),
                                           source_sha256=sha256(Path(src)), bytes=Path(dst).stat().st_size,
                                           frontend=slim['meta'].get('frontend'))
        print(name, dst, ck['step'], record['checkpoints'][name]['bytes'] // 1_000_000, 'MB')
    Path(a.out_json).write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
