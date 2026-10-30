# SPDX-License-Identifier: GPL-3.0-or-later
"""Assemble a full decoder checkpoint from a frontend-only (probe) checkpoint and a bridge model.

The result decodes bits -> wav exactly like the bridge model (frozen clean-speech vocoder
behind the mel bridge) but with the probe's fine-tuned frontend, so it can be evaluated
with neurodec.eval.run_eval and scripts.listen_lang like any other decoder.

  python -m scripts.assemble_decoder --frontend runs/probe_b49_wide_melgan/last.pt \
      --base runs/joint_b49_wide_bridge/step30000.pt --out runs/assembled/b49_wide_melgan.pt
"""
import argparse
from pathlib import Path

import torch

from neurodec.train import load_model, transfer_frontend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frontend', required=True, help='probe checkpoint (frontend.* only)')
    ap.add_argument('--base', required=True, help='full bridge decoder whose backend/bridge are reused')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    G, meta = load_model(a.base, 'cpu')
    ck = torch.load(a.frontend, map_location='cpu', weights_only=False)
    if not all(k.startswith('frontend.') for k in ck['G']):
        raise SystemExit('--frontend must be a frontend-only (probe) checkpoint')
    missing = transfer_frontend(G, ck)
    if missing:
        raise SystemExit(f'frontend transfer left {missing} uninitialised')
    meta = dict(meta, assembled=dict(frontend=str(a.frontend), frontend_step=ck['step'], base=str(a.base),
                                      probe=ck['meta'].get('probe')))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'step': ck['step'], 'G': G.state_dict(), 'meta': meta}, out)
    print(f'wrote {out}: frontend from {a.frontend} (step {ck["step"]}), backend from {a.base}')


if __name__ == '__main__':
    main()
