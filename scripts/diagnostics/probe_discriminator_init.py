# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only diagnostic: frozen real-speech pairs, two fresh D initializations.

This does not update the generator or any experiment checkpoint. Outputs are
diagnostic classification scores, not evidence of improved generated speech.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from neurodec.data.dataset import CropDataset, Transmissions
from neurodec.losses import d_loss
from neurodec.models.pwg import PWGDiscriminator
from neurodec.train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--steps', type=int, default=200)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2468)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if ck['sch_d']['last_epoch'] != 0:
        raise ValueError('Use a checkpoint whose discriminator has not trained.')
    cfg = ck['meta']['cfg']
    g = build_model(cfg, ck['meta']['frontend'], ck['meta'].get('bit_order')).eval()
    g.load_state_dict(ck['G'])
    ds = CropDataset(Transmissions(Path(cfg['data_dir']) / 'train'),
                     cfg['train']['crop_frames'], ck['meta']['block_frames'], seed=2468)
    pairs = [ds[i] for i in range(24)]
    bits, real = (torch.stack([p[k] for p in pairs]) for k in range(2))
    with torch.no_grad():
        fake = torch.cat([g(b) for b in bits.split(4)])
    del g
    reports = {}
    for mode in ['checkpoint_default_init', 'pwg_reference_init']:
        d = PWGDiscriminator(**cfg['model']['discriminator'])
        d.load_state_dict(ck['D'])
        if mode == 'pwg_reference_init':
            torch.manual_seed(2468)
            with torch.no_grad():
                for layer in d.modules():
                    if isinstance(layer, torch.nn.Conv1d):
                        v = layer.parametrizations.weight.original1
                        torch.nn.init.kaiming_normal_(v, nonlinearity='relu')
                        layer.parametrizations.weight.original0.copy_(v.norm(dim=(1, 2), keepdim=True))
                        if layer.bias is not None:
                            layer.bias.zero_()
        opt = torch.optim.RAdam(d.parameters(), lr=cfg['train']['lr_d'])

        @torch.no_grad()
        def measure(offset):
            r, f = d(real[offset:offset+8]), d(fake[offset:offset+8])
            return dict(loss=d_loss(r, f).item(), real_mean=r.mean().item(),
                        fake_mean=f.mean().item(), real_std=r.std().item(), fake_std=f.std().item(),
                        patch_accuracy=((r > .5).float().mean() + (f < .5).float().mean()).item() / 2)

        history = [dict(step=0, train=measure(0), unseen_crops=measure(16))]
        start = time.perf_counter()
        for step in range(args.steps):
            begin = (step % 4) * 4
            loss = d_loss(d(real[begin:begin+4]), d(fake[begin:begin+4]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), cfg['train']['grad_clip_d'])
            opt.step()
            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                item = dict(step=step+1, train=measure(0), unseen_crops=measure(16))
                history.append(item)
                print(mode, json.dumps(item), flush=True)
        reports[mode] = dict(elapsed_seconds=time.perf_counter()-start, history=history)
    report = dict(checkpoint=args.checkpoint, generator_step=ck['step'], device='cpu',
                  training_crops=16, unseen_crops=8, train_batch=4, seed=2468,
                  real_rms=real.square().mean().sqrt().item(), fake_rms=fake.square().mean().sqrt().item(),
                  note='Fixed generated speech; unseen crops are from the training split. No G updates.',
                  results=reports)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
