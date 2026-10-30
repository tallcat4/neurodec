# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only CPU probe of a saved discriminator on held-out speech crops."""
import argparse
import json
from pathlib import Path

import torch

from neurodec.data.dataset import CropDataset, Transmissions
from neurodec.losses import MultiResSTFTLoss, d_loss, g_adv_loss, feature_matching
from neurodec.models.decoder import LogMel
from neurodec.models.discriminators import build_discriminator
from neurodec.train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['meta']['cfg']
    g = build_model(cfg, ck['meta']['frontend'], ck['meta'].get('bit_order')).eval()
    g.load_state_dict(ck['G'])
    d = build_discriminator(cfg['model']).eval()
    d.load_state_dict(ck['D'])
    d.requires_grad_(False)
    tx = Transmissions(Path(cfg['data_dir']) / 'test_sim')
    ds = CropDataset(tx, cfg['train']['crop_frames'], ck['meta']['block_frames'], seed=1234)
    pairs = [ds[i] for i in range(4)]
    bits, real = (torch.stack([p[k] for p in pairs]) for k in range(2))
    with torch.no_grad():
        fake = g(bits)
    fake.requires_grad_(True)
    lam_fm = cfg['loss'].get('lambda_fm', 0)
    if lam_fm:
        dr, real_features = d(real, return_features=True)
        df, fake_features = d(fake, return_features=True)
        fm = feature_matching(real_features, fake_features)
    else:
        dr, df = d(real), d(fake)
    adv = g_adv_loss(df, cfg['loss']['adv'])
    sc, mag = MultiResSTFTLoss(**cfg['loss']['stft'])(fake, real)
    ga = torch.autograd.grad(cfg['loss']['lambda_adv'] * adv, fake, retain_graph=bool(lam_fm))[0]
    gf = torch.autograd.grad(lam_fm * fm, fake)[0] if lam_fm else torch.zeros_like(fake)
    gr = torch.autograd.grad(sc + mag, fake)[0]
    def stats(x):
        if isinstance(x, (list, tuple)):
            values = [stats(t) for t in x]
            return dict(mean=sum(t['mean'] for t in values)/len(values), per_critic=values)
        return dict(mean=x.detach().mean().item(), std=x.detach().std().item())
    scores = lambda x: list(x) if isinstance(x, (list, tuple)) else [x]
    report = dict(checkpoint=args.checkpoint, step=ck['step'], device='cpu', crops=4,
                  d_updates=ck['sch_d']['last_epoch'], d_real=stats(dr), d_fake=stats(df),
                  d_silence=stats(d(torch.zeros_like(real))),
                  adv=adv.item(), d_loss=d_loss(dr, df, cfg['loss']['adv']).item(),
                  sc=sc.item(), logmag=mag.item(),
                  weighted_adv_waveform_grad_norm=ga.norm().item(),
                  reconstruction_waveform_grad_norm=gr.norm().item(),
                  gradient_norm_ratio=(ga.norm() / gr.norm().clamp_min(1e-12)).item(),
                  weighted_feature_matching_waveform_grad_norm=gf.norm().item(),
                  reconstruction_reference='MR-STFT only, for comparability with previous diagnostics',
                  finite=all(torch.isfinite(x).all().item() for x in [fake, *scores(dr), *scores(df), ga, gr, gf]))
    if cfg['loss'].get('lambda_mel', 0):
        mel = LogMel()
        gm = torch.autograd.grad(cfg['loss']['lambda_mel'] * torch.nn.functional.l1_loss(mel(fake), mel(real)), fake)[0]
        report['weighted_mel_waveform_grad_norm'] = gm.norm().item()
        report['finite'] = report['finite'] and bool(torch.isfinite(gm).all())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
