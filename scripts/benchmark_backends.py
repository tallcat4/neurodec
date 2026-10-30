# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure complete G+D updates on one real-speech batch; discard all models.

Resident-batch timings exclude data loading, validation, and checkpoint I/O.
Run alone on the GPU. These are throughput measurements, not quality results.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from neurodec.config import load_config
from neurodec.data.dataset import CropDataset, Transmissions
from neurodec.losses import MultiResSTFTLoss, d_loss, g_adv_loss, feature_matching
from neurodec.models.decoder import LogMel
from neurodec.models.discriminators import build_discriminator
from neurodec.train import build_model


def measure(cfg, bits, wav, front, warmup, steps):
    torch.manual_seed(1)
    g = build_model(cfg, 'bits').cuda()
    if front is not None:
        g.frontend.load_state_dict(front, strict=True)
    aux = cfg['loss'].get('lambda_aux_mel', 0) if getattr(g.frontend, 'aux', None) is not None else 0
    d = build_discriminator(cfg['model']).cuda()
    og = torch.optim.RAdam(g.parameters(), lr=cfg['train']['lr_g'])
    od = torch.optim.RAdam(d.parameters(), lr=cfg['train']['lr_d'])
    stft = MultiResSTFTLoss(**cfg['loss']['stft']).cuda()
    mel = LogMel().cuda()
    lm, lf = cfg['loss'].get('lambda_mel', 0), cfg['loss'].get('lambda_fm', 0)
    freeze = cfg['train'].get('freeze_d_in_g', False)

    def update():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if aux:
                fake, pred = g(bits, return_aux=True)
            else:
                fake = g(bits)
        fake = fake.float()
        sc, mag = stft(fake, wav)
        loss = sc + mag
        if lm:
            loss = loss + lm * F.l1_loss(mel(fake), mel(wav))
        if aux:
            loss = loss + aux * F.l1_loss(pred.float(), mel(wav))
        if freeze:
            d.requires_grad_(False)
        if lf:
            scores, ff = d(fake, return_features=True)
            with torch.no_grad():
                _, rf = d(wav, return_features=True)
            loss = loss + lf * feature_matching(rf, ff)
        else:
            scores = d(fake)
        loss = loss + cfg['loss']['lambda_adv'] * g_adv_loss(scores, cfg['loss']['adv'])
        og.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), cfg['train']['grad_clip_g'])
        og.step()
        if freeze:
            d.requires_grad_(True)
        ld = d_loss(d(wav), d(fake.detach()), cfg['loss']['adv'])
        od.zero_grad(set_to_none=True)
        ld.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), cfg['train']['grad_clip_d'])
        od.step()
        return loss.detach(), ld.detach()

    for _ in range(warmup):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        lg, ld = update()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    if not torch.isfinite(lg + ld):
        raise RuntimeError('Nonfinite benchmark loss')
    peak = torch.cuda.max_memory_allocated()
    g.eval()
    inference = []
    with torch.no_grad():
        for i in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter()
            y = g(bits)
            torch.cuda.synchronize()
            if i >= 2:
                inference.append(time.perf_counter()-start)
    assert y.shape == wav.shape and torch.isfinite(y).all()
    frontend_inference = []
    with torch.no_grad():
        for i in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter()
            g.frontend(bits)
            torch.cuda.synchronize()
            if i >= 2:
                frontend_inference.append(time.perf_counter()-start)
    report = dict(generator_parameters=sum(p.numel() for p in g.parameters()),
                  frontend_parameters=sum(p.numel() for p in g.frontend.parameters()),
                  fp32_frontend_inference_batch_ms=statistics.median(frontend_inference)*1000,
                  discriminator_parameters=sum(p.numel() for p in d.parameters()),
                  median_update_ms=statistics.median(timings)*1000,
                  updates_per_second=1/statistics.mean(timings),
                  peak_allocated_gib=peak/1024**3,
                  fp32_inference_batch_ms=statistics.median(inference)*1000,
                  final_g_loss=lg.item(), final_d_loss=ld.item(), finite=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/backend_benchmark.json')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--configs', nargs='*', default=None, help='configs to measure (default: the three 2026-10-27 backends)')
    ap.add_argument('--no-frontend-weights', action='store_true', help='random frontend (needed for non-dequant frontends)')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    paths = a.configs or ['configs/experiments/ja_gan_pwginit_from46000.yaml', 'configs/experiments/ja_multidisc_cuda.yaml',
                          'configs/experiments/ja_istft_multidisc_cuda.yaml']
    cfg = load_config(paths[0])
    tx = Transmissions(Path(cfg['data_dir']) / 'test_sim')
    ds = CropDataset(tx, cfg['train']['crop_frames'], 2, seed=1234)
    pairs = [ds[i] for i in range(cfg['train']['batch_size'])]
    bits, wav = (torch.stack([x[k] for x in pairs]).cuda() for k in range(2))
    ck = torch.load('runs/ja_check/step46000.pt', map_location='cpu', weights_only=False)
    front = None if a.no_frontend_weights else {k.removeprefix('frontend.'): v for k, v in ck['G'].items() if k.startswith('frontend.')}
    del ck
    report = dict(device=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                  batch=bits.shape[0], frames=bits.shape[1], warmup=a.warmup, measured_steps=a.steps,
                  note='Resident dev-speech batch, full adversarial updates, discarded models; no data/validation/I-O time.',
                  results={})
    for path in paths:
        gc.collect(); torch.cuda.empty_cache()
        result = measure(load_config(path), bits, wav, front, a.warmup, a.steps)
        report['results'][path] = result
        print(path, json.dumps(result), flush=True)
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
