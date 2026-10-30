# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark disposable training updates on captured speech; never save model weights."""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from neurodec.config import load_config
from neurodec.data.dataset import CropDataset, Transmissions
from neurodec.losses import MultiResSTFTLoss
from neurodec.models.decoder import NeuralDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/experiments/ja_check.yaml')
    ap.add_argument('--ckpt', default='runs/ja_check/last.pt')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--out', default='results/cuda_benchmark.json')
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA is required'
    torch.set_num_threads(4)
    cfg = load_config(a.config)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    tx = Transmissions(Path(cfg['data_dir']) / 'train')
    # Only all-speech transmissions for this hardware check, never test speakers.
    keep = [i for i, r in enumerate(tx.rows) if 'synth' not in r['langs']]
    assert keep, 'Real-speech transmissions are required'
    tx.rows = [tx.rows[i] for i in keep]
    tx.bits = [tx.bits[i] for i in keep]
    tx.src = [tx.src[i] for i in keep]
    ds = CropDataset(tx, cfg['train']['crop_frames'], cfg['blackbox']['block_frames'])
    batch = [ds[i] for i in range(cfg['train']['batch_size'])]
    bits, wav = [torch.stack([row[i] for row in batch]).cuda() for i in (0, 1)]
    results = []
    for tuned in (False, True):
        torch.manual_seed(1)
        torch.backends.cudnn.benchmark = tuned
        torch.backends.cuda.matmul.allow_tf32 = tuned
        G = NeuralDecoder(cfg, bits.shape[-1]).cuda()
        G.load_state_dict(ck['G'])
        opt = torch.optim.RAdam(G.parameters(), lr=cfg['train']['lr_g'])
        opt.load_state_dict(copy.deepcopy(ck['opt_g']))
        loss = MultiResSTFTLoss(**cfg['loss']['stft']).cuda()
        torch.cuda.reset_peak_memory_stats()

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = G(bits)
            sc, mag = loss(y.float(), wav)
            value = sc + mag
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(G.parameters(), cfg['train']['grad_clip_g'])
            opt.step()
            return value.detach(), norm.detach()

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(a.steps):
            value, norm = step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert torch.isfinite(value) and torch.isfinite(norm), 'Nonfinite loss/gradient'
        row = dict(tuned=tuned, steps=a.steps, seconds=elapsed,
                   steps_per_second=a.steps / elapsed,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                   loss=value.item(), gradient_norm=norm.item())
        results.append(row)
        print(json.dumps(row), flush=True)
        del G, opt, loss
        torch.cuda.empty_cache()
    report = dict(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), checkpoint_step=ck['step'],
                  batch_size=len(batch), results=results,
                  note='Disposable generator-only updates on one real-speech training batch; excludes loading, validation, checkpoint I/O, and adversarial stage.')
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
