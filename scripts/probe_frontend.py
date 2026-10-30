# SPDX-License-Identifier: GPL-3.0-or-later
"""Frontend-only probe: captured bits -> clean-source log-mel regression.

Trains just the bits frontend plus its auxiliary mel head with an L1 loss against the
80-band log-mel of the clean source (hop = one codec frame). This measures how well a
frontend recovers the acoustic envelope from the bits, independent of the vocoder, its
noise and the GAN. The saved checkpoint only holds the frontend and can initialise a full
decoder with `neurodec.train --init-frontend-from`.

  python -m scripts.probe_frontend --config configs/experiments/ja_bittoken_multidisc_cuda.yaml --run runs/probe_bittoken --steps 5000
  python -m scripts.probe_frontend --config configs/experiments/ja_dequant_aux_multidisc_cuda.yaml --run runs/probe_dequant --steps 5000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurodec.config import load_config
from neurodec.data.dataset import CropDataset, Transmissions
from neurodec.device import pick_device, use_amp
from neurodec.models.decoder import FRAME, LogMel, build_frontend
from neurodec.train import MEL_DB, N_BITS, transfer_frontend
from neurodec.models.discriminators import MelCritic
from neurodec.losses import d_loss, g_adv_loss, feature_matching, periodicity_loss


def fixed_val_crops(val_tx, crop_frames, items):
    """Deterministic validation crops (same rule as neurodec.train.validate)."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(min(items, 4 * len(val_tx))):
        i = k % len(val_tx)
        T = val_tx.bits[i].shape[0]
        j = int(rng.integers(0, max(T - crop_frames, 0) // 2 + 1)) * 2
        bits = val_tx.bits[i][j : j + crop_frames]
        if bits.shape[0] < crop_frames:
            bits = np.pad(bits, ((0, crop_frames - bits.shape[0]), (0, 0)))
        out.append((torch.from_numpy(bits.astype(np.float32))[None], torch.from_numpy(val_tx.target(i, j, crop_frames))[None]))
    return out


@torch.no_grad()
def validate(frontend, mel, crops, device):
    frontend.eval()
    errors = []
    for bits, wav in crops:
        _, pred = frontend.forward_with_aux(bits.to(device))
        errors.append(torch.nn.functional.l1_loss(pred.float(), mel(wav.to(device))).item() * MEL_DB)
    frontend.train()
    return float(np.mean(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--frontend-kind', default=None, choices=[None, 'dequant', 'bittoken'])
    ap.add_argument('--mixer', default=None, choices=[None, 'transformer', 'mlp'])
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--hours', type=float, default=None)
    ap.add_argument('--lr', type=float, default=None, help='default: train.lr_g')
    ap.add_argument('--val-interval', type=int, default=250)
    ap.add_argument('--val-items', type=int, default=16)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--data-dir', default=None, help='override data_dir (training campaign)')
    ap.add_argument('--init-frontend-from', default=None, help='start from a full or probe checkpoint')
    ap.add_argument('--mel-adv', type=float, default=0.0, help='weight of the log-mel critic (LSGAN)')
    ap.add_argument('--mel-fm', type=float, default=0.0, help='weight of critic feature matching')
    ap.add_argument('--delta', type=float, default=0.0, help='weight of the L1 on frame-to-frame mel differences')
    ap.add_argument('--disc-start', type=int, default=0)
    ap.add_argument('--f0', type=float, default=0.0, help='weight of the log-F0 L1 (needs model.f0_head and precomputed F0)')
    ap.add_argument('--vuv', type=float, default=0.0, help='weight of the voicing BCE')
    ap.add_argument('--lr-d', type=float, default=1e-4)
    ap.add_argument('--val-dirs', nargs='*', default=None,
                    help='validation splits (default: <data_dir>/test_<last condition>); each is reported separately')
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.data_dir:
        cfg['data_dir'] = a.data_dir
    m, t = cfg['model'], cfg['train']
    if a.frontend_kind:
        m['frontend_kind'] = a.frontend_kind
    if a.mixer:
        m.setdefault('bittoken', {})['mixer'] = a.mixer
    m['aux_mels'] = int(m.get('aux_mels', 0) or 80)
    m['block_frames'] = cfg['blackbox']['block_frames']
    run = Path(a.run)
    if (run / 'last.pt').exists():
        raise FileExistsError(f'Refusing to overwrite existing probe: {run}')
    run.mkdir(parents=True, exist_ok=True)
    device = pick_device(t.get('device', 'auto'))
    if t.get('cpu_threads'):
        torch.set_num_threads(t['cpu_threads'])
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = t.get('cudnn_benchmark', False)
        torch.backends.cuda.matmul.allow_tf32 = t.get('tf32', False)
    seed = cfg['seed'] if a.seed is None else a.seed
    torch.manual_seed(seed)
    L = t['crop_frames']
    data = Path(cfg['data_dir'])
    tx = Transmissions(data / 'train', max_hours=a.hours)
    use_f0 = bool(a.f0 or a.vuv)
    ds = CropDataset(tx, L, cfg['blackbox']['block_frames'], t.get('ber_max', 0.0), t['items_per_epoch'], seed,
                     with_f0=use_f0)
    workers = t.get('workers', 4) if a.workers is None else a.workers
    dl = DataLoader(ds, batch_size=t['batch_size'], shuffle=False, num_workers=workers, drop_last=True,
                    persistent_workers=workers > 0, pin_memory=device.type == 'cuda')
    val_dirs = a.val_dirs or [str(data / f"test_{cfg['test_conditions'][-1]}")]
    crop_sets = {Path(v).parent.name + '/' + Path(v).name: fixed_val_crops(Transmissions(v), 4 * L, a.val_items)
                 for v in val_dirs}

    frontend = build_frontend(cfg, N_BITS[cfg['bits']]).to(device)
    if a.init_frontend_from:
        class _Holder: pass
        holder = _Holder(); holder.frontend = frontend
        ck0 = torch.load(a.init_frontend_from, map_location='cpu', weights_only=False)
        missing = transfer_frontend(holder, ck0)
        print(f'initialised frontend from {a.init_frontend_from} (step {ck0["step"]}, new: {missing})', flush=True)
    critic = MelCritic().to(device) if (a.mel_adv or a.mel_fm) else None
    opt_d = torch.optim.RAdam(critic.parameters(), lr=a.lr_d) if critic is not None else None
    spec = m.get('mel_spec', {})
    mel = LogMel(n_fft=spec.get('n_fft', 512), win=spec.get('win', 400), n_mels=m['aux_mels']).to(device)
    n_params = sum(p.numel() for p in frontend.parameters())
    lr = t['lr_g'] if a.lr is None else a.lr
    opt = torch.optim.RAdam(frontend.parameters(), lr=lr)
    amp = use_amp(device)
    print(f"probe {m.get('frontend_kind', 'dequant')}/{m.get('bittoken', {}).get('mixer', '-')}: "
          f"{n_params / 1e6:.2f}M params, {tx.hours:.2f} h, device {device}, lr {lr}", flush=True)

    history, running, step, epoch = [], [], 0, 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    while step < a.steps:
        ds.epoch = epoch
        for batch in dl:
            if step >= a.steps:
                break
            bits, wav = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                if use_f0:
                    _, pred, f0_pred = frontend.forward_with_aux(bits, with_f0=True)
                else:
                    _, pred = frontend.forward_with_aux(bits)
            pred, target = pred.float(), mel(wav)
            loss = torch.nn.functional.l1_loss(pred, target)
            l1 = loss.detach()
            if use_f0:
                l_f0, l_vuv = periodicity_loss(f0_pred.float(), batch[2].to(device))
                loss = loss + a.f0 * l_f0 + a.vuv * l_vuv
            if a.delta:
                loss = loss + a.delta * torch.nn.functional.l1_loss(pred[..., 1:] - pred[..., :-1],
                                                                   target[..., 1:] - target[..., :-1])
            use_critic = critic is not None and step >= a.disc_start
            if use_critic:
                critic.requires_grad_(False)
                fake_scores, fake_features = critic(pred, return_features=True)
                if a.mel_fm:
                    with torch.no_grad():
                        _, real_features = critic(target, return_features=True)
                    loss = loss + a.mel_fm * feature_matching([real_features], [fake_features])
                if a.mel_adv:
                    loss = loss + a.mel_adv * g_adv_loss(fake_scores)
                critic.requires_grad_(True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(frontend.parameters(), t['grad_clip_g'])
            opt.step()
            if use_critic:
                ld = d_loss(critic(target), critic(pred.detach()))
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_d.step()
            step += 1
            running.append(l1.item() * MEL_DB)
            if step % a.val_interval == 0 or step == a.steps:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                vals = {name: validate(frontend, mel, crops, device) for name, crops in crop_sets.items()}
                row = dict(step=step, train_mel_db=float(np.mean(running)), val_mel_db=next(iter(vals.values())),
                           val=vals, elapsed_seconds=elapsed, steps_per_second=step / elapsed)
                history.append(row)
                running = []
                print(f"step {step} train {row['train_mel_db']:.2f} dB val "
                      + ' '.join(f"{k} {v:.2f} dB" for k, v in vals.items())
                      + f" ({row['steps_per_second']:.1f} it/s)", flush=True)
                (run / 'metrics.jsonl').open('a', encoding='utf-8').write(json.dumps(row) + '\n')
        epoch += 1

    meta = {'cfg': {k: v for k, v in cfg.items() if k != 'blackbox'}, 'frontend': 'bits', 'hours': a.hours,
            'block_frames': cfg['blackbox']['block_frames'], 'bit_order': None,
            'probe': dict(steps=a.steps, lr=lr, seed=seed, params=n_params, mel_adv=a.mel_adv, mel_fm=a.mel_fm,
                          delta=a.delta, f0=a.f0, vuv=a.vuv, init_frontend_from=a.init_frontend_from)}
    torch.save({'step': step, 'G': {f'frontend.{k}': v.detach().cpu() for k, v in frontend.state_dict().items()},
                'meta': meta}, run / 'last.pt')
    best = min(history, key=lambda r: r['val_mel_db'])
    summary = dict(frontend_kind=m.get('frontend_kind', 'dequant'), mixer=m.get('bittoken', {}).get('mixer'),
                   params=n_params, steps=step, hours=tx.hours, data_dir=str(data), val_dirs=val_dirs, device=str(device),
                   steps_per_second=history[-1]['steps_per_second'], final=history[-1], best=best,
                   peak_allocated_gib=(torch.cuda.max_memory_allocated() / 2 ** 30) if device.type == 'cuda' else None)
    (run / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
