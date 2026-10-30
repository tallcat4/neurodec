# SPDX-License-Identifier: GPL-3.0-or-later
"""Training (paper Sec. III-B): generator alone first, then generator + discriminator.

  # paper setting, bits from the campaign (budget in hours)
  python -m neurodec.train --config configs/bits49.yaml --hours 12 --run runs/b49_12h
  # mel-conditioned PWG on unlimited clean speech (upper bound / pre-training)
  python -m neurodec.train --config configs/bits49.yaml --frontend mel --run runs/mel
  # bits decoder initialised from the mel run's upsample/GAU/generator
  python -m neurodec.train --config configs/bits49.yaml --hours 12 --init-from runs/mel/last.pt --run runs/b49_12h_pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import load_config
from .data.corpus import Utterance, build_manifest
from .data.dataset import CleanCropDataset, CropDataset, Transmissions
from .device import pick_device, use_amp
from .losses import MultiResSTFTLoss, PhaseLoss, d_loss, g_adv_loss, feature_matching, periodicity_loss
from .models.decoder import FRAME, NeuralDecoder, LogMel
from .models.discriminators import build_discriminator

N_BITS = {"bits49": 49, "bits72": 72, "mrl80": 80}  # bits per 20 ms model frame
MEL_DB = 10.0 / np.log(10.0)  # natural-log power mel -> dB


def build_model(cfg, frontend, bit_order=None):
    # The burst size is public frame structure (like a real radio's), so the model may use it.
    if "blackbox" in cfg and "block_frames" not in cfg["model"]:
        cfg = {**cfg, "model": {**cfg["model"], "block_frames": cfg["blackbox"]["block_frames"]}}
    return NeuralDecoder(cfg, N_BITS[cfg["bits"]], frontend, bit_order)


def save(path, step, G, D, opt_g, opt_d, sch_g, sch_d, meta):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"step": step, "G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(), "sch_g": sch_g.state_dict(), "sch_d": sch_d.state_dict(),
                "meta": meta}, temporary)
    temporary.replace(path)


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    G = build_model(meta["cfg"], meta["frontend"], meta.get("bit_order"))
    G.load_state_dict(ck["G"])
    return G.to(device).eval(), meta


def transfer_backend(G, ck) -> int:
    """Copy every non-frontend tensor (upsample/GAU/generator or iSTFT) from a checkpoint."""
    src, own = ck["G"], G.state_dict()
    shared = {k: v for k, v in src.items() if not k.startswith("frontend.") and k in own and own[k].shape == v.shape}
    if getattr(G, "bridge", None) is not None and "frontend.proj.weight" in src:
        # the mel backend's input projection becomes the bridge from predicted mel
        shared["bridge.weight"], shared["bridge.bias"] = src["frontend.proj.weight"], src["frontend.proj.bias"]
    fresh = ("frontend.", "generator.exc_proj.", "generator.gate.")  # heads a variant may add on top of a shared backend
    backend_keys = [k for k in own if not k.startswith(fresh)]
    missing = [k for k in backend_keys if k not in shared]
    if missing:
        raise ValueError(f'Backend transfer left {len(missing)}/{len(backend_keys)} tensors unmatched '
                         f'(e.g. {missing[:3]}); the backend architecture must be identical')
    G.load_state_dict(shared, strict=False)
    return len(shared)


def transfer_frontend(G, ck, frontend="bits") -> list[str]:
    """Copy the frontend from a full or probe checkpoint; only a new auxiliary head may stay fresh."""
    if ck['meta']['frontend'] != frontend:
        raise ValueError('Frontend transfer requires the same input kind')
    state = {k.removeprefix('frontend.'): v for k, v in ck['G'].items() if k.startswith('frontend.')}
    own = G.frontend.state_dict()
    for k in [k for k in state if (k.startswith('aux.') or k.startswith('f0.')) and (k not in own or own[k].shape != state[k].shape)]:
        del state[k]  # a differently sized auxiliary / periodicity head starts fresh
    missing, unexpected = G.frontend.load_state_dict(state, strict=False)
    if unexpected or any(not (k.startswith('aux.') or k.startswith('f0.')) for k in missing):  # new heads may start fresh
        raise ValueError(f'Frontend transfer mismatch: missing {missing}, unexpected {unexpected}')
    return list(missing)


def mel_utterances(cfg) -> list[Utterance]:
    """Every corpus utterance except test speakers/utterances.

    `mel_pretrain.corpora` (optional) replaces the campaign corpora for the clean-speech vocoder
    (cached as manifest_mel.jsonl), and `mel_pretrain.exclude_test_from` lists further data dirs
    whose test speakers/utterances must also be excluded (so an evaluation set of another
    campaign stays unseen)."""
    data = Path(cfg["data_dir"])
    mp = cfg.get("mel_pretrain", {}) or {}
    if mp.get("corpora"):
        utts = build_manifest(mp["corpora"], data / "manifest_mel.jsonl")
    else:
        utts = build_manifest(cfg["corpora"], data / "manifest.jsonl")
    test_spk, test_paths = set(), set()
    for d in [data, *map(Path, mp.get("exclude_test_from", []))]:
        test = json.loads((d / "split.json").read_text())["test"]
        test_spk |= {u["speaker"] for u in test}
        test_paths |= {Path(u["path"]).as_posix() for u in test}
    keep = [u for u in utts if u.speaker not in test_spk and Path(u.path).as_posix() not in test_paths and u.lang != "synth"]
    return keep or [u for u in utts if Path(u.path).as_posix() not in test_paths]


def gender_balance_weights(cfg, utts: list[Utterance]):
    """Per-utterance sampling weights giving male and female speech equal total probability
    (utterances of unknown gender keep weight 1). None unless `mel_pretrain.balance_gender`."""
    mp = cfg.get("mel_pretrain", {}) or {}
    if not mp.get("balance_gender"):
        return None
    from .data.corpus import speaker_genders
    genders = speaker_genders(mp.get("corpora") or cfg["corpora"])
    g = [genders.get(u.speaker, "?") for u in utts]
    hours = {k: sum(u.seconds for u, gg in zip(utts, g) if gg == k) / 3600 for k in ("M", "F")}
    if min(hours.values()) <= 0:
        raise ValueError(f"Gender balancing needs both genders in the pre-training speech: {hours}")
    total = hours["M"] + hours["F"]
    w = {"M": total / (2 * hours["M"]), "F": total / (2 * hours["F"]), "?": 1.0}
    print(f"mel pre-training speech: M {hours['M']:.1f} h, F {hours['F']:.1f} h -> sampling weights {w}", flush=True)
    return np.array([w[gg] for gg in g])


def train(cfg, run: Path, frontend="bits", hours=None, init_from=None, bit_order=None, resume=True,
          max_seconds=None, init_frontend_from=None, init_disc_from=None):
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    t = cfg["train"]
    if "block_frames" not in cfg["model"]:
        cfg = {**cfg, "model": {**cfg["model"], "block_frames": cfg["blackbox"]["block_frames"]}}
    if (run / "STOP").exists():
        raise ValueError(f"Remove {run / 'STOP'} before resuming a stopped run")
    if (init_frontend_from or init_from or init_disc_from) and resume and (run / 'last.pt').exists():
        raise ValueError('Initialization transfers require a fresh run (or --no-resume)')
    run.mkdir(parents=True, exist_ok=True)
    device = pick_device(t.get("device", "auto"))
    if t.get("cpu_threads"):
        torch.set_num_threads(t["cpu_threads"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = t.get("cudnn_benchmark", False)
        torch.backends.cuda.matmul.allow_tf32 = t.get("tf32", False)
        print(f"device: {torch.cuda.get_device_name(device)}; bf16 autocast; batch {t['batch_size']}", flush=True)
    torch.manual_seed(cfg["seed"])
    L = t["crop_frames"]
    data = Path(cfg["data_dir"])

    if frontend == "bits":
        tx = Transmissions(data / "train", max_hours=hours)
        lam_f0, lam_vuv = cfg['loss'].get('lambda_f0', 0.0), cfg['loss'].get('lambda_vuv', 0.0)
        ds = CropDataset(tx, L, cfg["blackbox"]["block_frames"], t.get("ber_max", 0.0),
                         t["items_per_epoch"], cfg["seed"], with_f0=bool(lam_f0 or lam_vuv))
        print(f"train: {len(tx)} transmissions, {tx.hours:.2f} h, offset {tx.offset}")
        val_tx = Transmissions(data / f"test_{cfg['test_conditions'][-1]}")
    else:
        lam_f0 = lam_vuv = 0.0
        exc = bool(cfg['model'].get('istft', {}).get('excitation', False))
        utts = mel_utterances(cfg)
        ds = CleanCropDataset(utts, L * FRAME, t["items_per_epoch"], cfg["seed"], with_f0=exc,
                              weights=gender_balance_weights(cfg, utts))
        val_tx = Transmissions(data / f"test_{cfg['test_conditions'][0]}")
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=t.get("workers", 4),
                    drop_last=True, persistent_workers=t.get("workers", 4) > 0,
                    pin_memory=device.type == "cuda")

    G = build_model(cfg, frontend, bit_order).to(device)
    D = build_discriminator(cfg['model']).to(device)
    if t.get("freeze_backend", False):
        # two-stage: only the frontend learns; the clean-speech vocoder (and bridge) stay fixed
        for name, p in G.named_parameters():
            p.requires_grad_(name.startswith("frontend."))
        print(f"frozen backend: {sum(p.numel() for p in G.parameters() if not p.requires_grad) / 1e6:.2f}M fixed, "
              f"{sum(p.numel() for p in G.parameters() if p.requires_grad) / 1e6:.2f}M trainable", flush=True)
    opt_g = torch.optim.RAdam([p for p in G.parameters() if p.requires_grad], lr=t["lr_g"])
    opt_d = torch.optim.RAdam(D.parameters(), lr=t["lr_d"])
    sch_g = torch.optim.lr_scheduler.StepLR(opt_g, t["lr_step_g"], 0.5)
    sch_d = torch.optim.lr_scheduler.StepLR(opt_d, t["lr_step_d"], 0.5)
    stft = MultiResSTFTLoss(**cfg["loss"]["stft"]).to(device)
    adv_kind, lam_adv = cfg["loss"]["adv"], cfg["loss"]["lambda_adv"]
    lam_fm, lam_mel = cfg['loss'].get('lambda_fm', 0.0), cfg['loss'].get('lambda_mel', 0.0)
    lam_aux = cfg['loss'].get('lambda_aux_mel', 0.0) if frontend == "bits" else 0.0
    if lam_fm and cfg['model'].get('discriminator_kind', 'pwg') != 'multi':
        raise ValueError('Feature matching requires the multi discriminator')
    if lam_aux and getattr(G.frontend, 'aux', None) is None:
        raise ValueError('lambda_aux_mel needs model.aux_mels > 0')
    n_aux = int(cfg['model'].get('aux_mels', 0) or 0)
    spec = cfg['model'].get('mel_spec', {})
    if n_aux and spec.get('n_mels', 80) != n_aux:
        raise ValueError('model.aux_mels must equal model.mel_spec.n_mels')
    mel = LogMel(n_fft=spec.get('n_fft', 512), win=spec.get('win', 400), n_mels=n_aux or spec.get('n_mels', 80)).to(device)         if (lam_mel or lam_aux) else None
    lam_phase = cfg['loss'].get('lambda_phase', 0.0)
    phase = PhaseLoss(**cfg['loss'].get('phase', {})).to(device) if lam_phase else None
    # The black-box section stays out of checkpoints; only the public burst size is kept.
    meta = {"cfg": {k: v for k, v in cfg.items() if k != "blackbox"}, "frontend": frontend, "hours": hours,
            "block_frames": cfg["blackbox"]["block_frames"],
            "bit_order": None if bit_order is None else [int(i) for i in bit_order]}

    step = 0
    last = run / "last.pt"
    if resume and last.exists():
        # Let the optimizer move parameter buffers to CUDA, while keeping its
        # non-capturable scalar step counters on CPU (avoids per-parameter sync).
        ck = torch.load(last, map_location="cpu", weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        sch_g.load_state_dict(ck["sch_g"]); sch_d.load_state_dict(ck["sch_d"])
        step = ck["step"]
        if 'frontend_initialization' in ck['meta']:
            meta['frontend_initialization'] = ck['meta']['frontend_initialization']
        print(f"resumed at step {step}")
    if step == 0 and init_from:
        ck = torch.load(init_from, map_location='cpu', weights_only=False)
        n = transfer_backend(G, ck)
        meta['backend_initialization'] = dict(checkpoint=str(init_from), step=ck['step'], frontend=ck['meta']['frontend'])
        print(f"initialised {n} backend tensors from {init_from} (step {ck['step']})", flush=True)
    if step == 0 and init_disc_from:
        ck = torch.load(init_disc_from, map_location='cpu', weights_only=False)
        D.load_state_dict(ck['D'], strict=True)  # same critic architecture required
        meta['discriminator_initialization'] = dict(checkpoint=str(init_disc_from), step=ck['step'])
        print(f"initialised critics from {init_disc_from} (step {ck['step']}); fresh D optimizer", flush=True)
    if step == 0 and init_frontend_from:
        ck = torch.load(init_frontend_from, map_location='cpu', weights_only=False)
        missing = transfer_frontend(G, ck, frontend)
        meta['frontend_initialization'] = dict(checkpoint=str(init_frontend_from), step=ck['step'])
        print(f"initialised frontend from step {ck['step']}" + (f" (new head: {missing})" if missing else ""),
              flush=True)

    (run / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    writer = SummaryWriter(run / "tb")
    amp = use_amp(device)
    start_step = step
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    timed_out = False
    stop_requested = False

    def progress(state):
        elapsed = time.perf_counter() - t0
        report = dict(state=state, step=step, start_step=start_step,
                      elapsed_seconds=elapsed, steps_per_second=(step - start_step) / max(elapsed, 1e-9),
                      remaining_seconds=None if max_seconds is None else max(0, max_seconds - elapsed),
                      max_seconds=max_seconds, device=str(device), updated_at=time.time())
        pending = run / "progress.json.tmp"
        pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending.replace(run / "progress.json")

    progress("training")
    epoch = 0
    while step < t["steps"] and not timed_out and not stop_requested:
        ds.epoch = epoch
        for batch in dl:
            if step >= t["steps"] or timed_out or stop_requested:
                break
            cond, wav = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            f0_teacher = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            if frontend == "mel":
                cond = wav
            gen_kwargs = {}
            if frontend == "mel" and f0_teacher is not None:  # excitation pre-training: teacher F0 / voicing / aperiodicity
                f0_hz = f0_teacher[..., 0] if f0_teacher.dim() == 3 else f0_teacher
                gen_kwargs = dict(f0=f0_hz, voiced=(f0_hz > 0).float(),
                                  aperiodicity=f0_teacher[..., 1:] if f0_teacher.dim() == 3 else None)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                if lam_aux:
                    y_hat, aux_pred = G(cond, return_aux=True, **gen_kwargs)
                else:
                    y_hat = G(cond, **gen_kwargs)
            y_hat = y_hat.float()
            sc, mag = stft(y_hat, wav)
            loss_g = sc + mag
            if lam_mel or lam_aux:
                mel_target = mel(wav)
            if lam_mel:
                loss_mel = torch.nn.functional.l1_loss(mel(y_hat), mel_target)
                loss_g = loss_g + lam_mel * loss_mel
            if lam_aux:
                loss_aux = torch.nn.functional.l1_loss(aux_pred.float(), mel_target)
                loss_g = loss_g + lam_aux * loss_aux
            if f0_teacher is not None and frontend == "bits":
                # periodicity heads: a second forward of the (cheap) frontend features
                _, _, f0_pred = G.frontend.forward_with_aux(cond, with_f0=True)
                loss_f0, loss_vuv = periodicity_loss(f0_pred.float(), f0_teacher)
                loss_g = loss_g + lam_f0 * loss_f0 + lam_vuv * loss_vuv
            if lam_phase:
                loss_if, loss_gd = phase(y_hat, wav)
                loss_g = loss_g + lam_phase * (loss_if + loss_gd)
            adv = step >= t["disc_start"]
            if adv:
                if t.get('freeze_d_in_g', False):
                    D.requires_grad_(False)
                if lam_fm:
                    fake_scores, fake_features = D(y_hat, return_features=True)
                    with torch.no_grad():
                        _, real_features = D(wav, return_features=True)
                    loss_fm = feature_matching(real_features, fake_features)
                    loss_g = loss_g + lam_fm * loss_fm
                else:
                    fake_scores = D(y_hat)
                loss_adv = g_adv_loss(fake_scores, adv_kind)
                loss_g = loss_g + lam_adv * loss_adv
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_([p for p in G.parameters() if p.requires_grad], t["grad_clip_g"])
            opt_g.step()
            sch_g.step()
            if adv:
                if t.get('freeze_d_in_g', False):
                    D.requires_grad_(True)
                ld = d_loss(D(wav), D(y_hat.detach()), adv_kind)
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), t["grad_clip_d"])
                opt_d.step()
                sch_d.step()
            step += 1
            timed_out = max_seconds is not None and time.perf_counter() - t0 >= max_seconds
            if step % t['log_interval'] == 0:
                stop_requested = (run / 'STOP').is_file()
            finished = timed_out or step == t["steps"] or stop_requested

            if step % t["log_interval"] == 0:
                writer.add_scalar("train/sc", sc.item(), step)
                writer.add_scalar("train/logmag", mag.item(), step)
                if adv:
                    writer.add_scalar("train/g_adv", loss_adv.item(), step)
                    writer.add_scalar("train/d", ld.item(), step)
                    if lam_fm:
                        writer.add_scalar('train/feature_matching', loss_fm.item(), step)
                if lam_mel:
                    writer.add_scalar('train/mel', loss_mel.item(), step)
                if lam_aux:
                    writer.add_scalar('train/aux_mel_db', loss_aux.item() * MEL_DB, step)
                if f0_teacher is not None and frontend == "bits":
                    writer.add_scalar('train/f0_log', loss_f0.item(), step)
                    writer.add_scalar('train/vuv_bce', loss_vuv.item(), step)
                if lam_phase:
                    writer.add_scalar('train/phase_if', loss_if.item(), step)
                    writer.add_scalar('train/phase_gd', loss_gd.item(), step)
                rate = (step - start_step) / (time.perf_counter() - t0)
                print(f"step {step} sc {sc.item():.3f} logmag {mag.item():.3f}"
                      + (f" adv {loss_adv.item():.3f} d {ld.item():.3f}" if adv else "")
                      + (f" aux {loss_aux.item() * MEL_DB:.2f}dB" if lam_aux else "")
                      + (f" if {loss_if.item():.3f} gd {loss_gd.item():.3f}" if lam_phase else "")
                      + f" ({rate:.1f} it/s; elapsed {(time.perf_counter() - t0) / 60:.1f} min)",
                      flush=True)
                progress("training")
            if step % t["eval_interval"] == 0 or finished:
                progress("validating")
                validate(G, val_tx, frontend, stft, device, writer, step, t, mel if lam_aux else None)
            if step % t["save_interval"] == 0 or finished:
                save(last, step, G, D, opt_g, opt_d, sch_g, sch_d, meta)
                if step % t.get("keep_interval", 10 ** 9) == 0 or finished:
                    save(run / f"step{step}.pt", step, G, D, opt_g, opt_d, sch_g, sch_d, meta)
        epoch += 1
    writer.close()
    progress("stopped" if stop_requested else "finished")
    print(f"training finished at step {step}; elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return run / "last.pt"


@torch.no_grad()
def validate(G, val_tx, frontend, stft, device, writer, step, t, aux_mel=None):
    G.eval()
    rng = np.random.default_rng(0)
    L, scs, mags, auxs = t["crop_frames"] * 4, [], [], []
    for k in range(min(t.get("val_items", 16), 4 * len(val_tx))):
        i = k % len(val_tx)
        T = val_tx.bits[i].shape[0]
        j = int(rng.integers(0, max(T - L, 0) // 2 + 1)) * 2
        bits = val_tx.bits[i][j : j + L]
        if bits.shape[0] < L:
            bits = np.pad(bits, ((0, L - bits.shape[0]), (0, 0)))
        wav = torch.from_numpy(val_tx.target(i, j, L))[None].to(device)
        cond = wav if frontend == "mel" else torch.from_numpy(bits.astype(np.float32))[None].to(device)
        kw = {}
        if frontend == "mel" and G.uses_excitation:
            from .data.dataset import frame_f0
            teacher = torch.from_numpy(frame_f0(wav[0].cpu().numpy(), L))[None].to(device)
            kw = dict(f0=teacher[..., 0], voiced=(teacher[..., 0] > 0).float(), aperiodicity=teacher[..., 1:])
        if aux_mel is not None:
            y, pred = G(cond, return_aux=True, **kw)
            auxs.append(torch.nn.functional.l1_loss(pred.float(), aux_mel(wav)).item() * MEL_DB)
        else:
            y = G(cond, **kw)
        a, b = stft(y, wav)
        scs.append(a.item()); mags.append(b.item())
        if k < 2:
            writer.add_audio(f"val/{k}_decoded", y[0].clamp(-1, 1).cpu(), step, sample_rate=8000)
            if step == t["eval_interval"]:
                writer.add_audio(f"val/{k}_target", wav[0].cpu(), step, sample_rate=8000)
    writer.add_scalar("val/sc", float(np.mean(scs)), step)
    writer.add_scalar("val/logmag", float(np.mean(mags)), step)
    if auxs:
        writer.add_scalar("val/aux_mel_db", float(np.mean(auxs)), step)
    print(f"[val] step {step} sc {np.mean(scs):.3f} logmag {np.mean(mags):.3f}"
          + (f" aux_mel {np.mean(auxs):.2f} dB" if auxs else ""), flush=True)
    G.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--frontend", default="bits", choices=["bits", "mel"])
    ap.add_argument("--hours", type=float, default=None, help="recording budget (prefix of the campaign)")
    ap.add_argument("--init-from", default=None)
    ap.add_argument('--init-frontend-from', default=None, help='transfer only the frontend into a fresh run')
    ap.add_argument('--init-disc-from', default=None, help='transfer the critics (same architecture) into a fresh run')
    ap.add_argument("--bit-order", default=None, help="json list from neurodec.analysis.bit_stats")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None, help="wall-time training budget; save before stopping")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.steps:
        cfg["train"]["steps"] = a.steps
    order = json.loads(Path(a.bit_order).read_text())["order"] if a.bit_order else None
    train(cfg, Path(a.run), a.frontend, a.hours, a.init_from, order, not a.no_resume, a.max_seconds,
          a.init_frontend_from, a.init_disc_from)


if __name__ == "__main__":
    main()
