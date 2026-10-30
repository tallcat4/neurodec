# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulated recording campaign: corpus -> PTT transmissions -> captured bits.

Per transmission this stores what a real campaign would have: the played clean source and
the captured bits. The radio delay is assumed constant, so one offset is estimated from
the training transmissions (align.estimate_offset) and written to alignment.json; the test
sets use the same value. `diag_lead_in` is an optional injected start jitter
(session.max_lead_in, 0 by default) kept only for diagnostics.

  python -m neurodec.data.build --config configs/bits49.yaml
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..blackbox import Radio
from ..config import load_config
from . import budget, corpus, session
from .acoustic_sim import AcousticSim
from .align import estimate_offset

FRAME = 160
SR = 8000


def _job(args):
    cfg, split, cond, idx, utts, out_path = args
    seed = cfg["seed"] * 7919 + {"train": 0, "test": 1}[split] * 104729 + idx
    rng = np.random.default_rng(seed)
    src, spans = session.render(utts, rng, tuple(cfg["session"]["gap"]), tuple(cfg["session"]["level_db"]))
    sim_cfg = dict(cfg["sim"], enabled=cond == "sim")
    mic = AcousticSim(sim_cfg)(src, np.random.default_rng(seed + 1))
    # The moment the radio starts framing relative to playback is not under our control.
    lead = int(rng.integers(0, cfg["session"]["max_lead_in"] + 1))
    mic = np.concatenate([np.zeros(lead, np.float32), mic])
    bits = Radio.from_config(cfg).transmit(mic)
    np.savez_compressed(out_path, bits=bits.astype(np.uint8), src=(src * 32767).astype(np.int16),
                        spans=np.array(spans, np.int64), diag_lead_in=lead)
    return {"file": out_path.name, "seconds": len(src) / SR, "frames": int(bits.shape[0]),
            "langs": sorted({u.lang for u in utts}),
            "speakers": sorted({u.speaker for u in utts})}


def build(cfg: dict, workers: int = 8) -> Path:
    out = Path(cfg["data_dir"])
    out.mkdir(parents=True, exist_ok=True)
    utts = corpus.build_manifest(cfg["corpora"], out / "manifest.jsonl")
    train, test = budget.split(utts, cfg["budget"], cfg["seed"])
    (out / "split.json").write_text(json.dumps(
        {"train": [asdict(u) for u in train], "test": [asdict(u) for u in test]}, ensure_ascii=False))

    plans = [("train", "sim", session.plan_sessions(train, cfg["session"]["seconds"]))]
    test_sessions = session.plan_sessions(test, cfg["session"]["seconds"])
    plans += [("test", c, test_sessions) for c in cfg["test_conditions"]]
    for split, cond, sessions in plans:
        d = out / (split if split == "train" else f"test_{cond}")
        d.mkdir(exist_ok=True)
        jobs = [(cfg, split, cond, i, s, d / f"{i:05d}.npz") for i, s in enumerate(sessions)]
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_job, jobs, chunksize=1))
        acc = 0.0
        for r in rows:  # cumulative hours in recording order -> budget subsets are prefixes
            acc += r["seconds"] / 3600
            r["cum_hours"] = acc
        (d / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{d}: {len(rows)} transmissions, {acc:.2f} h")
        if split == "train":
            align_constant(cfg, d, out)
    return out


def align_constant(cfg: dict, train_dir: Path, out: Path) -> dict:
    """One offset for the whole campaign, from up to align.max_sessions transmissions."""
    rows = [json.loads(l) for l in (train_dir / "index.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(cfg["seed"])
    pick = rng.permutation(len(rows))[: cfg["align"]["max_sessions"]]
    pairs = []
    for i in sorted(pick):
        z = np.load(train_dir / rows[i]["file"])
        pairs.append((z["bits"], z["src"].astype(np.float32) / 32767))
    al = estimate_offset(pairs, cfg["blackbox"]["block_frames"], tuple(cfg["align"]["search_frames"]))
    res = {"offset": al["offset"], "r2": al["r2"], "sessions": len(pairs)}
    (out / "alignment.json").write_text(json.dumps(res))
    print(f"alignment: offset {al['offset']} samples ({al['offset'] / SR * 1000:.1f} ms), R2 {al['r2']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(load_config(a.config), a.workers)


if __name__ == "__main__":
    main()
