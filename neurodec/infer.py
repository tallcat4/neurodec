# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode captured bits to audio with a trained model.

  python -m neurodec.infer --ckpt runs/b49_12h/last.pt --bits capture.npy --out out.wav
"""
import argparse

import numpy as np
import soundfile as sf
import torch

from .device import pick_device
from .models.decoder import FRAME


@torch.no_grad()
def decode(G, bits: np.ndarray, device, chunk_frames: int = 1500, context: int = 50, group: int = 2) -> np.ndarray:
    """bits [T, S] -> wav [160 T]. Long inputs are decoded in chunks with `context`
    frames of overlap on both sides (kept multiples of `group`)."""
    T = bits.shape[0]
    pad = (-T) % group
    b = np.pad(bits, ((0, pad), (0, 0))).astype(np.float32)
    Tp = b.shape[0]
    chunk_frames -= chunk_frames % group
    context -= context % group
    out = np.zeros(Tp * FRAME, np.float32)
    for s in range(0, Tp, chunk_frames):
        a, e = max(0, s - context), min(Tp, s + chunk_frames + context)
        x = torch.from_numpy(b[a:e])[None].to(device)
        y = G(x)[0].float().cpu().numpy()
        out[s * FRAME : min(Tp, s + chunk_frames) * FRAME] = y[(s - a) * FRAME : (min(Tp, s + chunk_frames) - a) * FRAME]
    return np.clip(out[: T * FRAME], -1, 1)


def main():
    from .train import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bits", required=True, help=".npy uint8 [T, S]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = pick_device()
    G, meta = load_model(a.ckpt, device)
    wav = decode(G, np.load(a.bits), device, group=meta["block_frames"])
    sf.write(a.out, wav, 8000)


if __name__ == "__main__":
    main()
