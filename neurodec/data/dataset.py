# SPDX-License-Identifier: GPL-3.0-or-later
"""Training crops from the simulated campaign.

Each item is `crop_frames` frames of captured bits and the matching clean-source samples,
cut with the campaign's constant alignment offset (alignment.json). Crops start on burst
boundaries, which are known from the frame structure.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .align import FRAME, aligned_target


class Transmissions:
    """All transmissions of one split, in memory (12 h is ~0.7 GB of int16 audio)."""

    def __init__(self, split_dir: str | Path, max_hours: float | None = None, offset: int | None = None):
        self.dir = Path(split_dir)
        rows = [json.loads(l) for l in (self.dir / "index.jsonl").read_text().splitlines()]
        if max_hours is not None:
            rows = [r for r in rows if r["cum_hours"] <= max_hours + 1e-9] or rows[:1]
        self.rows = rows
        self.bits, self.src, self.f0 = [], [], []
        for r in rows:
            z = np.load(self.dir / r["file"])
            self.bits.append(z["bits"])
            self.src.append(z["src"].astype(np.float32) / 32767)
            f0_file = self.dir / "f0" / (Path(r["file"]).stem + ".npy")  # scripts.precompute_f0 (training aid only)
            ap_file = self.dir / "f0" / (Path(r["file"]).stem + "_ap.npy")
            if f0_file.exists():
                f0 = np.load(f0_file).astype(np.float32)[:, None]
                ap = np.load(ap_file).astype(np.float32) if ap_file.exists() else np.ones((len(f0), 5), np.float32)
                self.f0.append(np.concatenate([f0, ap[: len(f0)]], axis=1))  # [T, 1 + bands]
            else:
                self.f0.append(None)
        if offset is None:
            offset = json.loads((self.dir.parent / "alignment.json").read_text())["offset"]
        self.offset = int(offset)

    @property
    def hours(self) -> float:
        return sum(r["seconds"] for r in self.rows) / 3600

    def __len__(self):
        return len(self.rows)

    def target(self, i: int, start: int = 0, n: int | None = None) -> np.ndarray:
        n = self.bits[i].shape[0] - start if n is None else n
        return aligned_target(self.src[i], self.offset, start, n)


def flip_bits(bits: np.ndarray, ber: float, rng: np.random.Generator) -> np.ndarray:
    if ber <= 0:
        return bits
    return bits ^ (rng.random(bits.shape) < ber).astype(bits.dtype)


class CropDataset(Dataset):
    def __init__(self, tx: Transmissions, crop_frames: int, block_frames: int,
                 ber_max: float = 0.0, items_per_epoch: int = 100_000, seed: int = 0,
                 min_rms_db: float = -50.0, max_tries: int = 10, with_f0: bool = False):
        self.tx, self.L, self.B = tx, crop_frames, block_frames
        self.with_f0 = with_f0
        if with_f0 and any(f is None for f in tx.f0):
            raise ValueError("F0 teacher missing: run scripts.precompute_f0 for this campaign")
        self.ber_max, self.n, self.seed = ber_max, items_per_epoch, seed
        frames = np.array([b.shape[0] for b in tx.bits], float)
        self.p = np.clip(frames - crop_frames, 1, None)
        self.p /= self.p.sum()
        self.min_rms = 10 ** (min_rms_db / 20)
        self.max_tries = max_tries
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, self.epoch, k))
        # Transmissions contain pauses; an all-silent target makes the spectral
        # convergence term blow up, so such crops are drawn again.
        for _ in range(self.max_tries):
            i, j, wav = self._draw(rng)
            if np.sqrt(np.mean(wav ** 2)) >= self.min_rms:
                break
        bits = self.tx.bits[i][j : j + self.L]
        if bits.shape[0] < self.L:
            bits = np.pad(bits, ((0, self.L - bits.shape[0]), (0, 0)))
        bits = flip_bits(bits, rng.uniform(0, self.ber_max), rng)
        if self.with_f0:
            f0 = self.tx.f0[i][j : j + self.L]
            if f0.shape[0] < self.L:
                f0 = np.pad(f0, ((0, self.L - f0.shape[0]), (0, 0)), constant_values=((0, 0), (0, 0)))
                f0[self.tx.f0[i][j : j + self.L].shape[0]:, 1:] = 1.0  # padded frames are unvoiced noise
            return torch.from_numpy(bits.astype(np.float32)), torch.from_numpy(wav), torch.from_numpy(f0.copy())
        return torch.from_numpy(bits.astype(np.float32)), torch.from_numpy(wav)

    def _draw(self, rng):
        i = int(rng.choice(len(self.p), p=self.p))
        T = self.tx.bits[i].shape[0]
        n_start = max((T - self.L) // self.B, 0) + 1
        j = int(rng.integers(0, n_start)) * self.B
        return i, j, self.tx.target(i, j, self.L)


class CleanCropDataset(Dataset):
    """Clean speech crops for mel-conditioned pre-training (needs no radio, so no budget)."""

    def __init__(self, utts, crop_samples: int, items_per_epoch: int = 100_000, seed: int = 0,
                 with_f0: bool = False, weights=None):
        from .corpus import load_audio

        self.utts, self.L, self.n, self.seed = utts, crop_samples, items_per_epoch, seed
        self.load = load_audio
        self.with_f0 = with_f0
        self.epoch = 0
        # optional per-utterance sampling probabilities (e.g. gender balancing); None = uniform
        self.p = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != len(utts) or (w < 0).any() or w.sum() <= 0:
                raise ValueError('weights must be non-negative, one per utterance, with a positive sum')
            self.p = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, self.epoch, k))
        idx = int(rng.integers(len(self.utts))) if self.p is None else int(rng.choice(len(self.utts), p=self.p))
        x = self.load(self.utts[idx])
        if len(x) < self.L:
            x = np.pad(x, (0, self.L - len(x)))
        j = int(rng.integers(0, len(x) - self.L + 1))
        x = x[j : j + self.L]
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (rng.uniform(-32, -20) / 20)
        wav = torch.from_numpy(np.clip(x, -1, 1).astype(np.float32))
        if self.with_f0:
            return wav, wav, torch.from_numpy(periodicity_teacher(wav.numpy(), self.L // FRAME))
        return wav, wav  # (condition, target): the mel frontend computes its input from the wav


def periodicity_teacher(wav: np.ndarray, frames: int) -> np.ndarray:
    """[frames, 1 + N_BANDS]: F0 (Hz, 0 = unvoiced) and band aperiodicity of a clean waveform."""
    from .periodicity import band_aperiodicity, frame_f0

    f0 = frame_f0(wav, frames)
    return np.concatenate([f0[:, None], band_aperiodicity(wav, f0)], axis=1).astype(np.float32)


frame_f0 = periodicity_teacher  # backwards-compatible name used by train/eval helpers
