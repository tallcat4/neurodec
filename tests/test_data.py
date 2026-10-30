# SPDX-License-Identifier: GPL-3.0-or-later
import json

import numpy as np
import pytest

from neurodec.data import budget
from neurodec.data.acoustic_sim import AcousticSim
from neurodec.data.align import aligned_target, estimate_offset
from neurodec.data.corpus import Utterance
from neurodec.data.dataset import CropDataset, flip_bits
from conftest import needs_codec

SIM = dict(drift_ppm=50, rt60=[0.05, 0.5], drr_db=[0, 15], level_db=[-30, -18], snr_db=[20, 40], agc=True, agc_prob=1.0)


def test_acoustic_sim_length_level_and_determinism(rng):
    x = (np.random.default_rng(1).standard_normal(16000) * 0.05).astype(np.float32)
    y1 = AcousticSim(SIM)(x, np.random.default_rng(7))
    y2 = AcousticSim(SIM)(x, np.random.default_rng(7))
    assert y1.shape == x.shape and np.array_equal(y1, y2)
    assert np.abs(y1).max() <= 1.0 and np.isfinite(y1).all()
    assert np.array_equal(AcousticSim(dict(SIM, enabled=False))(x, rng), x)


def _utts():
    out = []
    for lang, n_spk in [("ja", 12), ("en", 12), ("synth", 3)]:
        for s in range(n_spk):
            for k in range(30):
                out.append(Utterance(f"{lang}/{s}/{k}", f"{lang}/{s}", lang, lang, 5.0 + (k % 3)))
    return out


def test_budget_split_speaker_disjoint_and_within_budget():
    cfg = {"hours": 0.5, "lang_share": {"ja": 0.45, "en": 0.45, "synth": 0.1},
           "test_speakers_per_lang": 2, "test_minutes_per_lang": 1}
    train, test = budget.split(_utts(), cfg, seed=0)
    assert sum(u.seconds for u in train) <= 0.5 * 3600
    assert not {u.speaker for u in train} & {u.speaker for u in test}
    assert {u.lang for u in test} == {"ja", "en"}
    # Prefixes keep the language mix (nested budgets).
    prefix = train[: len(train) // 4]
    shares = {l: sum(u.seconds for u in prefix if u.lang == l) for l in ("ja", "en")}
    assert 0.7 < shares["ja"] / shares["en"] < 1.4


def test_aligned_target_convention():
    src = np.arange(1000, dtype=np.float32)
    t = aligned_target(src, offset=37, start_frame=1, n_frames=2)
    assert len(t) == 320 and t[0] == 160 - 37 and t[-1] == 160 + 319 - 37
    t0 = aligned_target(src, offset=37, start_frame=0, n_frames=1)
    assert (t0[:37] == 0).all() and t0[37] == 0.0 and t0[38] == 1.0


def test_estimate_offset_recovers_shift_on_toy_codec(rng):
    """Toy 'codec': 6 bits of quantised log energy, interleaved, framed with a known delay."""
    true = 213
    src = np.concatenate([np.random.default_rng(i).standard_normal(4000) * a
                          for i, a in enumerate(np.random.default_rng(0).uniform(0.001, 0.3, 60))]).astype(np.float32)
    T = len(src) // 160
    tgt = aligned_target(src, true, 0, T).reshape(T, 160)
    q = np.clip(((np.log(np.mean(tgt ** 2, 1) + 1e-6) + 14) * 4).astype(int), 0, 63)
    bits = ((q[:, None] >> np.arange(6)) & 1).astype(np.uint8)
    noise = rng.integers(0, 2, (T, 43)).astype(np.uint8)
    bits = np.concatenate([bits, noise], 1)[:, rng.permutation(49)]
    est = estimate_offset([(bits, src)], block_frames=1, search_frames=(-2, 4))
    assert abs(est["offset"] - true) <= 16, est["offset"]


@needs_codec
def test_estimate_offset_is_consistent_on_real_captures():
    from neurodec.blackbox import Radio
    from neurodec.data import synth

    radio = Radio("bits49", 3, 2)
    pairs = []
    for s in range(6):
        r = np.random.default_rng(s)
        items = [synth.glottal_sweep(int(r.uniform(2, 6) * 8000), r) for _ in range(12)]
        src = np.concatenate([np.concatenate([x / np.abs(x).max() * 10 ** (r.uniform(-30, -6) / 20), np.zeros(3000)])
                              for x in items]).astype(np.float32)
        pairs.append((radio.transmit(src), src))
    a = estimate_offset(pairs[:3], 2)["offset"]
    b = estimate_offset(pairs[3:], 2)["offset"]
    assert abs(a - b) <= 40, (a, b)


def test_flip_bits_rate(rng):
    b = np.zeros((1000, 49), np.uint8)
    assert flip_bits(b, 0.0, rng) is b
    assert abs(flip_bits(b, 0.05, rng).mean() - 0.05) < 0.005


class _FakeTx:
    def __init__(self):
        r = np.random.default_rng(0)
        self.bits = [r.integers(0, 2, (n, 49)).astype(np.uint8) for n in (300, 121)]
        self.src = [r.standard_normal(n * 160).astype(np.float32) for n in (300, 121)]
        self.offset = 50

    def target(self, i, start=0, n=None):
        return aligned_target(self.src[i], self.offset, start, n)


def test_crop_dataset_shapes_and_burst_alignment():
    tx = _FakeTx()
    ds = CropDataset(tx, crop_frames=50, block_frames=2, ber_max=0.0, items_per_epoch=40)
    for k in range(40):
        bits, wav = ds[k]
        assert bits.shape == (50, 49) and wav.shape == (8000,)
        # The crop's start frame j is recoverable from the audio; it must be even.
        i = next(i for i in range(2) if any(np.array_equal(bits.numpy()[0], tx.bits[i][j]) for j in range(0, len(tx.bits[i]), 2)))
        assert i in (0, 1)
