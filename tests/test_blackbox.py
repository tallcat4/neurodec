# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from neurodec.blackbox import Radio, Receiver
from neurodec.blackbox.encoder import encode
from neurodec.blackbox.interleave import Interleaver
from neurodec.data import synth
from conftest import needs_codec, needs_mrl


def test_interleaver_bijective_and_invertible(rng):
    for n, b in [(49, 1), (49, 2), (72, 2)]:
        il = Interleaver(n, seed=5, block_frames=b)
        assert sorted(il.perm) == list(range(n * b))
        bits = rng.integers(0, 2, (101, n)).astype(np.uint8)
        y = il.apply(bits)
        assert y.shape[0] % b == 0 and y.shape[0] >= 101
        assert np.array_equal(il.invert(y)[:101], bits)
        assert not np.array_equal(y[:100], bits[:100])


def test_interleaver_seed_changes_permutation():
    assert not np.array_equal(Interleaver(49, 1, 2).perm, Interleaver(49, 2, 2).perm)


def test_block_interleave_mixes_neighbouring_frames():
    il = Interleaver(49, 3, block_frames=2)
    bits = np.zeros((2, 49), np.uint8)
    bits[0] = 1  # only the first frame of the burst is set
    y = il.apply(bits)
    assert y[0].any() and y[1].any()


@needs_codec
def test_encoder_shapes_and_determinism(rng):
    x = (rng.standard_normal(8000) * 0.1).astype(np.float32)
    a, b = encode(x, "bits49"), encode(x, "bits49")
    assert a.shape[1] == 49 and a.dtype == np.uint8 and np.array_equal(a, b)
    assert 50 <= a.shape[0] <= 56
    f = encode(x, "bits72")
    assert f.shape == (a.shape[0], 72)


@needs_codec
def test_radio_roundtrip_through_receiver(rng):
    _, x = synth.make_probe(np.random.default_rng(3), seconds=(2, 2))
    for fmt in ["bits49", "bits72"]:
        tx = Receiver(fmt, 11, 2)
        bits = tx.transmit(x)
        assert bits.shape[1] == {"bits49": 49, "bits72": 72}[fmt]
        y = tx.receive(bits)
        assert len(y) >= len(x) - 160 and np.isfinite(y).all()
        # The interleaver really hides the codec's layout.
        plain = encode(x, fmt)
        assert not np.array_equal(bits[: len(plain)], plain)


@needs_mrl
def test_second_codec_roundtrip_and_frame_split(rng):
    """The CELP codec's 40 ms / 160 bit frames are exposed as 2 x 80 bits per 20 ms (burst = 2)."""
    _, x = synth.make_probe(np.random.default_rng(4), seconds=(2, 2))
    plain = encode(x, "mrl80")
    assert plain.shape[1] == 80 and plain.shape[0] % 2 == 0 and plain.dtype == np.uint8
    assert abs(plain.shape[0] - len(x) / 160) <= 4
    tx = Receiver("mrl80", 11, 2)
    bits = tx.transmit(x)
    assert bits.shape[1] == 80 and not np.array_equal(bits[: len(plain)], plain)
    y = tx.receive(bits)
    assert len(y) >= len(x) - 320 and np.isfinite(y).all() and np.abs(y).max() > 1e-3
