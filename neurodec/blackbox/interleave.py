# SPDX-License-Identifier: GPL-3.0-or-later
"""Secret bit interleaver, mimicking the unknown interleaving of a real air interface.

A fixed random permutation is applied to every block of `block_frames` frames
(block_frames * n_bits bits), then the block is cut back into frames. With
block_frames=1 it only shuffles bits inside a frame; with 2 it also mixes bits
between neighbouring frames, like interleaving inside a burst.
"""
import numpy as np


class Interleaver:
    def __init__(self, n_bits: int, seed: int, block_frames: int = 1):
        self.n_bits = n_bits
        self.block_frames = block_frames
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(n_bits * block_frames)
        self.inv = np.argsort(self.perm)

    def _blocks(self, bits: np.ndarray) -> np.ndarray:
        t = bits.shape[0]
        pad = (-t) % self.block_frames
        if pad:
            bits = np.concatenate([bits, np.zeros((pad, self.n_bits), bits.dtype)])
        return bits.reshape(-1, self.block_frames * self.n_bits)

    def apply(self, bits: np.ndarray) -> np.ndarray:
        """[T, n_bits] -> [ceil(T/B)*B, n_bits]."""
        return self._blocks(bits)[:, self.perm].reshape(-1, self.n_bits)

    def invert(self, bits: np.ndarray) -> np.ndarray:
        return self._blocks(bits)[:, self.inv].reshape(-1, self.n_bits)
