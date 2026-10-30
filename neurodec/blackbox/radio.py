# SPDX-License-Identifier: GPL-3.0-or-later
"""Transmitter (what we can feed and capture) and receiver (evaluation only)."""
import subprocess
from .workspace import audio_workspace
from pathlib import Path

import numpy as np
import soundfile as sf

from . import encoder
from .interleave import Interleaver
from .paths import decoder_bin

# Receiver command lines (evaluation only): ambe2wav takes the rate table, mrl2wav takes none.
_DECODER_ARGS = {"bits49": ["--rate", "34"], "bits72": ["--rate", "33"], "mrl80": []}


class Radio:
    """One PTT transmission: audio at the microphone -> captured frame bits."""

    def __init__(self, fmt: str, secret_seed: int, block_frames: int):
        self.fmt = fmt
        self.n_bits = encoder.n_bits(fmt)
        self.block_frames = block_frames  # burst size is public (frame structure)
        self._il = Interleaver(self.n_bits, secret_seed, block_frames)

    @classmethod
    def from_config(cls, cfg: dict) -> "Radio":
        bb = cfg["blackbox"]
        return cls(cfg["bits"], bb["secret_seed"], bb["block_frames"])

    def transmit(self, pcm_8k: np.ndarray) -> np.ndarray:
        return self._il.apply(encoder.encode(pcm_8k, self.fmt))


class Receiver(Radio):
    """The receiving radio's own decoder. Used only to score the baseline."""

    def receive(self, bits: np.ndarray) -> np.ndarray:
        raw = encoder.bits_to_bytes(self._il.invert(bits), self.fmt)
        with audio_workspace() as tmp:
            src, wav = Path(tmp) / "in.bin", Path(tmp) / "out.wav"
            src.write_bytes(raw)
            subprocess.run([str(decoder_bin(encoder.FORMATS[self.fmt]["codec"])), str(src), str(wav),
                            *_DECODER_ARGS[self.fmt]], check=True, capture_output=True)
            pcm, _ = sf.read(wav, dtype="float32")
        return pcm
