# SPDX-License-Identifier: GPL-3.0-or-later
"""Opaque encoders (wav2ambe, wav2mrl) called as programs: 8 kHz PCM in, frame bytes out.

A format names a codec and its frame layout. `split` cuts one codec frame into that many
20 ms model frames (e.g. the 40 ms / 160 bit CELP frame becomes 2 x 80 bits), so every
codec is seen as "n_bits per 20 ms" with the public burst size equal to `split`."""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import encoder_bin
from .workspace import audio_workspace

SR = 8000
FRAME = 160
# Frame layout from the air-interface standard: bits are MSB first, the rest of the
# last byte is padding.
FORMATS = {
    "bits49": {"codec": "ambe", "bytes": 7, "bits": 49, "split": 1, "args": ["--no-normalize"]},
    "bits72": {"codec": "ambe", "bytes": 9, "bits": 72, "split": 1, "args": ["--no-normalize", "--fec"]},
    # mrl-celp-c: 3.95 kbps, 20 bytes per 40 ms frame -> two 80-bit model frames of 20 ms
    "mrl80": {"codec": "mrl", "bytes": 20, "bits": 160, "split": 2, "args": []},
}


def n_bits(fmt: str) -> int:
    """Bits per 20 ms model frame."""
    spec = FORMATS[fmt]
    return spec["bits"] // spec["split"]


def to_int16(pcm: np.ndarray) -> np.ndarray:
    if pcm.dtype == np.int16:
        return pcm
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).round().astype(np.int16)


def encode(pcm: np.ndarray, fmt: str = "bits49") -> np.ndarray:
    """PCM (float in [-1, 1] or int16) at 8 kHz -> uint8 bits [n_frames, n_bits]."""
    spec = FORMATS[fmt]
    with audio_workspace() as tmp:
        wav, out = Path(tmp) / "in.wav", Path(tmp) / "out.bin"
        sf.write(wav, to_int16(pcm), SR, subtype="PCM_16")
        # (ambe: --no-normalize, the radio does not see the whole transmission in advance;
        # level changes come from the acoustic simulation instead.)
        subprocess.run([str(encoder_bin(spec["codec"])), str(wav), str(out), *spec["args"]],
                       check=True, capture_output=True)
        raw = np.fromfile(out, dtype=np.uint8)
    frames = raw.reshape(-1, spec["bytes"])
    bits = np.unpackbits(frames, axis=1)[:, : spec["bits"]]
    return bits.reshape(-1, n_bits(fmt))


def bits_to_bytes(bits: np.ndarray, fmt: str = "bits49") -> bytes:
    spec = FORMATS[fmt]
    whole = bits.shape[0] // spec["split"] * spec["split"]
    bits = bits[:whole].reshape(-1, spec["bits"])
    padded = np.zeros((bits.shape[0], spec["bytes"] * 8), dtype=np.uint8)
    padded[:, : spec["bits"]] = bits
    return np.packbits(padded, axis=1).tobytes()
