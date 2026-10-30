# SPDX-License-Identifier: GPL-3.0-or-later
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repository root

# Opaque codecs: only their command-line tools are used. The directory of each one can be
# overridden with an environment variable (NEURODEC_AMBE_DIR, NEURODEC_MRL_DIR).
CODECS = {
    "ambe": dict(env="NEURODEC_AMBE_DIR", dir="ambe2-c", encoder="encoder/wav2ambe", decoder="decoder/ambe2wav"),
    "mrl": dict(env="NEURODEC_MRL_DIR", dir="mrl-celp-c", encoder="encoder/wav2mrl", decoder="decoder/mrl2wav"),
}


def codec_dir(codec: str = "ambe") -> Path:
    spec = CODECS[codec]
    return Path(os.environ.get(spec["env"], _ROOT / spec["dir"]))


def _exe(path: Path) -> Path:
    return path.with_suffix(".exe") if os.name == "nt" else path


def encoder_bin(codec: str = "ambe") -> Path:
    return _exe(codec_dir(codec) / CODECS[codec]["encoder"])


def decoder_bin(codec: str = "ambe") -> Path:
    return _exe(codec_dir(codec) / CODECS[codec]["decoder"])


def ambe_dir() -> Path:  # backwards compatibility
    return codec_dir("ambe")
