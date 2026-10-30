# SPDX-License-Identifier: GPL-3.0-or-later
"""Black-box boundary: only neurodec.blackbox knows the codec and the interleaver.

Training, data loading, models and bit analysis must not touch ambe2wav (the receiving
radio's decoder) or the interleaver secret. data/build.py may use Radio.transmit only;
eval may use the Receiver for the baseline.
"""
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "neurodec"
FORBIDDEN = [r"ambe2wav", r"decoder_bin", r"\bReceiver\b", r"blackbox\.interleave", r"\bInterleaver\b",
             r"secret_seed", r"\._il\b", r"blackbox\.encoder", r"ambe2-c/(encoder|decoder)/(src|include)"]


def _files(exclude=("blackbox", "eval")):
    for f in PKG.rglob("*.py"):
        if f.relative_to(PKG).parts[0] in exclude:
            continue
        yield f


def test_no_codec_internals_outside_blackbox():
    for f in _files():
        text = f.read_text()
        for pat in FORBIDDEN:
            assert not re.search(pat, text), f"{f.relative_to(PKG)} references {pat}"


def test_eval_uses_only_public_receiver_api():
    text = (PKG / "eval" / "run_eval.py").read_text()
    for pat in [r"blackbox\.interleave", r"\bInterleaver\b", r"secret_seed", r"\._il\b", r"decoder_bin"]:
        assert not re.search(pat, text)


def test_blackbox_does_not_read_codec_sources():
    for f in (PKG / "blackbox").rglob("*.py"):
        assert not re.search(r"\.(c|h)\b[\"']", f.read_text()), f


def test_checkpoint_meta_has_no_blackbox_secret():
    text = (PKG / "train.py").read_text()
    assert 'k != "blackbox"' in text
