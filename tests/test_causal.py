# SPDX-License-Identifier: GPL-3.0-or-later
"""The causal decoder (model.causal) never uses later frames: changing frame t leaves every output sample
before 160 (t - lookahead) - n_fft / 2 unchanged (the centred iSTFT overlap-add is the only look-ahead)."""
import copy

import numpy as np
import pytest
import torch

from neurodec.models.decoder import NeuralDecoder


def small_cfg(causal=True, lookahead=0, excitation=True, gen_lookahead=0):
    m = {"frontend_kind": "bittoken", "backend": "istft", "aux_mels": 16, "mel_bridge": True, "f0_head": True,
         "ap_bands": 5, "upsample_scales": [4, 5, 8], "block_frames": 4, "causal": causal, "lookahead_frames": lookahead,
         "dequant": {"ffn_dims": [32, 24]},
         "bittoken": {"width": 32, "layers": 3, "expansion": 2, "mixer": "mlp", "mlp_hidden": 64},
         "istft": {"width": 24, "layers": 2, "expansion": 2, "n_fft": 512, "hop": 80, "upsample": "learned",
                   "excitation": excitation, "lookahead": gen_lookahead}}
    return {"model": m}


@pytest.mark.parametrize("lookahead,gen_lookahead", [(0, 0), (1, 0), (0, 1)])
@pytest.mark.parametrize("excitation", [False, True])
def test_no_future_dependence(lookahead, gen_lookahead, excitation):
    torch.manual_seed(0)
    G = NeuralDecoder(small_cfg(True, lookahead, excitation, gen_lookahead), 48).eval()
    lookahead += gen_lookahead
    for p in G.parameters():  # random weights everywhere (zero-initialised heads would hide leaks)
        torch.nn.init.normal_(p, std=0.1)
    bits = (torch.rand(1, 60, 48) > 0.5).float()
    t = 40
    flipped = bits.clone()
    flipped[0, t] = 1 - flipped[0, t]
    with torch.no_grad():
        torch.manual_seed(1)
        y0 = G(bits)[0]
        torch.manual_seed(1)
        y1 = G(flipped)[0]
    changed = torch.nonzero((y1 - y0).abs() > 1e-6)
    first = int(changed.min()) if len(changed) else len(y0)
    assert first >= 160 * (t - lookahead) - 256, first
    assert len(changed), "the flipped frame must change the output somewhere"
    if lookahead:  # the look-ahead is really used (not just allowed)
        assert first < 160 * (t - lookahead + 1) - 256, first


def test_non_causal_default_unchanged():
    cfg = small_cfg(False)
    G = NeuralDecoder(cfg, 48)
    assert not G.generator.causal and not G.frontend.causal
    with pytest.raises(ValueError):
        NeuralDecoder(copy.deepcopy(small_cfg(False, lookahead=1)), 48)
    with pytest.raises(ValueError):
        NeuralDecoder(copy.deepcopy(small_cfg(False, gen_lookahead=1)), 48)
