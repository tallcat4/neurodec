# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import io

import pytest
import torch

from neurodec.config import load_config
from neurodec.losses import MultiResSTFTLoss
from neurodec.models.decoder import NeuralDecoder
from neurodec.models.istft import ISTFTGenerator


@pytest.mark.parametrize('frames,group', [(1, 1), (10, 1), (20, 2)])
def test_istft_exact_length_and_frontend_gradients(frames, group):
    cfg = copy.deepcopy(load_config('configs/smoke.yaml'))
    cfg['model'].update(backend='istft', istft=dict(width=24, layers=2, expansion=2))
    cfg['model']['dequant']['group_frames'] = group
    g = NeuralDecoder(cfg, 49)
    y = g(torch.randint(0, 2, (2, frames, 49)).float())
    assert y.shape == (2, frames * 160) and torch.isfinite(y).all()
    y.square().mean().backward()
    grads = [p.grad for p in g.frontend.parameters() if p.requires_grad]
    assert all(x is not None and torch.isfinite(x).all() for x in grads)
    assert sum(x.abs().sum() for x in grads) > 0


def test_istft_reconstruction_loss_backprop_and_checkpoint():
    torch.manual_seed(3)
    g = ISTFTGenerator(8, width=24, layers=2, expansion=2)
    c = torch.randn(2, 8, 10)
    target = torch.randn(2, 1600) * .05
    a, b = MultiResSTFTLoss()(g(c), target)
    (a + b).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())
    state = io.BytesIO()
    torch.save(g.state_dict(), state)
    state.seek(0)
    restored = ISTFTGenerator(8, width=24, layers=2, expansion=2)
    restored.load_state_dict(torch.load(state, weights_only=True))
    assert torch.equal(g(c), restored(c))


def test_learned_condition_upsampler_starts_as_linear_interpolation():
    torch.manual_seed(5)
    g = ISTFTGenerator(8, width=24, layers=2, expansion=2, upsample='learned')
    c = torch.randn(2, 8, 10)
    linear = torch.nn.functional.interpolate(c, size=20, mode='linear', align_corners=False)
    learned = g.upsampler(c)
    assert learned.shape == linear.shape
    assert torch.allclose(learned[..., 2:-2], linear[..., 2:-2], atol=1e-5)
    y = g(c)
    assert y.shape == (2, 1600) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert g.upsampler.weight.grad is not None and g.upsampler.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        ISTFTGenerator(8, width=24, layers=2, expansion=2, upsample='cubic')


def test_harmonic_excitation_features_and_generator():
    import torch
    from neurodec.models.excitation import HarmonicExcitation
    exc = HarmonicExcitation(512, 80, 160)
    f0 = torch.full((2, 10), 120.0); f0[:, :3] = 0
    voiced = (f0 > 0).float()
    feats = exc(f0, voiced, 20)
    assert feats.shape == (2, 3 * 257, 20) and torch.isfinite(feats).all()
    g = ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True)
    c = torch.randn(2, 8, 10)
    y = g(c, f0=f0, voiced=voiced)
    assert y.shape == (2, 1600) and torch.isfinite(y).all()
    with pytest.raises(ValueError):
        g(c)


def test_decoder_bits_path_derives_f0_from_its_head():
    import torch
    from neurodec.models.decoder import NeuralDecoder
    cfg = copy.deepcopy(load_config('configs/smoke.yaml'))
    cfg['model'].update(backend='istft', istft=dict(width=24, layers=2, expansion=2, excitation=True),
                        frontend_kind='bittoken', aux_mels=80, f0_head=True, mel_bridge=True, block_frames=2,
                        bittoken=dict(d_token=8, token_layers=1, token_heads=2, width=16, layers=2, expansion=2, mixer='mlp'))
    G = NeuralDecoder(cfg, 49)
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    y = G(bits)
    assert y.shape == (2, 3200) and torch.isfinite(y).all()
    mcfg = copy.deepcopy(cfg); mcfg['model']['mel_bridge'] = False
    M = NeuralDecoder(mcfg, 49, frontend='mel')
    with pytest.raises(ValueError):
        M(torch.randn(2, 3200) * 0.1)
    f0 = torch.full((2, 20), 150.0)
    assert M(torch.randn(2, 3200) * 0.1, f0=f0, voiced=(f0 > 0).float()).shape == (2, 3200)


def test_cyclic_noise_excitation_is_periodic_and_finite():
    import torch
    from neurodec.models.excitation import HarmonicExcitation
    torch.manual_seed(1)
    exc = HarmonicExcitation(512, 80, 160, mode='cyclic')
    f0 = torch.full((1, 20), 100.0)
    e, _, _ = exc.waveform(f0, torch.ones_like(f0))
    assert e.shape == (1, 3200) and torch.isfinite(e).all()
    # the autocorrelation peaks at the 80-sample pitch period
    x = e[0] - e[0].mean()
    ac = torch.stack([(x[:-lag] * x[lag:]).sum() for lag in range(40, 121)])
    assert 75 <= 40 + int(ac.argmax()) <= 85
    feats = exc(f0, torch.ones_like(f0), 40)
    assert feats.shape == (1, 3 * 257, 40) and torch.isfinite(feats).all()


def test_bandwise_aperiodicity_mixing_and_head():
    import torch
    from neurodec.models.excitation import HarmonicExcitation
    from neurodec.models.frontends import BitTokenFrontend
    from neurodec.losses import periodicity_loss
    exc = HarmonicExcitation(512, 80, 160)
    f0 = torch.full((2, 10), 120.0); v = torch.ones_like(f0)
    ap = torch.zeros(2, 10, 5); ap[1] = 1.0  # item 1: pure noise in every band
    feats = exc(f0, v, 20, aperiodicity=ap)
    assert feats.shape == (2, 3 * 257, 20) and torch.isfinite(feats).all()
    f = BitTokenFrontend(49, block_frames=2, d_token=8, token_layers=1, token_heads=2, width=16, layers=1,
                         expansion=2, out_dim=56, aux_mels=80, mixer='mlp', f0_head=True, ap_bands=5)
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    _, _, head = f.forward_with_aux(bits, with_f0=True)
    assert head.shape == (2, 7, 20)
    teacher = torch.cat([torch.full((2, 20, 1), 110.0), torch.rand(2, 20, 5)], dim=2)
    l_f0, l_rest = periodicity_loss(head, teacher)
    assert torch.isfinite(l_f0) and torch.isfinite(l_rest)
    (l_f0 + l_rest).backward()
    assert f.f0.weight.grad.abs().sum() > 0


def test_wavehax_excitation_reaches_nyquist_and_conv2d_projector():
    import torch
    from neurodec.models.excitation import HarmonicExcitation
    torch.manual_seed(0)
    exc = HarmonicExcitation(512, 80, 160, mode='wavehax')
    f0 = torch.full((1, 40), 70.0)  # low male F0: 57 harmonics below Nyquist
    e, periodic, _ = exc.waveform(f0, torch.ones_like(f0))
    assert e.shape == (1, 6400) and torch.isfinite(e).all()
    spec = torch.stft(periodic[0], 512, 80, window=torch.hann_window(512), return_complex=True).abs().mean(1)
    hz = torch.arange(257) * 8000 / 512
    top = spec[(hz > 3300) & (hz < 3900)].mean(); mid = spec[(hz > 1000) & (hz < 1600)].mean()
    assert top > 0.3 * mid  # the 40-harmonic sum left this band empty below 100 Hz
    assert abs(periodic.pow(2).mean().sqrt().item() - 0.1) < 0.03  # constant power
    feats = exc(f0, torch.ones_like(f0), 80)
    assert feats.shape == (1, 4 * 257, 80) and torch.isfinite(feats).all()
    g = ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True, excitation_mode='wavehax',
                       excitation_proj='conv2d')
    y = g(torch.randn(1, 8, 40), f0=f0, voiced=torch.ones_like(f0))
    assert y.shape == (1, 6400) and torch.isfinite(y).all()
    with pytest.raises(ValueError):
        ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True, excitation_proj='mlp')


def test_source_filter_generator_follows_the_excitation_f0():
    import torch
    from neurodec.models.excitation import HarmonicExcitation
    torch.manual_seed(0)
    g = ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True, excitation_harmonics=66,
                       source_filter=True)
    c = torch.randn(1, 8, 40)
    outs = {}
    for f in (100.0, 160.0):
        f0 = torch.full((1, 40), f)
        y = g(c, f0=f0, voiced=torch.ones_like(f0))
        assert y.shape == (1, 6400) and torch.isfinite(y).all()
        spec = torch.stft(y[0], 512, 80, window=torch.hann_window(512), return_complex=True).abs()
        outs[f] = spec[:, 10:-10].mean(1)
    hz = torch.arange(257) * 8000 / 512
    # energy at the fundamental of each run beats energy at the other run's fundamental
    def at(spec, f): return spec[(hz > f - 12) & (hz < f + 12)].max()
    assert at(outs[100.0], 100) > at(outs[100.0], 160) and at(outs[160.0], 160) > at(outs[160.0], 100)
    with pytest.raises(ValueError):
        ISTFTGenerator(8, width=24, layers=2, expansion=2, source_filter=True)
    with pytest.raises(ValueError):
        ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True, excitation_mode='wavehax',
                       source_filter=True)(c, f0=torch.full((1, 40), 100.0), voiced=torch.ones(1, 40))
    feats, spec = HarmonicExcitation(512, 80, 160)(torch.full((1, 10), 120.0), torch.ones(1, 10), 25, return_spec=True)
    assert feats.shape[-1] == 25 and spec.shape == (1, 257, 25)


def test_soft_source_filter_gate_keeps_excitation_influence():
    import torch
    torch.manual_seed(0)
    g = ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True, excitation_harmonics=66,
                       source_filter=True, gate_floor=0.5)
    c = torch.randn(1, 8, 40)
    ys = {}
    for f in (100.0, 160.0):
        f0 = torch.full((1, 40), f)
        y = g(c, f0=f0, voiced=torch.ones_like(f0))
        assert y.shape == (1, 6400) and torch.isfinite(y).all()
        ys[f] = y
    assert (ys[100.0] - ys[160.0]).abs().mean() > 1e-3  # the F0 still changes the output
    with pytest.raises(ValueError):
        ISTFTGenerator(8, width=24, layers=2, expansion=2, excitation=True, source_filter=True, gate_floor=0.0)
