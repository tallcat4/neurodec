# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import torch

from neurodec.analysis.bit_stats import bit_order
from neurodec.config import load_config
from neurodec.losses import MultiResSTFTLoss, d_loss, g_adv_loss
from neurodec.models.decoder import NeuralDecoder
from neurodec.models.dequant import DequantNet, SoftPermutation
from neurodec.models.pwg import PWGDiscriminator

CFG = load_config("configs/smoke.yaml")


def test_dequant_shapes_paper_and_variants():
    x = torch.randint(0, 2, (2, 50, 49)).float()
    assert DequantNet(49)(x).shape == (2, 56, 50)
    assert DequantNet(49, group_frames=2)(x).shape == (2, 56, 25)
    assert DequantNet(49, attn_axis="full")(x[:, :10]).shape == (2, 56, 10)
    order = np.random.default_rng(0).permutation(98)
    assert DequantNet(49, group_frames=2, bit_order=order, soft_perm=True)(x).shape == (2, 56, 25)


def test_soft_permutation_is_doubly_stochastic():
    P = SoftPermutation(10).matrix()
    assert torch.allclose(P.sum(0), torch.ones(10), atol=1e-3)
    assert torch.allclose(P.sum(1), torch.ones(10), atol=1e-3)


def test_decoder_output_length_bits_and_mel():
    for g, S in [(1, 49), (2, 72)]:
        cfg = {**CFG, "model": {**CFG["model"], "dequant": {**CFG["model"]["dequant"], "group_frames": g}}}
        G = NeuralDecoder(cfg, S)
        assert G(torch.randint(0, 2, (2, 20, S)).float()).shape == (2, 3200)
    M = NeuralDecoder(CFG, 49, frontend="mel")
    assert M(torch.randn(2, 3200) * 0.1).shape == (2, 3200)


def test_backend_shared_between_mel_and_bits():
    a, b = NeuralDecoder(CFG, 49, "mel"), NeuralDecoder(CFG, 49, "bits")
    sa = {k: v for k, v in a.state_dict().items() if not k.startswith("frontend.")}
    sb = {k: v for k, v in b.state_dict().items() if not k.startswith("frontend.")}
    assert sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape for k in sa)


def test_losses_finite():
    y, y_hat = torch.randn(2, 8000) * 0.1, torch.randn(2, 8000) * 0.1
    sc, mag = MultiResSTFTLoss()(y_hat, y)
    D = PWGDiscriminator()
    for kind in ["lsgan", "bce"]:
        assert torch.isfinite(d_loss(D(y), D(y_hat), kind)) and torch.isfinite(g_adv_loss(D(y_hat), kind))
    assert torch.isfinite(sc) and torch.isfinite(mag)


def test_pwg_initialization_keeps_legacy_checkpoints_and_noise_stream():
    torch.manual_seed(17)
    legacy = PWGDiscriminator(layers=4, channels=8)
    expected_noise = torch.randn(2, 1600)
    torch.manual_seed(17)
    revised = PWGDiscriminator(layers=4, channels=8, initialization="pwg")
    assert torch.equal(torch.randn(2, 1600), expected_noise)
    assert legacy.state_dict().keys() == revised.state_dict().keys()
    wave = expected_noise * .05
    assert not torch.equal(legacy(wave), revised(wave))
    revised.load_state_dict(legacy.state_dict(), strict=True)
    assert torch.equal(legacy(wave), revised(wave))


def test_overfit_one_batch():
    torch.manual_seed(0)
    G = NeuralDecoder(CFG, 49)
    bits = torch.randint(0, 2, (2, 10, 49)).float()
    t = torch.linspace(0, 1, 1600)
    wav = torch.stack([0.3 * torch.sin(2 * np.pi * 200 * t), 0.3 * torch.sin(2 * np.pi * 500 * t)])
    stft = MultiResSTFTLoss()
    opt = torch.optim.Adam(G.parameters(), 1e-3)
    z = torch.randn(2, 1, 1600)
    losses = []
    # CPU kernels on Windows converge more slowly than the original Mac run.
    # Keep the same required loss reduction, with a bounded 120-update budget.
    for _ in range(120):
        sc, mag = stft(G(bits, z), wav)
        loss = sc + mag
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if losses[-1] < 0.6 * losses[0]:
            break
    assert losses[-1] < 0.6 * losses[0], (losses[0], losses[-1])


def test_bit_order_regroups_shuffled_fields():
    """Fields of 4 bits (Gray-coded AR(1) values) are shuffled; MI ordering should put
    bits of a field next to each other far more often than chance."""
    r = np.random.default_rng(0)
    T, F, W = 20000, 8, 4
    vals = np.zeros((T, F))
    for t in range(1, T):
        vals[t] = 0.9 * vals[t - 1] + r.standard_normal(F) * 0.45
    q = np.clip(((vals + 2.5) / 5 * 16).astype(int), 0, 15)
    gray = q ^ (q >> 1)
    bits = ((gray[:, :, None] >> np.arange(W)) & 1).reshape(T, F * W).astype(np.uint8)
    perm = r.permutation(F * W)
    shuffled = bits[:, perm]
    field = (perm // W)
    order = bit_order([shuffled], group=1)["order"]
    same = np.mean([field[a] == field[b] for a, b in zip(order[:-1], order[1:])])
    chance = np.mean([field[a] == field[b] for a, b in zip(range(F * W - 1), range(1, F * W))])
    assert same > 0.5 and same > 3 * chance, (same, chance)


def test_phase_loss_zero_for_identical_and_differentiable():
    from neurodec.losses import PhaseLoss
    torch.manual_seed(2)
    y = torch.randn(2, 8000) * 0.1
    l_if, l_gd = PhaseLoss()(y, y)
    assert l_if.item() < 1e-5 and l_gd.item() < 1e-5
    y_hat = (torch.randn(2, 8000) * 0.1).requires_grad_()
    l_if, l_gd = PhaseLoss()(y_hat, y)
    assert 0 < l_if.item() < 3.2 and 0 < l_gd.item() < 3.2
    (l_if + l_gd).backward()
    assert torch.isfinite(y_hat.grad).all() and y_hat.grad.abs().sum() > 0
