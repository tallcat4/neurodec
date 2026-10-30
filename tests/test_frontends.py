# SPDX-License-Identifier: GPL-3.0-or-later
import copy

import pytest
import torch

from neurodec.config import load_config
from neurodec.models.decoder import NeuralDecoder, build_frontend
from neurodec.models.dequant import DequantNet
from neurodec.models.frontends import BitTokenFrontend
from neurodec.train import build_model, transfer_backend, transfer_frontend

CFG = load_config("configs/smoke.yaml")


def small_cfg(frontend_kind="bittoken", backend="istft", aux=80, mixer="transformer"):
    cfg = copy.deepcopy(CFG)
    cfg["model"].update(frontend_kind=frontend_kind, aux_mels=aux, block_frames=2,
                        bittoken=dict(d_token=8, token_layers=1, token_heads=2, width=16, layers=2, expansion=2, mixer=mixer))
    if backend == "istft":
        cfg["model"].update(backend="istft", istft=dict(width=24, layers=2, expansion=2))
    return cfg


@pytest.mark.parametrize("mixer", ["transformer", "mlp"])
def test_bittoken_shapes_and_aux(mixer):
    f = BitTokenFrontend(49, block_frames=2, d_token=8, token_layers=1, token_heads=2, width=16, layers=2,
                         expansion=2, out_dim=56, aux_mels=80, mixer=mixer)
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    cond, aux = f.forward_with_aux(bits)
    assert cond.shape == (2, 56, 20) and aux.shape == (2, 80, 20)
    assert torch.isfinite(cond).all() and torch.isfinite(aux).all()
    assert torch.equal(f(bits), cond)
    with pytest.raises(ValueError):
        BitTokenFrontend(49, aux_mels=0, d_token=8, token_heads=2, width=8).forward_with_aux(bits)


@pytest.mark.parametrize("mixer", ["transformer", "mlp"])
def test_bittoken_does_not_depend_on_bit_adjacency(mixer):
    """Permuting bit positions together with their embeddings leaves the output unchanged,
    i.e. the frontend never assumes that neighbouring bits are related."""
    torch.manual_seed(0)
    f = BitTokenFrontend(12, block_frames=2, d_token=8, token_layers=1, token_heads=2, width=16, layers=1,
                         expansion=2, out_dim=5, mixer=mixer).eval()
    g = copy.deepcopy(f)
    perm = torch.randperm(12)
    with torch.no_grad():
        if mixer == "transformer":
            g.embed.copy_(f.embed[perm])
            g.read.weight.copy_(f.read.weight.view(16, 12, 8)[:, perm].reshape(16, 96))
        else:
            g.read[0].weight.copy_(f.read[0].weight[:, perm])
    bits = torch.randint(0, 2, (3, 10, 12)).float()
    assert torch.allclose(f(bits), g(bits[..., perm]), atol=1e-5)
    assert not torch.allclose(f(bits), f(bits[..., perm]), atol=1e-3)


def test_decoder_with_bittoken_frontend_returns_aux_and_backpropagates():
    cfg = small_cfg()
    G = NeuralDecoder(cfg, 49)
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    y, aux = G(bits, return_aux=True)
    assert y.shape == (2, 3200) and aux.shape == (2, 80, 20)
    (y.square().mean() + aux.abs().mean()).backward()
    grads = [p.grad for p in G.frontend.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    with pytest.raises(ValueError):
        NeuralDecoder(cfg, 49, frontend="mel")(torch.randn(2, 3200), return_aux=True)


def test_dequant_aux_head_keeps_legacy_state_dict():
    legacy_keys = set(DequantNet(49).state_dict())
    with_aux = DequantNet(49, aux_mels=80)
    assert legacy_keys <= set(with_aux.state_dict())
    assert {k for k in with_aux.state_dict() if k not in legacy_keys} == {"aux.weight", "aux.bias"}
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    cond, aux = with_aux.forward_with_aux(bits)
    assert cond.shape == (2, 56, 20) and aux.shape == (2, 80, 20)
    assert torch.allclose(with_aux(bits), cond)


def test_frontend_transfer_from_probe_and_legacy_checkpoints():
    cfg = small_cfg()
    probe = NeuralDecoder(cfg, 49)
    ck = {"step": 7, "G": {k: v for k, v in probe.state_dict().items() if k.startswith("frontend.")},
          "meta": {"frontend": "bits"}}
    target = NeuralDecoder(cfg, 49)
    assert transfer_frontend(target, ck) == []
    bits = torch.randint(0, 2, (1, 10, 49)).float()
    assert torch.equal(target.frontend(bits), probe.frontend(bits))
    # A legacy dequantizer without the head only leaves the new head uninitialised.
    legacy = copy.deepcopy(CFG); legacy["model"].update(backend="istft", istft=dict(width=24, layers=2, expansion=2))
    old = NeuralDecoder(legacy, 49)
    new_cfg = copy.deepcopy(legacy); new_cfg["model"]["aux_mels"] = 80
    new = NeuralDecoder(new_cfg, 49)
    missing = transfer_frontend(new, {"step": 1, "G": old.state_dict(), "meta": {"frontend": "bits"}})
    assert sorted(missing) == ["aux.bias", "aux.weight"]
    with pytest.raises(ValueError):
        transfer_frontend(target, {"step": 1, "G": old.state_dict(), "meta": {"frontend": "bits"}})
    with pytest.raises(ValueError):
        transfer_frontend(target, ck | {"meta": {"frontend": "mel"}})


def test_backend_transfer_from_mel_pretraining():
    cfg = small_cfg()
    mel_run = NeuralDecoder(cfg, 49, frontend="mel")
    bits_run = NeuralDecoder(cfg, 49)
    n = transfer_backend(bits_run, {"G": mel_run.state_dict(), "step": 3, "meta": {"frontend": "mel"}})
    assert n == len([k for k in bits_run.state_dict() if not k.startswith("frontend.")])
    c = torch.randn(1, 56, 10)
    assert torch.equal(bits_run.generator(c), mel_run.generator(c))
    other = copy.deepcopy(cfg); other["model"]["istft"]["width"] = 32
    with pytest.raises(ValueError):
        transfer_backend(NeuralDecoder(other, 49), {"G": mel_run.state_dict(), "step": 3, "meta": {"frontend": "mel"}})


def test_build_model_passes_public_burst_size_only():
    cfg = small_cfg()
    del cfg["model"]["block_frames"]
    G = build_model(cfg, "bits")
    assert G.frontend.block_frames == cfg["blackbox"]["block_frames"]
    with pytest.raises(ValueError):
        build_frontend({**cfg, "model": {**cfg["model"], "frontend_kind": "bittoken"}}, 49, bit_order=list(range(49)))


def test_mel_bridge_uses_predicted_mel_and_loads_mel_projection():
    cfg = small_cfg()
    cfg["model"]["mel_bridge"] = True
    G = NeuralDecoder(cfg, 49)
    mel_run = NeuralDecoder(small_cfg(), 49, frontend="mel")
    n = transfer_backend(G, {"G": mel_run.state_dict(), "step": 1, "meta": {"frontend": "mel"}})
    assert torch.equal(G.bridge.weight, mel_run.frontend.proj.weight)
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    y, aux = G(bits, return_aux=True)
    assert y.shape == (2, 3200) and aux.shape == (2, 80, 20)
    # the backend sees bridge(aux), not the 56-d latent
    expect = mel_run.generator(mel_run.frontend.proj(aux))
    assert torch.allclose(y, expect, atol=1e-5)
    with pytest.raises(ValueError):
        NeuralDecoder({**cfg, "model": {**cfg["model"], "aux_mels": 0}}, 49)


def test_f0_head_and_periodicity_loss():
    from neurodec.losses import periodicity_loss
    f = BitTokenFrontend(49, block_frames=2, d_token=8, token_layers=1, token_heads=2, width=16, layers=1,
                         expansion=2, out_dim=56, aux_mels=80, mixer="mlp", f0_head=True)
    bits = torch.randint(0, 2, (2, 20, 49)).float()
    cond, mel, f0 = f.forward_with_aux(bits, with_f0=True)
    assert f0.shape == (2, 2, 20)
    teacher = torch.tensor([[0.0] * 10 + [120.0] * 10] * 2)
    l_f0, l_vuv = periodicity_loss(f0, teacher)
    assert torch.isfinite(l_f0) and torch.isfinite(l_vuv)
    (l_f0 + l_vuv).backward()
    assert f.f0.weight.grad is not None and f.f0.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        BitTokenFrontend(49, d_token=8, token_heads=2, width=8, aux_mels=80, mixer="mlp").forward_with_aux(bits, with_f0=True)
