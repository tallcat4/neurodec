# SPDX-License-Identifier: GPL-3.0-or-later
import torch

from neurodec.losses import d_loss, g_adv_loss, feature_matching
from neurodec.models.discriminators import MultiDiscriminator


def test_multi_critic_generator_gradient_and_feature_target_detach():
    torch.manual_seed(4)
    d = MultiDiscriminator(periods=(2, 5), period_channels=(8, 16), ffts=(256,), spectrum_channels=8)
    d.requires_grad_(False)
    real = (torch.randn(2, 1600) * .05).requires_grad_()
    fake = (torch.randn(2, 1600) * .05).requires_grad_()
    real_scores, real_features = d(real, return_features=True)
    fake_scores, fake_features = d(fake, return_features=True)
    loss = g_adv_loss(fake_scores) + feature_matching(real_features, fake_features)
    loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all() and fake.grad.abs().sum() > 0
    assert real.grad is None
    assert all(p.grad is None for p in d.parameters())
    d.requires_grad_(True)
    d_loss(d(real.detach()), d(fake.detach())).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())


def test_multi_critic_loss_does_not_scale_with_duplicate_critics():
    real, fake = torch.randn(2, 30), torch.randn(2, 30)
    assert torch.allclose(d_loss([real] * 3, [fake] * 3), d_loss(real, fake))
    assert torch.allclose(g_adv_loss([fake] * 3), g_adv_loss(fake))


def test_spectral_critic_silent_short_audio_is_finite():
    d = MultiDiscriminator(periods=(3,), period_channels=(8, 16), ffts=(256,), spectrum_channels=8)
    wave = torch.zeros(2, 160, requires_grad=True)
    loss = g_adv_loss(d(wave))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(wave.grad).all()


def test_mel_critic_shapes_and_gradients():
    from neurodec.models.discriminators import MelCritic
    torch.manual_seed(6)
    c = MelCritic(channels=(8, 16, 16))
    mel = (torch.randn(2, 80, 50) * 2 - 3).requires_grad_()
    score, feats = c(mel, return_features=True)
    assert score.shape[0] == 2 and len(feats) == 3 and torch.isfinite(score).all()
    g_adv_loss(score).backward()
    assert mel.grad is not None and mel.grad.abs().sum() > 0
