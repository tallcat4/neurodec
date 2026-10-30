# SPDX-License-Identifier: GPL-3.0-or-later
"""Full neural decoder: frontend (bits or mel) -> Upsample -> GAU -> PWG generator."""
import math

import numpy as np
import torch
import torch.nn as nn

from .dequant import DequantNet
from .frontends import BitTokenFrontend
from .pwg import GAU, PWGGenerator, Upsample
from .istft import ISTFTGenerator

FRAME = 160
SR = 8000


def mel_filterbank(n_fft: int, n_mels: int, sr: int = SR, fmin: float = 0.0, fmax: float | None = None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l + 1e-9), (r - bins) / (r - c + 1e-9)), 0, None)
    return torch.tensor(fb, dtype=torch.float32)


class LogMel(nn.Module):
    """Frame-synchronous log-mel (hop 160 = one codec frame)."""

    def __init__(self, n_fft=512, win=400, n_mels=80):
        super().__init__()
        self.n_fft, self.win = n_fft, win
        self.register_buffer("fb", mel_filterbank(n_fft, n_mels), persistent=False)
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    def forward(self, wav):  # [B, N] -> [B, n_mels, N / 160]
        spec = torch.stft(wav, self.n_fft, FRAME, self.win, self.window, center=True, return_complex=True)
        mel = torch.matmul(self.fb, spec.abs() ** 2)
        return torch.log(mel + 1e-5)[..., : wav.shape[-1] // FRAME]


class MelFrontend(nn.Module):
    """Upper bound / pre-training condition: clean log-mel instead of bits."""

    def __init__(self, n_mels=80, out_dim=56, n_fft=512, win=400):
        super().__init__()
        self.mel = LogMel(n_fft=n_fft, win=win, n_mels=n_mels)
        self.proj = nn.Conv1d(n_mels, out_dim, 5, padding=2)
        self.out_dim = out_dim

    def forward(self, wav):
        return self.proj(self.mel(wav))


def build_frontend(cfg: dict, n_bits: int, bit_order=None) -> nn.Module:
    """The bits frontend selected by model.frontend_kind (default: the paper's dequantizer)."""
    m = cfg["model"]
    dq = m["dequant"]
    kind = m.get("frontend_kind", "dequant")
    aux = int(m.get("aux_mels", 0) or 0)
    if kind == "dequant":
        return DequantNet(n_bits, group_frames=dq.get("group_frames", 1), kernels=dq["kernels"],
                          channels=dq["channels"], kernel_1d=dq["kernel_1d"], heads=dq["heads"],
                          d_head=dq["d_head"], d_ff=dq["d_ff"], ffn_dims=dq["ffn_dims"],
                          attn_axis=dq["attn_axis"], bit_order=bit_order, soft_perm=dq.get("soft_perm", False),
                          aux_mels=aux)
    if kind == "bittoken":
        if bit_order is not None:
            raise ValueError("The bit-token frontend is position-aware by construction; bit_order is not used")
        if dq.get("group_frames", 1) != 1:
            raise ValueError("The bit-token frontend works per frame; set dequant.group_frames to 1")
        return BitTokenFrontend(n_bits, block_frames=int(m.get("block_frames", 1)), out_dim=dq["ffn_dims"][-1],
                                aux_mels=aux, f0_head=bool(m.get("f0_head", False)),
                                ap_bands=int(m.get("ap_bands", 0)), **m.get("bittoken", {}))
    raise ValueError(f"Unknown frontend_kind: {kind}")


class NeuralDecoder(nn.Module):
    def __init__(self, cfg: dict, n_bits: int, frontend: str = "bits", bit_order=None):
        super().__init__()
        m = cfg["model"]
        dq = m["dequant"]
        g = dq.get("group_frames", 1) if frontend == "bits" and m.get("frontend_kind", "dequant") == "dequant" else 1
        if frontend == "bits":
            self.frontend = build_frontend(cfg, n_bits, bit_order)
            scales = list(m["upsample_scales"]) + ([g] if g > 1 else [])
        else:
            spec = m.get("mel_spec", {})  # e.g. {n_mels: 160, n_fft: 1024, win: 800} for finer low-band resolution
            self.frontend = MelFrontend(n_mels=spec.get("n_mels", 80), out_dim=dq["ffn_dims"][-1],
                                        n_fft=spec.get("n_fft", 512), win=spec.get("win", 400))
            scales = list(m["upsample_scales"])
        assert math.prod(scales) == FRAME * g
        self.kind = frontend
        self.backend_kind = m.get("backend", "pwg")
        # Mel bridge: condition the backend on the frontend's predicted log-mel through the
        # same projection a mel-conditioned backend uses, so a vocoder trained only on clean
        # speech can be kept frozen while the frontend learns (two-stage decoding).
        self.bridge = None
        if m.get("mel_bridge", False):
            if frontend != "bits" or self.backend_kind != "istft" or not int(m.get("aux_mels", 0) or 0):
                raise ValueError("mel_bridge needs the bits frontend, the istft backend and aux_mels > 0")
            self.bridge = nn.Conv1d(int(m["aux_mels"]), self.frontend.out_dim, 5, padding=2)
        if self.backend_kind == "istft":
            self.generator = ISTFTGenerator(self.frontend.out_dim, samples_per_frame=FRAME * g, **m.get("istft", {}))
            return
        if self.backend_kind != "pwg":
            raise ValueError(f"Unknown backend: {self.backend_kind}")
        self.upsample = Upsample(self.frontend.out_dim, scales)
        self.gau = GAU(self.frontend.out_dim, m["aux_channels"], m.get("gau_kernel", 1))
        gen = m["generator"]
        self.generator = PWGGenerator(aux=m["aux_channels"], layers=gen["layers"], stacks=gen["stacks"],
                                      res=gen["residual"], gate=gen["gate"], skip=gen["skip"], kernel=gen["kernel"])

    def back_end(self):
        """Parameters shared between the mel and bits decoders (for pre-training)."""
        if self.backend_kind == "istft":
            return {"generator": self.generator}
        return {"upsample": self.upsample, "gau": self.gau, "generator": self.generator}

    @property
    def uses_excitation(self) -> bool:
        return getattr(self.generator, "excitation", None) is not None

    def forward(self, cond: torch.Tensor, z: torch.Tensor | None = None, return_aux: bool = False,
                f0: torch.Tensor | None = None, voiced: torch.Tensor | None = None,
                aperiodicity: torch.Tensor | None = None):
        """cond: bits [B, T, S] or clean wav [B, N] (mel frontend). Returns wav [B, 160 T]
        (and the frontend's auxiliary log-mel prediction [B, aux_mels, T] when return_aux).
        With harmonic excitation, the mel frontend needs f0/voiced per frame (teacher), while
        the bits frontend derives them from its own F0 head unless given."""
        need_f0 = self.uses_excitation and f0 is None
        if return_aux or self.bridge is not None or (need_f0 and self.kind == "bits"):
            if self.kind != "bits":
                raise ValueError("Auxiliary mel prediction exists only for the bits frontend")
            if need_f0:
                c, aux, f0_pred = self.frontend.forward_with_aux(cond, with_f0=True)
                f0_pred = f0_pred.detach().float()
                f0, voiced = f0_pred[:, 0].exp(), (torch.sigmoid(f0_pred[:, 1]) > 0.5).float()
                if f0_pred.shape[1] > 2:
                    aperiodicity = torch.sigmoid(f0_pred[:, 2:]).transpose(1, 2)  # [B, T, bands]
            else:
                c, aux = self.frontend.forward_with_aux(cond)
            if self.bridge is not None:
                c = self.bridge(aux)
        else:
            c, aux = self.frontend(cond), None
        if self.backend_kind == "istft":
            if z is not None:
                raise ValueError("The iSTFT backend is deterministic and does not take noise")
            y = self.generator(c, f0=f0, voiced=voiced, aperiodicity=aperiodicity) if self.uses_excitation \
                else self.generator(c)
        else:
            c = self.gau(self.upsample(c))
            if z is None:
                z = torch.randn(c.shape[0], 1, c.shape[-1], device=c.device)
            y = self.generator(z, c)
        return (y, aux) if return_aux else y
