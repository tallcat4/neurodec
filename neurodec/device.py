# SPDX-License-Identifier: GPL-3.0-or-later
"""Device selection: CUDA > MPS > CPU, so the same code runs on the M5 and the 4070 Ti Super."""
import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_amp(device: torch.device) -> bool:
    # bf16 autocast only on CUDA; MPS/CPU stay in fp32.
    return device.type == "cuda"
