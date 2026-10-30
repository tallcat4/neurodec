# SPDX-License-Identifier: GPL-3.0-or-later
"""The black box: stands in for the real radio.

Only this package knows about ambe2-c and the secret interleaver. Everything else in
neurodec sees what an SDR capture would give: one bit vector per 20 ms frame.
"""
from .radio import Radio, Receiver

__all__ = ["Radio", "Receiver"]
