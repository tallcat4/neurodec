# SPDX-License-Identifier: GPL-3.0-or-later
"""Pack utterances into PTT transmissions, the way they would be played to the radio."""
import numpy as np

from .corpus import SR, Utterance, load_audio


def plan_sessions(utts: list[Utterance], seconds: float) -> list[list[Utterance]]:
    """Greedy packing in list order (keeps the budget order of the train list)."""
    sessions, cur, acc = [], [], 0.0
    for u in utts:
        if cur and acc + u.seconds > seconds:
            sessions.append(cur)
            cur, acc = [], 0.0
        cur.append(u)
        acc += u.seconds
    if cur:
        sessions.append(cur)
    return sessions


def render(utts: list[Utterance], rng: np.random.Generator, gap=(0.3, 1.0), level_db=(-32, -20)):
    """Clean source for one transmission and the [start, end) sample range of each utterance."""
    parts, spans, pos = [], [], 0
    lead = np.zeros(int(rng.uniform(*gap) * SR), np.float32)
    parts.append(lead)
    pos += len(lead)
    for u in utts:
        x = load_audio(u)
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        if u.lang != "synth":  # probes carry their own level
            x = x / rms * 10 ** (rng.uniform(*level_db) / 20)
        x = np.clip(x, -1, 1).astype(np.float32)
        parts.append(x)
        spans.append((pos, pos + len(x)))
        pos += len(x)
        g = np.zeros(int(rng.uniform(*gap) * SR), np.float32)
        parts.append(g)
        pos += len(g)
    return np.concatenate(parts), spans
