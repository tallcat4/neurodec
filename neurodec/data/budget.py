# SPDX-License-Identifier: GPL-3.0-or-later
"""Pick what to "record": a speaker-disjoint test set and a budget-ordered train list.

The train list is ordered so that any prefix is balanced over languages and speakers;
the 1/3/6/12 h subsets are prefixes of the same list (nested), as if recording stopped
earlier.
"""
from collections import defaultdict

import numpy as np

from .corpus import Utterance


def _round_robin(utts: list[Utterance], rng: np.random.Generator) -> list[Utterance]:
    by_spk = defaultdict(list)
    for u in utts:
        by_spk[u.speaker].append(u)
    queues = [list(rng.permutation(v)) for v in by_spk.values()]
    rng.shuffle(queues)
    out = []
    while queues:
        queues = [q for q in queues if q]
        for q in queues:
            out.append(q.pop())
    return out


def split(utts: list[Utterance], cfg: dict, seed: int = 0) -> tuple[list[Utterance], list[Utterance]]:
    """Returns (train, test). Train is ordered and cut at cfg['hours'] in total."""
    rng = np.random.default_rng(seed)
    by_lang = defaultdict(list)
    for u in utts:
        by_lang[u.lang].append(u)

    train_parts, test = {}, []
    for lang, us in sorted(by_lang.items()):
        speakers = sorted({u.speaker for u in us})
        n_test = 0 if lang == "synth" else min(cfg["test_speakers_per_lang"], max(len(speakers) - 1, 0))
        test_spk = set(rng.permutation(speakers)[:n_test]) if n_test else set()
        if lang != "synth" and not test_spk:
            # Single-speaker corpus: hold out utterances instead of speakers.
            order = rng.permutation(len(us))
            cut = max(1, len(us) // 20)
            held = {int(i) for i in order[:cut]}
            test_pool = [u for i, u in enumerate(us) if i in held]
            train_pool = [u for i, u in enumerate(us) if i not in held]
        else:
            test_pool = [u for u in us if u.speaker in test_spk]
            train_pool = [u for u in us if u.speaker not in test_spk]
        limit = cfg["test_minutes_per_lang"] * 60
        acc = 0.0
        for u in _round_robin(test_pool, rng):
            if acc >= limit:
                break
            test.append(u)
            acc += u.seconds
        train_parts[lang] = _round_robin(train_pool, rng)

    # Interleave languages by their share so every prefix keeps the mix.
    share = cfg["lang_share"]
    total = cfg["hours"] * 3600
    used = {k: 0.0 for k in train_parts}
    idx = {k: 0 for k in train_parts}
    train, acc = [], 0.0
    while acc < total:
        avail = [k for k in train_parts if idx[k] < len(train_parts[k]) and share.get(k, 0) > 0]
        if not avail:
            break
        k = min(avail, key=lambda k: used[k] / share[k])
        u = train_parts[k][idx[k]]
        idx[k] += 1
        if acc + u.seconds > total:
            break
        train.append(u)
        used[k] += u.seconds
        acc += u.seconds
    return train, test
