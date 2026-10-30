# SPDX-License-Identifier: GPL-3.0-or-later
"""Frame/sample alignment between captured bits and the played audio.

The radio's delay is taken to be constant (fixed codec delay; playback and capture are
started together), so a single offset is estimated once from all training transmissions.

For every candidate sample offset d, frame j of a capture is paired with source samples
[160 j - d, 160 j - d + 160). A ridge regression from the bits of one burst (block_frames
frames, the public burst size; random tanh features for some non-linearity) to log band
energies is fitted on half of the data and scored on the other half. The best held-out
R^2 wins. Nothing about the codec is needed beyond "the level is in the bits somewhere".

Convention: target[n] = src[n - offset], so frame j covers target[160 j : 160 j + 160].
"""
import numpy as np
from scipy import signal

FRAME = 160
SR = 8000


def _band_cumsums(src: np.ndarray, pad: int, n_bands: int) -> list[np.ndarray]:
    edges = np.linspace(0, SR / 2, n_bands + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if a == 0:
            sos = signal.butter(4, b, "lowpass", fs=SR, output="sos")
        elif b >= SR / 2:
            sos = signal.butter(4, a, "highpass", fs=SR, output="sos")
        else:
            sos = signal.butter(4, [a, b], "bandpass", fs=SR, output="sos")
        x = np.pad(signal.sosfilt(sos, src.astype(np.float64)), (pad, pad))
        out.append(np.concatenate([[0.0], np.cumsum(x ** 2)]))
    return out


def _features(bits: np.ndarray, block_frames: int, R: np.ndarray | None) -> np.ndarray:
    T, S = bits.shape
    U = T // block_frames
    X = bits[: U * block_frames].reshape(U, block_frames * S).astype(np.float64) * 2 - 1
    if R is not None:
        X = np.concatenate([X, np.tanh(1.5 * X @ R)], axis=1)
    return np.concatenate([X, np.ones((U, 1))], axis=1)


def estimate_offset(pairs: list[tuple[np.ndarray, np.ndarray]], block_frames: int = 1,
                    search_frames=(-2, 12), n_bands: int = 4, n_rf: int = 512,
                    lam: float = 3.0, chunk: int = 250, seed: int = 0) -> dict:
    """pairs: [(bits [T, S] 0/1, src [N] float), ...] sharing one offset.
    Returns {'offset', 'r2', 'candidates', 'curve'}."""
    S = pairs[0][0].shape[1] * block_frames
    R = np.random.default_rng(seed).standard_normal((S, n_rf)) / np.sqrt(S) if n_rf else None
    pad = FRAME * (max(abs(search_frames[0]), abs(search_frames[1])) + 2)
    cand = np.arange(search_frames[0] * FRAME, search_frames[1] * FRAME + 1)

    Xs, folds, cums, n_units = [], [], [], []
    for bits, src in pairs:
        X = _features(bits, block_frames, R)
        Xs.append(X)
        folds.append((np.arange(X.shape[0]) * block_frames // chunk) % 2)  # alternate chunks
        cums.append(_band_cumsums(src, pad, n_bands))
        n_units.append(X.shape[0])
    X = np.concatenate(Xs)
    fold = np.concatenate(folds)
    tr, va = fold == 0, fold == 1
    if va.sum() < 10:
        tr = va = np.ones(len(X), bool)
    A = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T)

    scores = np.empty(len(cand))
    for i in range(0, len(cand), 32):
        d = cand[i : i + 32]
        Ys = []
        for c, U in zip(cums, n_units):
            starts = FRAME * np.arange(U * block_frames)[None, :] - d[:, None] + pad
            Y = np.stack([np.log(b[starts + FRAME] - b[starts] + 1e-6) for b in c], axis=-1)
            Ys.append(Y.reshape(len(d), U, -1))  # [D, U, block_frames * n_bands]
        Y = np.concatenate(Ys, axis=1)
        W = np.einsum("fu,duk->dfk", A, Y[:, tr])
        pred = np.einsum("uf,dfk->duk", X[va], W)
        yv = Y[:, va]
        ss_res = ((yv - pred) ** 2).sum(axis=(1, 2))
        ss_tot = ((yv - yv.mean(axis=1, keepdims=True)) ** 2).sum(axis=(1, 2)) + 1e-9
        scores[i : i + 32] = 1 - ss_res / ss_tot
    best = int(np.argmax(scores))
    return {"offset": int(cand[best]), "r2": float(scores[best]), "candidates": cand, "curve": scores}


def aligned_target(src: np.ndarray, offset: int, start_frame: int, n_frames: int) -> np.ndarray:
    """target[160 start_frame : 160 (start_frame + n_frames)], with target[n] = src[n - offset]."""
    a = FRAME * start_frame - offset
    b = a + FRAME * n_frames
    out = np.zeros(b - a, np.float32)
    lo, hi = max(a, 0), min(b, len(src))
    if hi > lo:
        out[lo - a : hi - a] = src[lo:hi]
    return out
