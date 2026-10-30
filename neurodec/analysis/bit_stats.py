# SPDX-License-Identifier: GPL-3.0-or-later
"""Bit statistics from captured data only, and a bit ordering that undoes interleaving.

Bits that belong to the same quantised parameter depend on each other, inside a frame and
across neighbouring frames (parameters change slowly). The affinity of two bits is their
normalised mutual information at lag 0 plus half of it at lags +-1 unit; hierarchical
clustering with optimal leaf ordering puts strongly dependent bits next to each other,
which is what the paper's 5-bit-wide CNN kernels expect.

  python -m neurodec.analysis.bit_stats --config configs/bits49.yaml --hours 12 --out runs/bit_order.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def units(bits: np.ndarray, group: int) -> np.ndarray:
    T, S = bits.shape
    U = T // group
    return bits[: U * group].reshape(U, group * S)


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def pairwise_nmi(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """NMI(X_i; Y_j) = I / min(H_i, H_j) for binary columns, [S, S]."""
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    n = len(X)
    px, py = X.mean(0), Y.mean(0)
    p11 = X.T @ Y / n
    p10 = px[:, None] - p11
    p01 = py[None, :] - p11
    p00 = 1 - p11 - p10 - p01
    mi = 0.0
    for pj, a, b in ((p11, px[:, None], py[None]), (p10, px[:, None], 1 - py[None]),
                     (p01, 1 - px[:, None], py[None]), (p00, 1 - px[:, None], 1 - py[None])):
        pj = np.clip(pj, 1e-12, None)
        mi = mi + pj * np.log2(pj / np.clip(a * b, 1e-12, None))
    h = np.minimum(entropy(px)[:, None], entropy(py)[None])
    return np.where(h > 1e-6, np.clip(mi, 0, None) / np.maximum(h, 1e-6), 0.0)


def affinity(seqs: list[np.ndarray], group: int = 1) -> np.ndarray:
    U0, U1 = [], []
    for b in seqs:
        u = units(b, group)
        U0.append(u[:-1]); U1.append(u[1:])
    A, B = np.concatenate(U0), np.concatenate(U1)
    lag0 = pairwise_nmi(np.concatenate([A, B[-1:]]), np.concatenate([A, B[-1:]]))
    lag1 = pairwise_nmi(A, B)
    aff = lag0 + 0.5 * (lag1 + lag1.T)
    np.fill_diagonal(aff, 0.0)
    return aff


def bit_order(seqs: list[np.ndarray], group: int = 1) -> dict:
    aff = affinity(seqs, group)
    S = aff.shape[0]
    dist = 1.0 - aff / (aff.max() + 1e-12)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(optimal_leaf_ordering(Z, squareform(dist, checks=False)))
    ent = entropy(np.concatenate([units(b, group) for b in seqs]).mean(0))
    return {"order": [int(i) for i in order], "group": group, "entropy": ent.tolist(), "n_bits": S,
            "constant_bits": [int(i) for i in np.where(ent < 1e-3)[0]]}


def main():
    from ..config import load_config
    from ..data.dataset import Transmissions

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    tx = Transmissions(Path(cfg["data_dir"]) / "train", max_hours=a.hours)
    g = cfg["model"]["dequant"].get("group_frames", 1)
    res = bit_order(tx.bits, g)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    print(f"{res['n_bits']} bits per unit, {len(res['constant_bits'])} constant; order written to {a.out}")


if __name__ == "__main__":
    main()
