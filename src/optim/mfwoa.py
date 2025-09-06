"""Simple MFWOA implementation (pure functions where possible).

This is a compact MFWOA variant supporting multiple tasks (different K). It keeps positions
in continuous [0,255] and enforces sorted unique thresholds.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Sequence, Tuple


def _ensure_sorted_unique(vec: np.ndarray) -> np.ndarray:
    vals = np.clip(vec, 0.0, 255.0)
    vals = np.sort(vals)
    # ensure strictly increasing by small epsilon
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + 1e-3
    return vals


def continuous_to_thresholds(pos: np.ndarray) -> List[int]:
    arr = _ensure_sorted_unique(pos)
    return [int(round(x)) for x in arr]


def mfwoa_optimize(
    hist: np.ndarray,
    K: int,
    pop_size: int = 30,
    iters: int = 100,
    objective: Callable[[np.ndarray], float] = None,
    rng: np.random.Generator = None,
) -> Tuple[List[int], float]:
    """Simplified single-task WOA (not full multifactorial) but structured so it can be extended.

    Args:
        hist: 256-bin histogram
        K: number of thresholds
        objective: function mapping position vector (len K) to score (higher better)
    Returns:
        best thresholds and best score
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if objective is None:
        raise ValueError("objective must be provided")
    dim = K
    # population: positions in [0,255]
    pop = rng.uniform(0.0, 255.0, size=(pop_size, dim))
    # evaluate
    scores = np.array([objective(_ensure_sorted_unique(ind)) for ind in pop])
    best_idx = int(np.argmax(scores))
    best_pos = pop[best_idx].copy()
    best_score = float(scores[best_idx])

    for t in range(iters):
        a = 2.0 * (1 - t / max(1, iters - 1))  # linear decrease
        for i in range(pop_size):
            r1 = rng.random()
            r2 = rng.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()
            if p < 0.5:
                if abs(A) < 1:
                    # encircle prey (best)
                    D = abs(C * best_pos - pop[i])
                    new_pos = best_pos - A * D
                else:
                    # random agent
                    rand_idx = rng.integers(pop_size)
                    X_rand = pop[rand_idx]
                    D = abs(C * X_rand - pop[i])
                    new_pos = X_rand - A * D
            else:
                # spiral update
                b = 1.0
                distance = abs(best_pos - pop[i])
                l = rng.uniform(-1, 1)
                new_pos = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
            # repair and evaluate
            new_pos = _ensure_sorted_unique(new_pos)
            new_score = objective(new_pos)
            if new_score > scores[i]:
                pop[i] = new_pos
                scores[i] = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pos = new_pos.copy()
        # end population
    # return integer thresholds
    return continuous_to_thresholds(best_pos), best_score
