"""Simplified Multifactorial WOA (MFWOA) for multilevel thresholding.

This implementation is a pragmatic, testable variant:
- Unified population encoding length = max(Ks)
- Each individual has a skill factor (assigned task)
- Adaptive rmp scalar (global) updated by simple success metric
- Knowledge transfer via mixing with leader of other task when cross-task chosen
- Evaluations performed on the individual's assigned task only (keeps functions pure where possible)
"""
from __future__ import annotations

import numpy as np
from typing import List, Sequence, Tuple

from src.metrics.fuzzy_entropy import compute_fuzzy_entropy

EPS = 1e-12


def _ensure_sorted_unique(vec: np.ndarray) -> np.ndarray:
    vals = np.clip(vec, 0.0, 255.0)
    vals = np.sort(vals)
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + 1e-3
    return vals


def continuous_to_thresholds(pos: np.ndarray, K: int) -> List[int]:
    arr = _ensure_sorted_unique(pos[:K])
    return [int(round(x)) for x in arr]


def mfwoa_multitask(
    hists: Sequence[np.ndarray],
    Ks: Sequence[int],
    pop_size: int = 50,
    iters: int = 200,
    rng: np.random.Generator = None,
    rmp_init: float = 0.3,
    pop_per_task: int = None,
    elitism: int = 3,
    rmp_schedule: Sequence[tuple] = None,
) -> Tuple[List[List[int]], List[float], dict]:
    """Run simplified MFWOA for multiple tasks simultaneously.

    Args:
        hists: sequence of histograms (one per task). For same image multiple Ks, pass same hist repeated.
        Ks: sequence of ints number of thresholds per task.
    Returns:
        (best_thresholds_per_task, best_scores_per_task)
    """
    if rng is None:
        rng = np.random.default_rng(123)
    T = len(Ks)
    maxK = max(Ks)
    # allow specifying population per task; if provided, scale total pop
    if pop_per_task is not None:
        pop_size = int(pop_per_task) * T
    # objectives per task
    def make_obj(hist, K):
        return lambda pos: compute_fuzzy_entropy(hist, continuous_to_thresholds(pos, K))

    objectives = [make_obj(h, K) for h, K in zip(hists, Ks)]

    # initialize population and skill factors
    pop = rng.uniform(0.0, 255.0, size=(pop_size, maxK))
    skill_factors = rng.integers(low=0, high=T, size=pop_size)
    fitness = np.full(pop_size, -np.inf)
    # evaluate initial
    for i in range(pop_size):
        sf = int(skill_factors[i])
        fitness[i] = objectives[sf](pop[i])
    # best per task
    best_pos = [None] * T
    best_score = [-np.inf] * T
    for t in range(T):
        mask = (skill_factors == t)
        if mask.any():
            idx = int(np.argmax(fitness * mask))
            best_pos[t] = pop[idx].copy()
            best_score[t] = float(fitness[idx])

    rmp = float(rmp_init)
    success_count = 0
    total_xt = 0

    # history logging
    history = {t: [] for t in range(T)}
    nfe = 0

    for g in range(iters):
        frac = g / max(1, iters)
        a = 2.0 * (1 - g / max(1, iters - 1))
        # rmp scheduling: if provided, override rmp according to schedule
        if rmp_schedule is not None:
            # rmp_schedule is sequence of tuples (start_frac, end_frac, rmp_value)
            for (s_frac, e_frac, val) in rmp_schedule:
                if frac >= s_frac and frac < e_frac:
                    rmp = float(val)
                    break
        # If single-task, we may want to force rmp to zero to avoid transfer noise
        rmp_local = 0.0 if T == 1 else rmp

        # Prepare next generation container (we will allow elitism preservation)
        next_pop = pop.copy()

        for i in range(pop_size):
            sf = int(skill_factors[i])
            p_cross = rng.random()
            if p_cross < rmp_local:
                # cross-task interaction: pick random other task's leader
                other_tasks = [t for t in range(T) if t != sf]
                if other_tasks:
                    other = rng.choice(other_tasks)
                    leader = best_pos[other]
                    # mix leader genes into current
                    beta = rng.random()
                    new_pos = pop[i] * (1 - beta) + leader * beta
                else:
                    # fallback to intra-task
                    leader = best_pos[sf]
                    new_pos = pop[i].copy()
            else:
                # intra-task WOA operations relative to own task's leader
                leader = best_pos[sf]
                r1 = rng.random()
                r2 = rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = rng.random()
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * leader - pop[i])
                        new_pos = leader - A * D
                    else:
                        rand_idx = rng.integers(pop_size)
                        X_rand = pop[rand_idx]
                        D = abs(C * X_rand - pop[i])
                        new_pos = X_rand - A * D
                else:
                    b = 1.0
                    l = rng.uniform(-1, 1)
                    distance = abs(leader - pop[i])
                    new_pos = distance * np.exp(b * l) * np.cos(2 * np.pi * l) + leader
            new_pos = _ensure_sorted_unique(new_pos)
            new_fit = objectives[sf](new_pos)
            total_xt += 1
            nfe += 1
            # replace in next_pop when improved
            if new_fit > fitness[i]:
                next_pop[i] = new_pos
                fitness[i] = new_fit
                success_count += 1
                if new_fit > best_score[sf]:
                    best_score[sf] = float(new_fit)
                    best_pos[sf] = new_pos.copy()

        # apply elitism: preserve top-N individuals per task into next_pop
        if elitism and elitism > 0:
            for t in range(T):
                mask = (skill_factors == t)
                if np.count_nonzero(mask) == 0:
                    continue
                # select indices of mask sorted by fitness desc
                idxs = np.where(mask)[0]
                # if fewer individuals than elitism, preserve all
                if idxs.size <= elitism:
                    continue
                sorted_idxs = idxs[np.argsort(-fitness[idxs])]
                elite_idxs = sorted_idxs[:elitism]
                # copy elites into their positions in next_pop
                for ei in elite_idxs:
                    next_pop[ei] = pop[ei].copy()
        # finalize population for next generation
        pop = next_pop
        # adapt rmp simply based on recent success fraction (if no explicit schedule)
        if rmp_schedule is None and total_xt > 0 and g % 10 == 0 and g > 0:
            frac_succ = success_count / (total_xt + EPS)
            # nudge rmp toward observed success (clamped)
            rmp = float(np.clip(0.5 * rmp + 0.5 * frac_succ, 0.05, 0.95))
            # reset counters
            success_count = 0
            total_xt = 0
        # log best per-task for history
        for t in range(T):
            history[t].append(best_score[t])
    # finalize best thresholds
    best_thresholds = [continuous_to_thresholds(best_pos[t], Ks[t]) if best_pos[t] is not None else [] for t in range(T)]
    diagnostics = {'history': history, 'nfe': nfe}
    return best_thresholds, [float(s) for s in best_score], diagnostics
