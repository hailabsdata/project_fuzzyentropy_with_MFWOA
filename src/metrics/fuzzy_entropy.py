"""Fuzzy Entropy utilities (pure functions).

Các hàm thuần: histogram_from_image, compute_fuzzy_entropy, membership generators.
"""
from __future__ import annotations

import numpy as np
from typing import Literal, Sequence

EPS = 1e-12
# When a threshold set produces empty classes, we treat the true FE as invalid/zero
# for plotting and comparison (FE_true). For optimizers we still need a strong
# penalty so they avoid such solutions; set FITNESS_PENALTY to a large positive
# value returned when for_minimization=True and classes are empty.
FITNESS_PENALTY = 1e6


def histogram_from_image(image: np.ndarray) -> np.ndarray:
    """Trả về histogram 256-bin (counts) cho ảnh xám 2D."""
    if image.ndim != 2:
        raise ValueError("image must be grayscale 2D array")
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 255))
    return hist.astype(float)


def _triangular_membership(centers: np.ndarray) -> np.ndarray:
    """Trả về ma trận mu (C x 256) theo hàm membership tam giác, centers in [0..255]."""
    levels = np.arange(256)
    C = centers.size
    mu = np.zeros((C, 256), dtype=float)
    # boundaries: midpoints between centers
    boundaries = np.empty(C + 1)
    boundaries[0] = 0.0
    boundaries[-1] = 255.0
    midpoints = (centers[:-1] + centers[1:]) / 2.0 if C > 1 else np.array([])
    if C > 1:
        boundaries[1:-1] = midpoints
    for i in range(C):
        left = boundaries[i]
        right = boundaries[i + 1]
        center = centers[i]
        # ramp up to center
        if center > left:
            left_idx = (levels >= left) & (levels <= center)
            mu[i, left_idx] = (levels[left_idx] - left) / (center - left + EPS)
        # ramp down from center
        if center < right:
            right_idx = (levels >= center) & (levels <= right)
            mu[i, right_idx] = (right - levels[right_idx]) / (right - center + EPS)
    # ensure numerical bounds
    mu = np.clip(mu, 0.0, 1.0)
    return mu


def _gaussian_membership(centers: np.ndarray, sigma: float = 16.0) -> np.ndarray:
    levels = np.arange(256)
    C = centers.size
    mu = np.zeros((C, 256), dtype=float)
    for i, c in enumerate(centers):
        mu[i] = np.exp(-0.5 * ((levels - c) / (sigma + EPS)) ** 2)
    mu = np.clip(mu, 0.0, 1.0)
    return mu


MembershipType = Literal["triangular", "gaussian"]


def compute_fuzzy_entropy(
    hist: np.ndarray, thresholds: Sequence[int], membership: MembershipType = "triangular",
    for_minimization: bool = False
) -> float:
    """Tính tổng Fuzzy Entropy cho một tập ngưỡng.

    Args:
        hist: 256-bin histogram (counts).
        thresholds: iterable các ngưỡng trong [0..255], not necessarily sorted.
        membership: 'triangular' hoặc 'gaussian'.
        for_minimization: Nếu True, trả về -entropy cho bài toán tối thiểu hóa.

    Returns:
        Entropy value (hoặc -entropy nếu for_minimization=True).
        Lớn hơn là tốt hơn (khi for_minimization=False).
    """
    if hist.shape[0] != 256:
        raise ValueError("hist must be length-256 array")
    
    # Chuẩn hóa histogram
    p_levels = hist / (np.sum(hist) + EPS)
    
    # Ràng buộc ngưỡng
    th = np.array([int(t) for t in thresholds])
    from src.seg.utils import enforce_threshold_constraints
    th = enforce_threshold_constraints(th)
    
    # Define centers and compute membership
    centers = np.concatenate(([0.0], th.astype(float), [255.0]))
    if membership == "triangular":
        mu = _triangular_membership(centers)
    elif membership == "gaussian":
        mu = _gaussian_membership(centers)
    else:
        raise ValueError("unknown membership")
    
    # Compute class probabilities
    p_classes = mu.dot(p_levels)
    
    # Penalize empty classes: produce a well-defined FE_true and a separate
    # fitness penalty for minimizers. FE_true is defined as 0.0 when any class
    # is empty (so plotting/comparison is stable). For minimization (optimizers)
    # return a large positive penalty so the optimizer steers away from invalid
    # threshold sets.
    if np.any(p_classes < EPS):
        # invalid split: no meaningful entropy
        fe_true = 0.0
        if for_minimization:
            return float(FITNESS_PENALTY)
        else:
            return float(fe_true)
    else:
        # Compute entropy with numerical stability
        H = -np.sum(p_classes * np.log(p_classes + EPS))
        return float(-H if for_minimization else H)
