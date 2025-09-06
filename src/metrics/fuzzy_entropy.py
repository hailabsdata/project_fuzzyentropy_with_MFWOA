"""Fuzzy Entropy utilities (pure functions).

Các hàm thuần: histogram_from_image, compute_fuzzy_entropy, membership generators.
"""
from __future__ import annotations

import numpy as np
from typing import Literal, Sequence

EPS = 1e-12


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
    hist: np.ndarray, thresholds: Sequence[int], membership: MembershipType = "triangular"
) -> float:
    """Tính tổng Fuzzy Entropy cho một tập ngưỡng.

    Args:
        hist: 256-bin histogram (counts).
        thresholds: iterable các ngưỡng trong [0..255], not necessarily sorted.
        membership: 'triangular' hoặc 'gaussian'.

    Returns:
        tổng entropy (float). Lớn hơn là tốt hơn.
    """
    if hist.shape[0] != 256:
        raise ValueError("hist must be length-256 array")
    th = np.array(sorted(int(t) for t in thresholds))
    # define centers: include 0 and 255 as boundary centers
    centers = np.concatenate(([0.0], th.astype(float), [255.0]))
    C = centers.size
    if membership == "triangular":
        mu = _triangular_membership(centers)
    elif membership == "gaussian":
        mu = _gaussian_membership(centers)
    else:
        raise ValueError("unknown membership")
    # normalize histogram to probability mass per gray level
    p_levels = hist / (np.sum(hist) + EPS)
    # fuzzy probability per class: sum over gray levels of mu * p_levels
    p_classes = mu.dot(p_levels)
    # avoid zeros
    p_classes = np.clip(p_classes, EPS, 1.0)
    H = -np.sum(p_classes * np.log(p_classes))
    return float(H)
