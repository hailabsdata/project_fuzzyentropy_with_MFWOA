"""Multilevel thresholding utilities (pure functions).
"""
from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple


def apply_thresholds(image: np.ndarray, thresholds: Sequence[int]) -> np.ndarray:
    """Trả về ảnh phân đoạn (labels 0..K) bằng cách dùng np.digitize.

    thresholds should be sorted or unsorted; result labels are 0..len(thresholds).
    """
    if image.ndim != 2:
        raise ValueError("image must be grayscale 2D array")
    bins = np.array(sorted(int(t) for t in thresholds), dtype=int)
    labels = np.digitize(image, bins, right=False)
    return labels


def labels_to_mask(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Trả về mask boolean cho class_id."""
    return (labels == class_id).astype(np.uint8) * 255
