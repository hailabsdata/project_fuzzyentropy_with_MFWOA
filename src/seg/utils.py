"""Utilities for threshold constraints and visualization."""
from typing import List, Tuple
import numpy as np


def enforce_threshold_constraints(thresholds: np.ndarray, min_gap: int = 1) -> np.ndarray:
    """Enforce constraints on thresholds: sorting, clipping, and minimum gap.
    
    Args:
        thresholds: Array of threshold values
        min_gap: Minimum gap between consecutive thresholds
    
    Returns:
        Constrained threshold array
    """
    # Round and sort
    t = np.round(thresholds)
    t.sort()
    
    # Clip to valid range
    t = np.clip(t, 1, 254)
    
    # Enforce minimum gap
    for i in range(1, len(t)):
        if t[i] - t[i-1] < min_gap:
            t[i] = t[i-1] + min_gap
    
    # Clip again in case enforcing gaps pushed values out of range
    t = np.clip(t, 1, 254)
    
    return t


def get_level_mapping(thresholds: List[int]) -> List[float]:
    """Get intensity level mapping for segmentation based on midpoints.
    
    Args:
        thresholds: List of threshold values
    
    Returns:
        List of intensity levels for each segment
    """
    bounds = [0] + sorted(thresholds) + [255]
    return [(bounds[i] + bounds[i+1])/2 for i in range(len(bounds)-1)]


def apply_thresholds_with_levels(img: np.ndarray, thresholds: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Apply thresholds and return both segmented image and pixel counts.
    
    Args:
        img: Input grayscale image
        thresholds: List of threshold values
    
    Returns:
        Tuple of (segmented image, pixel counts per class)
    """
    levels = get_level_mapping(thresholds)
    bounds = [0] + sorted(thresholds) + [255]
    result = np.zeros_like(img, dtype=np.float32)
    pixel_counts = []
    
    for i in range(len(bounds)-1):
        mask = (img >= bounds[i]) & (img < bounds[i+1])
        result[mask] = levels[i]
        pixel_counts.append(int(mask.sum()))
    
    return result.astype(np.uint8), pixel_counts