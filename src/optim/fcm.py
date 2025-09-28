"""Fuzzy C-means implementation for image thresholding."""
from typing import List
import numpy as np
from scipy.spatial.distance import cdist


def fcm_thresholding(hist: np.ndarray, K: int, max_iter: int = 100) -> List[int]:
    """Fuzzy C-means thresholding with robust handling of edge cases.
    
    Args:
        hist: Image histogram
        K: Number of thresholds (clusters - 1)
        max_iter: Maximum number of iterations
        
    Returns:
        List of threshold values
    """
    # Prepare intensity levels and weights
    levels = np.arange(256)
    total = hist.sum()
    if total == 0:
        # Handle empty histogram
        return [int(255 * (i+1)/(K+1)) for i in range(K)]
    weights = hist / total
    
    # Initialize cluster centers evenly spread
    centers = np.linspace(0, 255, K+1)
    
    # Fuzzy parameter
    m = 2.0
    eps = 1e-10  # Small constant to avoid division by zero
    
    for _ in range(max_iter):
        old_centers = centers.copy()
        
        # Calculate distances to centers
        distances = cdist(levels.reshape(-1, 1), centers.reshape(-1, 1))
        
        # Handle zero distances
        distances = np.maximum(distances, eps)
        
        # Update membership matrix with numerical stability
        try:
            # Calculate membership values
            exp = -2/(m-1)
            tmp = distances ** exp
            # Normalize row-wise and handle potential infinities
            row_sums = tmp.sum(axis=1, keepdims=True)
            membership = np.divide(tmp, row_sums, where=row_sums>0)
            membership = np.nan_to_num(membership, nan=1.0/len(centers))
            
            # Update centers with weighted average
            membership_m = membership ** m
            numerator = membership_m.T @ (levels * weights)
            denominator = membership_m.T @ weights
            new_centers = np.divide(numerator, denominator, where=denominator>0)
            
            # Handle any remaining NaN centers
            valid_centers = ~np.isnan(new_centers)
            if valid_centers.all():
                centers = new_centers
            else:
                # Interpolate NaN centers
                valid_idx = np.where(valid_centers)[0]
                if len(valid_idx) > 0:
                    invalid_idx = np.where(~valid_centers)[0]
                    for idx in invalid_idx:
                        # Find nearest valid centers
                        nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                        if idx == 0:  # First center
                            centers[idx] = 0
                        elif idx == len(centers)-1:  # Last center
                            centers[idx] = 255
                        else:  # Interpolate
                            centers[idx] = centers[nearest]
                else:
                    # If all centers are invalid, reset to uniform distribution
                    centers = np.linspace(0, 255, K+1)
                
        except (RuntimeWarning, FloatingPointError):
            # If numerical issues occur, reset centers
            centers = np.linspace(0, 255, K+1)
            continue
            
        # Check convergence with tolerance
        if np.max(np.abs(centers - old_centers)) < 0.1:
            break
    
    # Ensure centers are properly ordered and within bounds
    centers = np.clip(np.sort(centers), 0, 255)
    
    # Convert centers to thresholds, avoiding duplicates
    thresholds = []
    for i in range(K):
        t = int(round((centers[i] + centers[i+1])/2))
        if not thresholds or t > thresholds[-1]:
            thresholds.append(t)
    
    # If we lost some thresholds due to duplicates, add evenly spaced ones
    if len(thresholds) < K:
        # Fallback: fill remaining thresholds by evenly spacing the available range
        existing = [0] + thresholds + [255]
        needed = K - len(thresholds)
        # compute gaps and fill proportionally
        new_thresholds = thresholds.copy()
        # simple strategy: use linspace over (0..255) excluding boundaries
        candidate = list(np.linspace(1, 254, K + 2)[1:-1].astype(int))
        # merge candidate values preserving uniqueness and order
        merged = []
        for c in candidate:
            if not new_thresholds or c > new_thresholds[-1]:
                merged.append(c)
            if len(merged) + len(new_thresholds) >= K:
                break
        # final thresholds: take existing ones and fill from merged candidates
        thresholds = []
        ei = 0
        mi = 0
        # merge sorted existing centers and new candidates
        existing_sorted = sorted(new_thresholds)
        while len(thresholds) < K and (ei < len(existing_sorted) or mi < len(merged)):
            next_existing = existing_sorted[ei] if ei < len(existing_sorted) else 1e9
            next_merged = merged[mi] if mi < len(merged) else 1e9
            if next_existing <= next_merged:
                if not thresholds or next_existing > thresholds[-1]:
                    thresholds.append(next_existing)
                ei += 1
            else:
                if not thresholds or next_merged > thresholds[-1]:
                    thresholds.append(next_merged)
                mi += 1
        # If still short, pad with evenly spaced values
        while len(thresholds) < K:
            thresholds.append(1 + len(thresholds) * (253 // (K + 1)))
    
    return thresholds