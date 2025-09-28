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
    while len(thresholds) < K:
        for i in range(len(thresholds)-1):
            if thresholds[i+1] - thresholds[i] > 2:
                thresholds.insert(i+1, thresholds[i] + (thresholds[i+1] - thresholds[i])//2)
                if len(thresholds) == K:
                    break
    
    return thresholds