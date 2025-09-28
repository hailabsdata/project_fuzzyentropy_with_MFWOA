"""Standard Whale Optimization Algorithm implementation."""
from typing import Tuple, Callable
import numpy as np


def woa_optimize(hist: np.ndarray, K: int, pop_size: int, iters: int,
                objective: Callable[[np.ndarray], float]) -> Tuple[list[int], float]:
    """Standard WOA optimization for thresholding.
    
    Args:
        hist: Image histogram
        K: Number of thresholds
        pop_size: Population size (should be >= 20*K)
        iters: Number of iterations (should be >= 100)
        objective: Objective function to minimize
    
    Returns:
        Best thresholds and score
    """
    from src.seg.utils import enforce_threshold_constraints
    
    # Initialize population with good spread
    whales = np.zeros((pop_size, K), dtype=np.float32)
    for i in range(pop_size):
        # Random perturbation around evenly spaced positions
        base = np.linspace(1, 254, K+2)[1:-1]  # Skip boundaries
        noise = np.random.uniform(-10, 10, K)
        whales[i] = np.clip(base + noise, 1, 254)
    
    best_whale = None
    best_score = float('inf')
    
    # WOA parameters
    b = 1  # spiral parameter
    
    for t in range(iters):
        # Update a linearly from 2 to 0
        a = 2 * (1 - t/iters)
        
        # For each whale
        for i in range(pop_size):
            # Evaluate current whale with constraints
            current = enforce_threshold_constraints(whales[i])
            score = objective(current)
            
            # Update best solution
            if score < best_score:
                best_score = score
                best_whale = current.copy()
            
            # Random values for update mechanism
            r = np.random.random()
            A = 2 * a * r - a  # [-a,a]
            C = 2 * r  # [0,2]
            l = np.random.uniform(-1, 1)  # [-1,1]
            p = np.random.random()  # probability for update type
            
            if p < 0.5:
                # Encircling prey or search for prey
                if abs(A) < 1:
                    # Encircling prey (exploitation)
                    D = abs(C * best_whale - current)
                    new_pos = best_whale - A * D
                else:
                    # Search for prey (exploration)
                    random_idx = np.random.randint(pop_size)
                    random_whale = enforce_threshold_constraints(whales[random_idx])
                    D = abs(C * random_whale - current)
                    new_pos = random_whale - A * D
            else:
                # Spiral update (local search)
                D = abs(best_whale - current)
                spiral = D * np.exp(b * l) * np.cos(2 * np.pi * l)
                new_pos = best_whale + spiral
            
            # Adaptive step size reduction
            step_scale = 1.0 - 0.9 * (t/iters)
            new_pos = current + step_scale * (new_pos - current)
            
            # Enforce constraints
            new_pos = enforce_threshold_constraints(new_pos)
            whales[i] = new_pos
    
    # Final constraints and conversion to integers
    final_thresholds = enforce_threshold_constraints(best_whale)
    return [int(t) for t in final_thresholds], float(best_score)