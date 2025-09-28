"""GA implementation for image thresholding."""
from typing import Tuple, Callable
import numpy as np


def ga_optimize(hist: np.ndarray, K: int, pop_size: int, iters: int,
               objective: Callable[[np.ndarray], float]) -> Tuple[list[int], float]:
    """GA optimization for thresholding.
    
    Args:
        hist: Image histogram
        K: Number of thresholds
        pop_size: Population size (should be >= 20*K)
        iters: Number of iterations (should be >= 100)
        objective: Objective function (e.g. fuzzy entropy)
    
    Returns:
        Best thresholds and score
    """
    from src.seg.utils import enforce_threshold_constraints
    
    # Initialize population with good spread
    pop = np.zeros((pop_size, K), dtype=np.float32)
    for i in range(pop_size):
        # Random perturbation around evenly spaced positions
        base = np.linspace(1, 254, K+2)[1:-1]  # Skip boundaries
        noise = np.random.uniform(-10, 10, K)
        pop[i] = np.clip(base + noise, 1, 254)
    
    # Parameters
    mutation_rate = 0.1
    elite_size = max(1, pop_size // 10)
    mutation_range = 10.0  # Maximum mutation step size
    
    best_solution = None
    best_score = float('inf')
    
    for iter in range(iters):
        # Evaluate fitness
        scores = np.array([objective(enforce_threshold_constraints(p)) for p in pop])
        
        # Update best solution
        min_idx = scores.argmin()
        if scores[min_idx] < best_score:
            best_solution = pop[min_idx].copy()
            best_score = scores[min_idx]
        
        # Sort by fitness
        idx = np.argsort(scores)
        pop = pop[idx]
        
        # Elitism
        new_pop = [pop[i].copy() for i in range(elite_size)]
        
        # Adaptive mutation rate based on iteration
        current_mutation = mutation_rate * (1 - iter/iters)
        current_range = mutation_range * (1 - iter/iters)
        
        # Generate rest of new population
        while len(new_pop) < pop_size:
            # Tournament selection
            t_size = 3
            candidates = np.random.randint(0, pop_size//2, size=(t_size, 2))  # Focus on better half
            p1_idx = candidates[0].min()
            p2_idx = candidates[1].min()
            
            # Blend crossover (BLX-alpha)
            alpha = 0.3  # Blend parameter
            diff = pop[p2_idx] - pop[p1_idx]
            min_val = np.minimum(pop[p1_idx], pop[p2_idx]) - alpha * np.abs(diff)
            max_val = np.maximum(pop[p1_idx], pop[p2_idx]) + alpha * np.abs(diff)
            child = np.random.uniform(min_val, max_val)
            
            # Non-uniform mutation
            if np.random.random() < current_mutation:
                mutation_mask = np.random.random(K) < 0.3  # Mutate ~30% of genes
                mutation = np.random.uniform(-current_range, current_range, K)
                child[mutation_mask] += mutation[mutation_mask]
            
            # Enforce constraints
            child = enforce_threshold_constraints(child)
            new_pop.append(child)
        
        pop = np.array(new_pop)
    
    # Final constraints and conversion to integers
    final_thresholds = enforce_threshold_constraints(best_solution)
    return [int(t) for t in final_thresholds], float(best_score)