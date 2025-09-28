"""PSO implementation for image thresholding."""
from typing import Tuple, Callable
import numpy as np


def pso_optimize(hist: np.ndarray, K: int, pop_size: int, iters: int,
                objective: Callable[[np.ndarray], float]) -> Tuple[list[int], float]:
    """PSO optimization for thresholding.
    
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
    
    # Initialize particles evenly across grayscale range
    particles = np.zeros((pop_size, K), dtype=np.float32)
    for i in range(pop_size):
        # Random perturbation around evenly spaced positions
        base = np.linspace(1, 254, K+2)[1:-1]  # Skip boundaries
        noise = np.random.uniform(-10, 10, K)
        particles[i] = np.clip(base + noise, 1, 254)
    
    velocities = np.zeros_like(particles)
    
    # Parameters
    w = 0.7  # Inertia
    c1 = 2.0  # Cognitive (particle)
    c2 = 2.0  # Social (swarm)
    
    # Personal best
    pbest = particles.copy()
    pbest_scores = np.array([objective(enforce_threshold_constraints(p)) for p in pbest])
    
    # Global best
    gbest_idx = pbest_scores.argmin()
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    for iter in range(iters):
        # Linearly decrease inertia
        w_iter = w * (1 - iter / iters)

        # Update velocities using per-dimension random factors
        r1 = np.random.random(particles.shape)
        r2 = np.random.random(particles.shape)
        velocities = (
            w_iter * velocities
            + c1 * r1 * (pbest - particles)
            + c2 * r2 * (gbest - particles)
        )

        # Limit velocity magnitude
        velocities = np.clip(velocities, -5.0, 5.0)

        # Update positions
        particles = particles + velocities

        # Enforce constraints on each particle
        for i in range(pop_size):
            particles[i] = enforce_threshold_constraints(particles[i])

        # Update personal bests
        scores = np.array([objective(enforce_threshold_constraints(p)) for p in particles])
        improved = scores < pbest_scores
        if np.any(improved):
            pbest[improved] = particles[improved]
            pbest_scores[improved] = scores[improved]

        # Update global best
        min_idx = pbest_scores.argmin()
        if pbest_scores[min_idx] < gbest_score:
            gbest = pbest[min_idx].copy()
            gbest_score = pbest_scores[min_idx]
    
    # Final constraints and conversion to integers
    final_thresholds = enforce_threshold_constraints(gbest)
    return [int(t) for t in final_thresholds], float(gbest_score)