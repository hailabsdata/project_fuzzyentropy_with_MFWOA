"""CLI to compare algorithms on images and report PSNR/SSIM/time with benchmark plots."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
from src.optim.woa import woa_optimize
from src.optim.mfwoa_multitask import mfwoa_multitask
from src.optim.pso import pso_optimize
from src.optim.ga import ga_optimize
from src.optim.fcm import fcm_thresholding
from src.seg.thresholding import apply_thresholds
from src.metrics.metrics import psnr, ssim


def greedy_otsu_multi(hist: np.ndarray, K: int) -> List[int]:
    """Greedy multi-level Otsu: iteratively add thresholds that maximize between-class variance."""
    levels = np.arange(256)
    prob = hist / (hist.sum() + 1e-12)

    def between_class_variance(thresholds: List[int]) -> float:
        # compute class means and probabilities
        bins = [0] + sorted(thresholds) + [256]
        total_mean = (prob * levels).sum()
        bc_var = 0.0
        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            p = prob[a:b].sum()
            if p <= 0:
                continue
            mean = (prob[a:b] * levels[a:b]).sum() / p
            bc_var += p * (mean - total_mean) ** 2
        return float(bc_var)

    thresholds: List[int] = []
    candidates = list(range(1, 255))
    for _ in range(K):
        best_t = None
        best_score = -1.0
        for t in candidates:
            if t in thresholds:
                continue
            sc = between_class_variance(thresholds + [t])
            if sc > best_score:
                best_score = sc
                best_t = t
        if best_t is None:
            break
        thresholds.append(best_t)
    return sorted(thresholds)


def run_algorithms_on_image(img: np.ndarray, K: int, pop: int, iters: int) -> Dict[str, Dict[str, Any]]:
    """Run all algorithms on an image and return results."""
    hist = histogram_from_image(img)
    results = {}

    # Common objective functions
    def fe_from_pos(pos):
        ths = [int(round(x)) for x in pos]
        return compute_fuzzy_entropy(hist, ths, for_minimization=False)

    def fitness_from_pos(pos):
        # return value suitable for minimizers (i.e. -FE)
        ths = [int(round(x)) for x in pos]
        return compute_fuzzy_entropy(hist, ths, for_minimization=True)

    # object passed to PSO/GA/WOA should be a minimization objective
    obj_func = fitness_from_pos

    # Helper for visualizing results
    def visualize_result(thresholds: List[int]) -> np.ndarray:
        labels = apply_thresholds(img, thresholds)
        # Map each class to the midpoint gray level between bin edges for consistent PSNR
        bins = [0] + sorted(thresholds) + [255]
        midpoints = []
        for i in range(len(bins) - 1):
            a, b = bins[i], bins[i + 1]
            mid = int(round((a + b) / 2.0))
            midpoints.append(mid)
        out = np.zeros_like(img, dtype=np.uint8)
        for cls_id, mid in enumerate(midpoints):
            out[labels == cls_id] = mid
        return out

    # Otsu (greedy)
    t0 = time.perf_counter()
    otsu_th = greedy_otsu_multi(hist, K)
    t_otsu = time.perf_counter() - t0
    vis_otsu = visualize_result(otsu_th)
    fe_otsu = compute_fuzzy_entropy(hist, otsu_th, for_minimization=False)
    fitness_otsu = compute_fuzzy_entropy(hist, otsu_th, for_minimization=True)
    results['Otsu'] = {
        'thresholds': otsu_th,
        'time': t_otsu,
        'vis': vis_otsu,
        'fe': fe_otsu,
        'fitness': fitness_otsu,
    }

    # Fuzzy C-means
    t0 = time.perf_counter()
    fcm_th = fcm_thresholding(hist, K)
    t_fcm = time.perf_counter() - t0
    vis_fcm = visualize_result(fcm_th)
    fe_fcm = compute_fuzzy_entropy(hist, fcm_th, for_minimization=False)
    fitness_fcm = compute_fuzzy_entropy(hist, fcm_th, for_minimization=True)
    results['FCM'] = {
        'thresholds': fcm_th,
        'time': t_fcm,
        'vis': vis_fcm,
        'fe': fe_fcm,
        'fitness': fitness_fcm,
    }

    # PSO
    t0 = time.perf_counter()
    pso_th, pso_score = pso_optimize(hist, K, pop_size=pop, iters=iters, objective=obj_func)
    t_pso = time.perf_counter() - t0
    vis_pso = visualize_result(pso_th)
    fe_pso = compute_fuzzy_entropy(hist, pso_th, for_minimization=False)
    results['PSO'] = {
        'thresholds': pso_th,
        'time': t_pso,
        'vis': vis_pso,
        'fe': fe_pso,
        'fitness': float(pso_score),
    }

    # GA
    t0 = time.perf_counter()
    ga_th, ga_score = ga_optimize(hist, K, pop_size=pop, iters=iters, objective=obj_func)
    t_ga = time.perf_counter() - t0
    vis_ga = visualize_result(ga_th)
    fe_ga = compute_fuzzy_entropy(hist, ga_th, for_minimization=False)
    results['GA'] = {
        'thresholds': ga_th,
        'time': t_ga,
        'vis': vis_ga,
        'fe': fe_ga,
        'fitness': float(ga_score),
    }

    # WOA with Fuzzy Entropy
    t0 = time.perf_counter()
    woa_th, woa_score = woa_optimize(hist, K, pop_size=pop, iters=iters, objective=obj_func)
    t_woa = time.perf_counter() - t0
    vis_woa = visualize_result(woa_th)
    fe_woa = compute_fuzzy_entropy(hist, woa_th, for_minimization=False)
    results['WOA'] = {
        'thresholds': woa_th,
        'time': t_woa,
        'vis': vis_woa,
        'fe': fe_woa,
        'fitness': float(woa_score),
    }

    # MFWOA (multitask run with same hist repeated)
    t0 = time.perf_counter()
    hists = [hist]
    Ks = [K]
    best_ths, best_scores, mf_diag = mfwoa_multitask(hists, Ks, pop_size=pop, iters=iters)
    t_mf = time.perf_counter() - t0
    mf_th = best_ths[0]
    vis_mf = visualize_result(mf_th)
    # mfwoa_multitask returns scores that are FE (higher is better), convert to fitness for consistency
    fe_mf = float(best_scores[0]) if best_scores and len(best_scores) > 0 else compute_fuzzy_entropy(hist, mf_th, for_minimization=False)
    results['MFWOA'] = {
        'thresholds': mf_th,
        'time': t_mf,
        'vis': vis_mf,
        'fe': fe_mf,
        'fitness': float(-fe_mf),
        'mf_history': mf_diag.get('history') if isinstance(mf_diag, dict) else None,
        'mf_nfe': mf_diag.get('nfe') if isinstance(mf_diag, dict) else None,
    }

    return results


def plot_benchmark(results_data: list, out_dir: Path):
    """Plot benchmark comparisons."""
    # Convert results to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(results_data)
    
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Algorithm Comparison')
    
    # Time comparison
    sns.boxplot(data=df, x='algo', y='time', ax=ax1)
    ax1.set_title('Execution Time')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # PSNR comparison
    sns.boxplot(data=df, x='algo', y='psnr', ax=ax2)
    ax2.set_title('PSNR')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('PSNR')
    ax2.tick_params(axis='x', rotation=45)
    
    # SSIM comparison
    sns.boxplot(data=df, x='algo', y='ssim', ax=ax3)
    ax3.set_title('SSIM')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('SSIM')
    ax3.tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create convergence plot for metaheuristic algorithms
    plt.figure(figsize=(10, 6))
    for algo in ['PSO', 'GA', 'WOA', 'MFWOA']:
        scores = df[df['algo'] == algo]['score'].values
        plt.plot(scores, label=algo)
    plt.title('Convergence Comparison (Fuzzy Entropy)')
    plt.xlabel('Image Index')
    plt.ylabel('Fuzzy Entropy')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / 'convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare segmentation algorithms')
    parser.add_argument('--input', required=True)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--pop', type=int, default=30)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--out', default='compare_results')
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'compare.csv'
    
    # Store results for plotting
    results_data = []
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['image', 'algo', 'thresholds', 'time', 'psnr', 'ssim', 'fe', 'fitness'])
        images = []
        if input_path.is_dir():
            images = sorted(input_path.glob('*.png'))
        else:
            images = [input_path]
        
        for p in images:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res = run_algorithms_on_image(img, args.K, args.pop, args.iters)
            
            for algo, info in res.items():
                vis = info['vis']
                ps = psnr(img, vis)
                ss = ssim(img, vis)
                img_hist = histogram_from_image(img)
                fe_val = info.get('fe', compute_fuzzy_entropy(img_hist, info['thresholds'], for_minimization=False))
                fitness_val = info.get('fitness', compute_fuzzy_entropy(img_hist, info['thresholds'], for_minimization=True))

                # Save to CSV
                writer.writerow([
                    p.name, algo, ';'.join(map(str, info['thresholds'])),
                    f"{info['time']:.4f}", f"{ps:.4f}", f"{ss:.4f}", f"{fe_val:.6f}", f"{fitness_val:.6f}"
                ])

                # Store for plotting (keep 'score' as true FE for backwards compatibility)
                results_data.append({
                    'image': p.name,
                    'algo': algo,
                    'time': info['time'],
                    'psnr': ps,
                    'ssim': ss,
                    'score': fe_val,
                    'fe': fe_val,
                    'fitness': fitness_val,
                })
                
                # Save visualization and json
                vis_path = out_dir / f"{p.stem}_{algo}.png"
                json_path = out_dir / f"{p.stem}_{algo}.json"
                cv2.imwrite(str(vis_path), vis)
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump({
                        'thresholds': info['thresholds'],
                        'time': info['time'],
                        'psnr': ps,
                        'ssim': ss,
                        'fe': fe_val,
                        'fitness': fitness_val,
                    }, jf, indent=2)
    
    # Generate benchmark plots
    plot_benchmark(results_data, out_dir)
    print(f"Comparison saved -> {csv_path}")
    print(f"Benchmark plots saved -> {out_dir}/benchmark.png, {out_dir}/convergence.png")


if __name__ == '__main__':
    main()
