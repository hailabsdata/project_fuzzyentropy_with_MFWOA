"""CLI to compare algorithms (MFWOA, WOA, Otsu) on images and report PSNR/SSIM/time."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
from src.optim.mfwoa import mfwoa_optimize
from src.optim.mfwoa_multitask import mfwoa_multitask
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


def run_algorithms_on_image(img: np.ndarray, K: int, pop: int, iters: int):
    hist = histogram_from_image(img)
    results = {}
    # Otsu (greedy)
    t0 = time.perf_counter()
    otsu_th = greedy_otsu_multi(hist, K)
    t_otsu = time.perf_counter() - t0
    labels_otsu = apply_thresholds(img, otsu_th)
    vis_otsu = (labels_otsu.astype(np.float32) / max(1, labels_otsu.max()) * 255.0).astype(np.uint8)
    results['Otsu'] = {'thresholds': otsu_th, 'time': t_otsu, 'vis': vis_otsu}

    # WOA
    t0 = time.perf_counter()
    def obj_woa(pos):
        return compute_fuzzy_entropy(hist, [int(round(x)) for x in pos])
    woa_th, woa_score = mfwoa_optimize(hist, K, pop_size=pop, iters=iters, objective=obj_woa)
    t_woa = time.perf_counter() - t0
    labels_woa = apply_thresholds(img, woa_th)
    vis_woa = (labels_woa.astype(np.float32) / max(1, labels_woa.max()) * 255.0).astype(np.uint8)
    results['WOA'] = {'thresholds': woa_th, 'time': t_woa, 'vis': vis_woa, 'score': woa_score}

    # MFWOA (multitask run with same hist repeated)
    t0 = time.perf_counter()
    hists = [hist]
    Ks = [K]
    best_ths, best_scores = mfwoa_multitask(hists, Ks, pop_size=pop, iters=iters)
    t_mf = time.perf_counter() - t0
    mf_th = best_ths[0]
    labels_mf = apply_thresholds(img, mf_th)
    vis_mf = (labels_mf.astype(np.float32) / max(1, labels_mf.max()) * 255.0).astype(np.uint8)
    results['MFWOA'] = {'thresholds': mf_th, 'time': t_mf, 'vis': vis_mf, 'score': best_scores[0]}

    return results


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
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['image', 'algo', 'thresholds', 'time', 'psnr', 'ssim'])
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
                writer.writerow([p.name, algo, ';'.join(map(str, info['thresholds'])), f"{info['time']:.4f}", f"{ps:.4f}", f"{ss:.4f}"])
                # save visualization and json
                vis_path = out_dir / f"{p.stem}_{algo}.png"
                json_path = out_dir / f"{p.stem}_{algo}.json"
                cv2.imwrite(str(vis_path), vis)
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump({'thresholds': info['thresholds'], 'time': info['time']}, jf, indent=2)
    print(f"Comparison saved -> {csv_path}")


if __name__ == '__main__':
    main()
