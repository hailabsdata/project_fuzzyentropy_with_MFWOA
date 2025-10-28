"""CLI to compare algorithms on images and report PSNR/SSIM/time with benchmark plots."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from pathlib import Path
from src.cli.seed_logger import SeedLogger

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# [UNCHANGED] – các import gốc
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
            mean = (prob[a:b] * levels[a:b]).sum() / (p + 1e-12)
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


# =========================
# [ADDED] Tiện ích chung
# =========================
def visualize_result(img_gray: np.ndarray, thresholds: List[int]) -> np.ndarray:
    """Map mỗi lớp thành midpoint cường độ để tính PSNR/SSIM nhất quán."""
    labels = apply_thresholds(img_gray, thresholds)
    bins = [0] + sorted(thresholds) + [255]
    mids = [int(round((bins[i] + bins[i + 1]) / 2.0)) for i in range(len(bins) - 1)]
    out = np.zeros_like(img_gray, dtype=np.uint8)
    for cls_id, mid in enumerate(mids):
        out[labels == cls_id] = mid
    return out


# ===========================================
# [UNCHANGED] Chạy so sánh đơn K (single-K)
# ===========================================
from typing import Any, Dict
import time
import numpy as np

def run_algorithms_on_image(
    img: np.ndarray,
    K: int,
    pop: int,
    iters: int,
    *,
    # --- các tham số mới, tùy chọn (KHÔNG bắt buộc) ---
    seed: int = 42,
    save_per_seed: bool = False,
    save_root: str | 'Path' = "results",
    image_stem: str | None = None,
    noise_tag: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Chạy Otsu/FCM/PSO/GA/WOA/MFWOA (single-K) trên 1 ảnh và trả dict kết quả.
    Nếu save_per_seed=True thì sẽ lưu CSV + mask + overlay cho từng seed/thuật toán/K.

    Trả về:
      {
        'Otsu': {K, thresholds, time, vis, fe, fitness, psnr?, ssim?},
        'FCM':  {...},
        'PSO':  {...},
        'GA':   {...},
        'WOA':  {...},
        'MFWOA':{..., 'mf_history'?, 'mf_nfe'?}
      }
    """
    # logger (nếu bạn đã tạo file src/cli/seed_logger.py)
    try:
        from src.cli.seed_logger import SeedLogger
    except Exception:
        SeedLogger = None  # vẫn chạy được nếu chưa có file

    # --- Helper nhỏ: thresholds -> mask nhãn 0..K ---
    def _thresholds_to_mask(img_gray: np.ndarray, ths: list[int]) -> np.ndarray:
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        t = np.sort(np.array(ths, dtype=np.int32))
        bins = np.concatenate(([0], t, [255]))
        mask = np.zeros_like(img_gray, dtype=np.uint8)
        C = len(bins) - 1  # số lớp = K+1
        for c in range(C):
            lo, hi = bins[c], bins[c+1]
            if c == C - 1:
                mask[img_gray >= lo] = c
            else:
                mask[(img_gray >= lo) & (img_gray < hi)] = c
        return mask

    # --- Chuẩn bị chung ---
    img_gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = histogram_from_image(img)
    results: Dict[str, Dict[str, Any]] = {}

    # mục tiêu (minimization) cho PSO/GA/WOA
    def fitness_from_pos(pos):
        ths = [int(round(x)) for x in pos]
        return compute_fuzzy_entropy(hist, ths, for_minimization=True)

    # logger theo seed (nếu bật)
    logger = None
    if save_per_seed and SeedLogger is not None:
        image_stem = image_stem or "image"
        logger = SeedLogger(save_root, image_stem, noise_tag)

    # ===================== OTSU =====================
    t0 = time.perf_counter()
    otsu_th = greedy_otsu_multi(hist, K)
    t_otsu = time.perf_counter() - t0
    vis_otsu = visualize_result(img, otsu_th)
    fe_otsu = compute_fuzzy_entropy(hist, otsu_th, for_minimization=False)
    fitness_otsu = compute_fuzzy_entropy(hist, otsu_th, for_minimization=True)
    psnr_otsu = None
    ssim_otsu = None
    try:
        psnr_otsu = float(_psnr(img_gray, vis_otsu))
        ssim_otsu = float(_ssim(img_gray, vis_otsu))
    except Exception:
        pass
    results['Otsu'] = {
        'K': K, 'thresholds': otsu_th, 'time': t_otsu,
        'vis': vis_otsu, 'fe': fe_otsu, 'fitness': fitness_otsu,
        'psnr': psnr_otsu, 'ssim': ssim_otsu,
    }
    # [LOG] Otsu
    try:
        if logger:
            logger.log(
                seed=seed, algo="Otsu", K=K,
                metrics={"FE": fe_otsu, "PSNR": psnr_otsu, "SSIM": ssim_otsu, "time_ms": int(t_otsu*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, otsu_th)
            )
    except Exception as e:
        print(f"[SeedLogger] Otsu warn: {e}")

    # ===================== FCM =====================
    t0 = time.perf_counter()
    fcm_th = fcm_thresholding(hist, K)
    t_fcm = time.perf_counter() - t0
    vis_fcm = visualize_result(img, fcm_th)
    fe_fcm = compute_fuzzy_entropy(hist, fcm_th, for_minimization=False)
    fitness_fcm = compute_fuzzy_entropy(hist, fcm_th, for_minimization=True)
    psnr_fcm = None
    ssim_fcm = None
    try:
        psnr_fcm = float(_psnr(img_gray, vis_fcm))
        ssim_fcm = float(_ssim(img_gray, vis_fcm))
    except Exception:
        pass
    results['FCM'] = {
        'K': K, 'thresholds': fcm_th, 'time': t_fcm,
        'vis': vis_fcm, 'fe': fe_fcm, 'fitness': fitness_fcm,
        'psnr': psnr_fcm, 'ssim': ssim_fcm,
    }
    # [LOG] FCM
    try:
        if logger:
            logger.log(
                seed=seed, algo="FCM", K=K,
                metrics={"FE": fe_fcm, "PSNR": psnr_fcm, "SSIM": ssim_fcm, "time_ms": int(t_fcm*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, fcm_th)
            )
    except Exception as e:
        print(f"[SeedLogger] FCM warn: {e}")

    # ===================== PSO =====================
    t0 = time.perf_counter()
    pso_th, pso_score = pso_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    t_pso = time.perf_counter() - t0
    vis_pso = visualize_result(img, pso_th)
    fe_pso = compute_fuzzy_entropy(hist, pso_th, for_minimization=False)
    psnr_pso = None
    ssim_pso = None
    try:
        psnr_pso = float(_psnr(img_gray, vis_pso))
        ssim_pso = float(_ssim(img_gray, vis_pso))
    except Exception:
        pass
    results['PSO'] = {
        'K': K, 'thresholds': pso_th, 'time': t_pso,
        'vis': vis_pso, 'fe': fe_pso, 'fitness': float(pso_score),
        'psnr': psnr_pso, 'ssim': ssim_pso,
    }
    # [LOG] PSO
    try:
        if logger:
            logger.log(
                seed=seed, algo="PSO", K=K,
                metrics={"FE": fe_pso, "PSNR": psnr_pso, "SSIM": ssim_pso, "time_ms": int(t_pso*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, pso_th)
            )
    except Exception as e:
        print(f"[SeedLogger] PSO warn: {e}")

    # ===================== GA =====================
    t0 = time.perf_counter()
    ga_th, ga_score = ga_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    t_ga = time.perf_counter() - t0
    vis_ga = visualize_result(img, ga_th)
    fe_ga = compute_fuzzy_entropy(hist, ga_th, for_minimization=False)
    psnr_ga = None
    ssim_ga = None
    try:
        psnr_ga = float(_psnr(img_gray, vis_ga))
        ssim_ga = float(_ssim(img_gray, vis_ga))
    except Exception:
        pass
    results['GA'] = {
        'K': K, 'thresholds': ga_th, 'time': t_ga,
        'vis': vis_ga, 'fe': fe_ga, 'fitness': float(ga_score),
        'psnr': psnr_ga, 'ssim': ssim_ga,
    }
    # [LOG] GA
    try:
        if logger:
            logger.log(
                seed=seed, algo="GA", K=K,
                metrics={"FE": fe_ga, "PSNR": psnr_ga, "SSIM": ssim_ga, "time_ms": int(t_ga*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, ga_th)
            )
    except Exception as e:
        print(f"[SeedLogger] GA warn: {e}")

    # ===================== WOA =====================
    t0 = time.perf_counter()
    woa_th, woa_score = woa_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    t_woa = time.perf_counter() - t0
    vis_woa = visualize_result(img, woa_th)
    fe_woa = compute_fuzzy_entropy(hist, woa_th, for_minimization=False)
    psnr_woa = None
    ssim_woa = None
    try:
        psnr_woa = float(_psnr(img_gray, vis_woa))
        ssim_woa = float(_ssim(img_gray, vis_woa))
    except Exception:
        pass
    results['WOA'] = {
        'K': K, 'thresholds': woa_th, 'time': t_woa,
        'vis': vis_woa, 'fe': fe_woa, 'fitness': float(woa_score),
        'psnr': psnr_woa, 'ssim': ssim_woa,
    }
    # [LOG] WOA
    try:
        if logger:
            logger.log(
                seed=seed, algo="WOA", K=K,
                metrics={"FE": fe_woa, "PSNR": psnr_woa, "SSIM": ssim_woa, "time_ms": int(t_woa*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, woa_th)
            )
    except Exception as e:
        print(f"[SeedLogger] WOA warn: {e}")

    # ===================== MFWOA (single-task) =====================
    t0 = time.perf_counter()
    hists = [hist]
    Ks = [K]
    best_ths, best_scores, mf_diag = mfwoa_multitask(hists, Ks, pop_size=pop, iters=iters)
    t_mf = time.perf_counter() - t0
    mf_th = best_ths[0]
    vis_mf = visualize_result(img, mf_th)
    # best_scores[0] kỳ vọng là FE (maximize). Nếu không có, tính lại.
    fe_mf = float(best_scores[0]) if (isinstance(best_scores, (list, tuple)) and len(best_scores) > 0) \
            else compute_fuzzy_entropy(hist, mf_th, for_minimization=False)
    psnr_mf = None
    ssim_mf = None
    try:
        psnr_mf = float(_psnr(img_gray, vis_mf))
        ssim_mf = float(_ssim(img_gray, vis_mf))
    except Exception:
        pass
    results['MFWOA'] = {
        'K': K, 'thresholds': mf_th, 'time': t_mf,
        'vis': vis_mf, 'fe': fe_mf,
        'fitness': float(-fe_mf),  # đồng bộ "minimization"
        'psnr': psnr_mf, 'ssim': ssim_mf,
        'mf_history': mf_diag.get('history') if isinstance(mf_diag, dict) else None,
        'mf_nfe': mf_diag.get('nfe') if isinstance(mf_diag, dict) else None,
    }
    # [LOG] MFWOA
    try:
        if logger:
            logger.log(
                seed=seed, algo="MFWOA", K=K,
                metrics={"FE": fe_mf, "PSNR": psnr_mf, "SSIM": ssim_mf, "time_ms": int(t_mf*1000)},
                img_gray=img_gray, mask=_thresholds_to_mask(img_gray, mf_th)
            )
    except Exception as e:
        print(f"[SeedLogger] MFWOA warn: {e}")

    return results



def plot_benchmark(results_data: list, out_dir: Path):
    """Plot benchmark comparisons."""
    import pandas as pd
    df = pd.DataFrame(results_data)

    # [ADDED] nếu chỉ có MFWOA đa nhiệm, các thuật toán khác có thể không tồn tại
    if df.empty:
        return

    # Set style
    plt.style.use('seaborn')

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
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

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

    # [UNCHANGED-ish] Convergence “giả lập” theo ảnh-index (không phải curve theo iteration)
    plt.figure(figsize=(10, 6))
    for algo in sorted(df['algo'].unique()):
        scores = df[df['algo'] == algo]['score'].values
        if scores.size > 0:
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
    # [ADDED] hỗ trợ đa nhiệm (multi-K) cho MFWOA
    parser.add_argument('--Ks', type=str, default=None,
                        help='Comma-separated Ks for MFWOA multitask, e.g., 2,3,4,5,6')
    # [ADDED] vẫn giữ single-K để backward compatible
    parser.add_argument('--K', type=int, default=3, help='Single K if not using --Ks (default: 3)')
    parser.add_argument('--pop', type=int, default=30)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--out', default='compare_results')
    args = parser.parse_args()

    from datetime import datetime
    # [CLEANUP] dựng out_dir 1 lần
    out_dir = Path(args.out) if args.out and args.out.strip() != "" else Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    csv_path = out_dir / 'compare.csv'

    results_data = []

    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        # [CHANGED] thêm cột K để phân biệt các kết quả theo K
        writer.writerow(['image', 'algo', 'K', 'thresholds', 'time', 'psnr', 'ssim', 'fe', 'fitness'])

        # Chuẩn bị danh sách ảnh
        if input_path.is_dir():
            images = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.tif')) + list(input_path.glob('*.jpg')))
        else:
            images = [input_path]

        for p in images:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Cannot read image: {p}")
                continue

            # ===============================
            # [ADDED] Nhánh đa nhiệm MFWOA
            # ===============================
            if args.Ks:  # nếu có --Ks, chạy MFWOA đa nhiệm và BỎ QUA các thuật toán khác
                Ks_list = [int(x) for x in args.Ks.split(',') if x.strip()]
                mf_results = run_mfwoa_multitask_on_image(img, Ks_list, args.pop, args.iters)

                for r in mf_results:
                    vis = r['vis']
                    ps = psnr(img, vis)
                    ss = ssim(img, vis)
                    # Ghi CSV
                    writer.writerow([
                        p.name, r['algo'], r['K'],
                        ';'.join(map(str, r['thresholds'])),
                        f"{r['time']:.4f}", f"{ps:.4f}", f"{ss:.4f}",
                        f"{r['fe']:.6f}", f"{r['fitness']:.6f}"
                    ])
                    # Lưu hình/JSON
                    vis_path = out_dir / f"{p.stem}_{r['algo']}_K{r['K']}.png"
                    json_path = out_dir / f"{p.stem}_{r['algo']}_K{r['K']}.json"
                    cv2.imwrite(str(vis_path), vis)
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump({
                            'algo': r['algo'],
                            'K': r['K'],
                            'thresholds': r['thresholds'],
                            'time': r['time'],
                            'psnr': ps,
                            'ssim': ss,
                            'fe': r['fe'],
                            'fitness': r['fitness'],
                            'mf_diag_keys': list(r.get('mf_diag', {}).keys()) if isinstance(r.get('mf_diag'), dict) else None
                        }, jf, indent=2)

                    # Thêm vào data để vẽ biểu đồ
                    results_data.append({
                        'image': p.name,
                        'algo': r['algo'],
                        'K': r['K'],
                        'time': r['time'],
                        'psnr': ps,
                        'ssim': ss,
                        'score': r['fe'],      # dùng FE làm "score"
                        'fe': r['fe'],
                        'fitness': r['fitness'],
                    })

                # [IMPORTANT] tiếp ảnh kế tiếp, không chạy các thuật toán khác
                continue

            # ==================================
            # [UNCHANGED] Nhánh single-K (so sánh)
            # ==================================
            res = run_algorithms_on_image(img, args.K, args.pop, args.iters)

            for algo, info in res.items():
                vis = info['vis']
                ps = psnr(img, vis)
                ss = ssim(img, vis)
                img_hist = histogram_from_image(img)
                fe_val = info.get('fe', compute_fuzzy_entropy(img_hist, info['thresholds'], for_minimization=False))
                fitness_val = info.get('fitness', compute_fuzzy_entropy(img_hist, info['thresholds'], for_minimization=True))
                K_used = info.get('K', args.K)

                # Save to CSV
                writer.writerow([
                    p.name, algo, K_used,
                    ';'.join(map(str, info['thresholds'])),
                    f"{info['time']:.4f}", f"{ps:.4f}", f"{ss:.4f}",
                    f"{fe_val:.6f}", f"{fitness_val:.6f}"
                ])

                # Store for plotting
                results_data.append({
                    'image': p.name,
                    'algo': algo,
                    'K': K_used,
                    'time': info['time'],
                    'psnr': ps,
                    'ssim': ss,
                    'score': fe_val,   # FE
                    'fe': fe_val,
                    'fitness': fitness_val,
                })

                # Save visualization and json
                # [ADDED] gắn hậu tố _K{K} để rõ ràng
                vis_path = out_dir / f"{p.stem}_{algo}_K{K_used}.png"
                json_path = out_dir / f"{p.stem}_{algo}_K{K_used}.json"
                cv2.imwrite(str(vis_path), vis)
                with open(json_path, 'w', encoding='utf-8') as jf:
                    json.dump({
                        'algo': algo,
                        'K': K_used,
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
