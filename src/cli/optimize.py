"""CLI entrypoint to run optimization on an image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
from src.optim.mfwoa import mfwoa_optimize
from src.seg.thresholding import apply_thresholds


def main():
    parser = argparse.ArgumentParser(description="MFWOA fuzzy entropy multilevel thresholding")
    parser.add_argument("--image", required=True, help="Path to grayscale image")
    parser.add_argument("--K", type=int, default=2, help="Number of thresholds")
    parser.add_argument("--pop", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    img_path = Path(args.image)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read image {img_path}")
    hist = histogram_from_image(img)

    def obj(pos):
        th = [int(round(x)) for x in pos]
        return compute_fuzzy_entropy(hist, th, membership="triangular")

    best_th, best_score = mfwoa_optimize(hist, args.K, pop_size=args.pop, iters=args.iters, objective=obj)

    labels = apply_thresholds(img, best_th)
    seg_path = out_dir / "segmented.png"
    # map labels to 0..255 for visualization
    Klabels = labels.max()
    vis = (labels.astype(np.float32) / max(1, Klabels) * 255.0).astype(np.uint8)
    cv2.imwrite(str(seg_path), vis)

    json_path = out_dir / "thresholds.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"thresholds": best_th, "score": best_score}, f, indent=2)

    print(f"done -> {seg_path}, {json_path}")


if __name__ == "__main__":
    main()
