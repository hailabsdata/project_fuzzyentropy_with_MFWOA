"""Experiment harness: run MFWOA multitask across images and Ks and save results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.metrics.fuzzy_entropy import histogram_from_image
from src.optim.mfwoa_multitask import mfwoa_multitask


def run_on_folder(input_dir: Path, Ks: List[int], pop: int, iters: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        header = ["image"] + [f"K={k}_thresholds" for k in Ks] + [f"K={k}_score" for k in Ks]
        writer.writerow(header)

        for img_path in sorted(input_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            hist = histogram_from_image(img)
            hists = [hist for _ in Ks]
            best_ths, best_scores = mfwoa_multitask(hists, Ks, pop_size=pop, iters=iters)
            # save per-image json
            json_path = outdir / f"{img_path.stem}_results.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"thresholds": best_ths, "scores": best_scores}, jf, indent=2)
            row = [img_path.name] + [";".join(map(str, th)) for th in best_ths] + [str(s) for s in best_scores]
            writer.writerow(row)
    print(f"Experiments saved -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MFWOA experiments on folder of grayscale PNGs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--Ks", required=True, help="comma-separated K values, e.g. 2,3,4")
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", default="experiments")
    args = parser.parse_args()
    Ks = [int(x) for x in args.Ks.split(",")]
    run_on_folder(Path(args.input), Ks, args.pop, args.iters, Path(args.out))


if __name__ == "__main__":
    main()
