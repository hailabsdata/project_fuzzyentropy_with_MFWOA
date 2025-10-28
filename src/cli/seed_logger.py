from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import cv2

class SeedLogger:
    def __init__(self, root: str | Path, image_stem: str, noise_tag: str | None = None):
        self.root = Path(root)
        self.image_stem = image_stem
        self.noise_tag = noise_tag or "clean"
        self.base = self.root / image_stem / self.noise_tag
        self.base.mkdir(parents=True, exist_ok=True)

        self.agg_csv = self.base / "metrics_all_seeds.csv"
        self._csv_fields = ["seed","algo","K","FE","PSNR","SSIM","time_ms","notes","out_mask","out_overlay"]
        self._csv_inited = self.agg_csv.exists()

    def _append_row(self, row: dict):
        with self.agg_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_fields)
            if not self._csv_inited:
                w.writeheader()
                self._csv_inited = True
            w.writerow(row)

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path):
        m = (mask.astype(np.uint8) if mask.dtype != np.uint8 else mask)
        cv2.imwrite(str(path), m)

    @staticmethod
    def _save_overlay(img_gray: np.ndarray, mask: np.ndarray, path: Path, alpha: float = 0.6):
        # tô màu theo nhãn mask rồi chồng lên ảnh gốc xám
        if img_gray.ndim == 2:
            g3 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            g3 = img_gray
        # tạo màu “an toàn” từ nhãn
        lab = mask.astype(np.int32)
        if lab.max() == 0:
            cm = np.zeros((*lab.shape, 3), np.uint8)
        else:
            norm = (lab * (255 // max(int(lab.max()), 1))).astype(np.uint8)
            cm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        over = (alpha * g3 + (1 - alpha) * cm).astype(np.uint8)
        cv2.imwrite(str(path), over)

    def log(self, *, seed: int, algo: str, K: int, metrics: dict,
            img_gray: np.ndarray, mask: np.ndarray, subdir: str | None = None, notes: str | None = None):
        # thư mục per-seed
        seed_dir = self.base / f"seed_{seed:03d}"
        if subdir:
            seed_dir = seed_dir / subdir
        seed_dir.mkdir(parents=True, exist_ok=True)

        mask_path = seed_dir / f"{algo}_K{K}_mask.png"
        overlay_path = seed_dir / f"{algo}_K{K}_overlay.png"
        self._save_mask(mask, mask_path)
        self._save_overlay(img_gray, mask, overlay_path)

        row = {
            "seed": seed,
            "algo": algo,
            "K": K,
            "FE": metrics.get("FE"),
            "PSNR": metrics.get("PSNR"),
            "SSIM": metrics.get("SSIM"),
            "time_ms": int(metrics.get("time_ms", 0)),
            "notes": notes or "",
            "out_mask": str(mask_path.relative_to(self.base)),
            "out_overlay": str(overlay_path.relative_to(self.base)),
        }
        self._append_row(row)

        # CSV riêng cho seed
        per_seed_csv = seed_dir / "metrics.csv"
        header_new = not per_seed_csv.exists()
        with per_seed_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if header_new:
                w.writeheader()
            w.writerow(row)
