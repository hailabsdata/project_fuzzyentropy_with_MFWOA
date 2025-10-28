"""Image quality metrics: PSNR and SSIM (pure functions).

Simple, dependency-light implementations suitable for grayscale images.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(original: np.ndarray, compared: np.ndarray, data_range: float = 255.0) -> float:
	if original.shape != compared.shape:
		raise ValueError("shapes must match")
	mse = np.mean((original.astype(np.float64) - compared.astype(np.float64)) ** 2)
	if mse == 0:
		return float('inf')
	return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(original: np.ndarray, compared: np.ndarray, data_range: float = 255.0, gaussian_sigma: float = 1.5) -> float:
	"""Compute a single-scale SSIM index for two grayscale images.

	Reference: Wang et al., 2004. Simplified implementation using gaussian filter.
	"""
	if original.shape != compared.shape:
		raise ValueError("shapes must match")
	orig = original.astype(np.float64)
	comp = compared.astype(np.float64)

	K1 = 0.01
	K2 = 0.03
	L = data_range
	C1 = (K1 * L) ** 2
	C2 = (K2 * L) ** 2

	# gaussian weighted means
	mu1 = gaussian_filter(orig, gaussian_sigma)
	mu2 = gaussian_filter(comp, gaussian_sigma)

	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2

	sigma1_sq = gaussian_filter(orig * orig, gaussian_sigma) - mu1_sq
	sigma2_sq = gaussian_filter(comp * comp, gaussian_sigma) - mu2_sq
	sigma12 = gaussian_filter(orig * comp, gaussian_sigma) - mu1_mu2

	sigma1_sq = sigma1_sq.clip(min=0)
	sigma2_sq = sigma2_sq.clip(min=0)

	num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
	den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
	ssim_map = num / (den + 1e-12)
	return float(np.mean(ssim_map))

def dice_macro(pred_labels: np.ndarray, gt_labels: np.ndarray, labels=None, eps=1e-9):
    """
    Dice cho phân đoạn đa lớp (macro-average).
    - pred_labels, gt_labels: ảnh nhãn (HxW), kiểu int
    - labels: danh sách lớp cần tính; mặc định = hợp các lớp có trong GT/pred
    Trả về: mean Dice ∈ [0..1] (NaN nếu không lớp nào hợp lệ)
    """
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("dice_macro: pred và gt phải cùng kích thước")
    if labels is None:
        labels = np.unique(np.concatenate([np.unique(gt_labels), np.unique(pred_labels)]))
    dices = []
    for c in labels:
        p = (pred_labels == c)
        g = (gt_labels == c)
        denom = p.sum() + g.sum()
        if denom == 0:   # lớp không xuất hiện ở cả hai
            continue
        inter = (p & g).sum()
        dices.append((2.0 * inter + eps) / (denom + eps))
    return float(np.mean(dices)) if len(dices) else np.nan
