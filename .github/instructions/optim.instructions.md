# Phạm vi
MFWOA + Fuzzy Entropy.

# Hướng dẫn
- `fuzzy_entropy(img: np.ndarray, thresholds: Sequence[int], membership: str="triangular", **kw) -> float` (thuần, vector hoá).
- `mfwoa_optimize(obj_fn, k_thresholds: int, bounds=(0,255), pop=30, iters=200, seed=42, callback=None) -> (thresholds, best, log)`.
- Ràng buộc: sort + unique ngưỡng; phạt nghiệm vi phạm miền/độ trùng; giải bài **maximize entropy** (hoặc minimize -H).
- Ghi log mỗi vòng: best fitness, thời gian, (tuỳ chọn) khoảng cách hội tụ.
- Test synthetic: histogram đa modal → kiểm tra ngưỡng hội tụ xung quanh các đỉnh (kỳ vọng).
- Tham khảo cơ chế WOA (encircle/spiral/search) và MFO (skill factor, scalar fitness, rmp adaptive) khi hiện thực MFWOA. :contentReference[oaicite:6]{index=6}
- Tham khảo thiết lập Fuzzy Entropy/membership và cách mở rộng nhiều mức. :contentReference[oaicite:7]{index=7}
