# Mục tiêu
Viết `src/metrics/fuzzy_entropy.py`:
- Hàm `fuzzy_entropy(img, thresholds, membership="triangular", **kw)` (NumPy vector hoá, ảnh xám uint8).
- Cho phép nhiều membership (triangular/gaussian/S-shaped).
- Sinh 2–3 test trong `tests/test_entropy.py`.

# Ngữ cảnh
#file:../src/metrics/fuzzy_entropy.py
#file:../tests/test_entropy.py
#file:../docs/detailed-design.md

# Lưu ý
Bám công thức fuzzy entropy đa mức và cách nội suy ngưỡng theo giao điểm membership. :contentReference[oaicite:8]{index=8}
