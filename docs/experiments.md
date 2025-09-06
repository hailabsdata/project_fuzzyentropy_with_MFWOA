# Experiments (MFWOA + Fuzzy Entropy)

Hướng dẫn nhanh thực nghiệm:

1. Chuẩn bị folder chứa ảnh xám PNG (ví dụ: `datasets/lena.png`, `datasets/bsds/*.png`).
2. Chạy:

```powershell
python -m src.cli.experiment --input datasets --Ks 2,3,4 --pop 50 --iters 200 --out results
```

Kết quả: một file `results/results.csv` và nhiều file `*_results.json` chứa ngưỡng và điểm số cho mỗi ảnh.

Gợi ý so sánh:
- So sánh mức entropy tối đa thu được với các phương pháp baseline (Otsu đa mức, PSO...).
- Vẽ histogram ảnh cùng với ngưỡng tối ưu để trực quan hoá.
