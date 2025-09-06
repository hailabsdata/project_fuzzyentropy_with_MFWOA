Phân đoạn ảnh xám bằng Multilevel Thresholding tối đa hóa Fuzzy Entropy, giải bằng MFWOA (skeleton project).

Quick start (PowerShell):

# cài môi trường
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# chạy tối ưu cho ảnh
python -m src.cli.optimize --image path\to\lena.png --K 2 --pop 30 --iters 100

Kết quả: file JSON ngưỡng và ảnh phân đoạn trong thư mục hiện tại.
