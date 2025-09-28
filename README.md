Phân đoạn ảnh xám sử dụng thuật toán MFWOA dựa trên độ đo mờ Fuzzy Entropy

Quick start (PowerShell):

# cài môi trường
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# chạy tối ưu cho ảnh
python -m src.cli.optimize --image path\to\lena.png --K 2 --pop 30 --iters 100

python -m src.ui.app

Kết quả: file JSON ngưỡng và ảnh phân đoạn trong thư mục hiện tại.
