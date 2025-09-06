# Mục tiêu
Xây dựng hệ thống phân đoạn ảnh xám bằng ngưỡng hoá đa cấp (multilevel thresholding), **tối đa hoá Fuzzy Entropy** làm hàm mục tiêu, dùng **MFWOA** làm bộ giải tối ưu. Có **GUI Tkinter** để demo (Lena, BSDS), hiển thị histogram, ngưỡng tối ưu, mask và ảnh kết quả.

# Ngôn ngữ & Thư viện
- Python 3.10+, OpenCV, NumPy, SciPy; (tùy chọn: matplotlib để vẽ).
- Tkinter cho GUI; PyTest cho kiểm thử; ruff + black cho style.
- DevOps: GitHub Actions (CI/CD), AWS (S3, ECR, ECS Fargate, CloudWatch).

# Ràng buộc thuật toán
- **Fuzzy Entropy**: là hàm mục tiêu cho phân ngưỡng đa cấp; cho phép thay đổi hàm membership (triangular/gaussian/S-shaped…). (Cơ sở Fuzzy Entropy và thiết lập đa mức: xem tài liệu đã đính kèm). :contentReference[oaicite:2]{index=2}
- **MFWOA (Multifactorial Whale Optimization Algorithm)**: dùng cơ chế WOA + chiến lược **chia sẻ tri thức liên-nhiệm vụ** (knowledge transfer) và **rmp tự thích nghi** để tối ưu đồng thời nhiều tác vụ (số ngưỡng khác nhau). Tôn trọng ràng buộc: ngưỡng trong [0..255], tăng dần, loại trùng. :contentReference[oaicite:3]{index=3}

# Kiến trúc đề xuất
src/
├─ io/ # đọc/ghi ảnh, histogram
├─ metrics/ # fuzzy_entropy.py (biến thể membership)
├─ seg/ # thresholding.py (áp ngưỡng đa cấp, hậu xử lý)
├─ optim/ # mfwoa.py (bộ giải); utils WOA/MFEA-style
├─ ui/ # Tkinter app, view histogram/ảnh/ngưỡng
├─ cli/ # optimize.py, eval.py (thực nghiệm, so sánh)
└─ utils/ # seed, timer, viz
tests/
docs/
.github/
# Quy ước code
- PEP8 + type hints + docstring; không hard-code đường dẫn; dùng `pathlib`.
- Mọi hàm tính toán (entropy, cập nhật vị trí, ràng buộc ngưỡng) phải **thuần (pure)** và có test.
- Dùng RNG có seed (`numpy.random.default_rng`) để tái lập thí nghiệm.
- Ưu tiên vector hoá NumPy cho histogram/entropy, hạn chế vòng lặp Python.

# GUI (Tkinter)
- Chức năng: mở ảnh → xem histogram → chọn K (số ngưỡng) → chạy MFWOA → vẽ ngưỡng và xem kết quả.
- Tránh block UI: tối ưu chạy trong thread phụ, cập nhật tiến độ bằng `after()`.
- Cho phép lưu ảnh mask, ảnh phân đoạn, JSON ngưỡng, CSV kết quả.

# DevOps
- CI: ruff, black --check, pytest, upload artefacts (ảnh kết quả/CSV/JSON).
- CD (tùy chọn): build Docker → push ECR → chạy batch trên ECS Fargate; đọc đầu vào từ S3, ghi đầu ra về S3. Dùng OIDC (`aws-actions/configure-aws-credentials`).
- Secrets: `AWS_ROLE_ARN`, `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REPO`.

# Tài liệu & Hướng dẫn học thuật (bắt buộc)
1) **Ngưỡng hoá đa cấp + Fuzzy Entropy**: nguyên lý, membership, công thức entropy tổng (H) và cách suy ra ngưỡng từ tham số membership. :contentReference[oaicite:4]{index=4}  
2) **MFWOA**: cơ chế WOA (bao vây, tấn công xoắn ốc, khám phá), chia sẻ tri thức đa nhiệm + rmp adaptive, quy trình tổng thể MFWOA. :contentReference[oaicite:5]{index=5}  
> Luôn bám sát `docs/detailed-design.md` và cập nhật `docs/experiments.md` sau mỗi thực nghiệm.

# Chuẩn PR & Review
- Conventional Commits; PR cần: mô tả, ảnh minh hoạ UI/kết quả, link artefacts CI.
- Không merge khi lint/test fail.
- **Copilot/Chat phải trả lời bằng tiếng Việt.**