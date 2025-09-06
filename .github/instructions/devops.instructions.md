# CI
- Ubuntu-latest; Python 3.10.
- Steps: checkout → setup-python → pip install deps → ruff → black --check → pytest -q → upload-artifact (ảnh demo, CSV/JSON, báo cáo).
- Bật `concurrency` để tránh chạy chồng.

# AWS (CD tuỳ chọn)
- Dùng OIDC: `aws-actions/configure-aws-credentials@v4`.
- Build & Push Docker → ECR → chạy ECS Fargate task (tham số: S3_INPUT, S3_OUTPUT, K, SEED).
- Log ra CloudWatch; output (ảnh/JSON/CSV) lên S3.
