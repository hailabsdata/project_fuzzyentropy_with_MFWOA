# Phạm vi
Áp dụng cho Tkinter GUI.

# Hướng dẫn
- Tách App (Tk) → MainFrame (ttk.Frame) → các panel: LoadPanel, HistogramPanel, ResultPanel.
- Dùng `PIL.ImageTk` để hiển thị, giữ tỉ lệ, có nút lưu kết quả (mask, ảnh phân đoạn, JSON/CSV).
- Việc tối ưu (MFWOA) chạy thread riêng; cập nhật progress qua `after()`.
- Tạo `render_histogram(canvas, hist, thresholds)` để vẽ histogram + vạch ngưỡng.
- UI không tính entropy; chỉ gọi sang `metrics/`, `optim/`, `seg/`.
- Phím tắt: Ctrl+O (Open), Ctrl+R (Run), Ctrl+S (Save).
