# Antigravity AI Agent Rules & Operational Guidelines

Tài liệu này định nghĩa các quy định và luật làm việc bắt buộc dành cho AI Agent (**Antigravity**) khi làm việc trong dự án.

---

## 1. Đồng bộ Implementation Plan & Report theo Format Slide (16:9 Layout)

1. **Đồng bộ 1-1 (1-to-1 Synchronization)**:
   - File `implementation_plan.md` và File Báo cáo (`report.md` hoặc các file report trong `outputs/**/report/*.md`) phải luôn được đồng bộ 1-1 về mặt nội dung, số liệu, cấu trúc và định hướng.
   - Khi chỉnh sửa hoặc cập nhật `implementation_plan.md`, nội dung tương ứng trong Báo cáo phải được cập nhật sao y 1-1 để đảm bảo tính nhất quán tuyệt đối.
   
2. **Định dạng Layout Slide (16:9 Slide Presentation Format)**:
   - Tất cả các báo cáo và implementation plan phải được trình bày theo định dạng **Slide 16:9 Markdown**.
   - Phân chia các slide rõ ràng bằng đường gạch ngang (`---`).
   - Mỗi slide bao gồm tiêu đề rõ ràng, bố cục cân đối, sử dụng bảng biểu, Mermaid diagram, thẻ ghi chú (`> [!NOTE]`, `> [!IMPORTANT]`, `> [!TIP]`) tối ưu cho trình bày slide và báo cáo kỹ thuật.

---

## 2. Cam kết Mã Nguồn Thật & Kết Quả Thực Nghiệm Thật (Zero Hallucination Policy)

1. **Tuyệt đối KHÔNG tạo ra mã code ảo (No Fake / Mock Code)**:
   - Tất cả mã nguồn được viết hoặc chỉnh sửa phải là mã hoạt động thực tế, có logic hoàn chỉnh, không sinh mã ảo, không để hàm trống (`pass`), không dùng dữ liệu giả lập hoặc placeholder chưa hoàn thiện.
   
2. **Tuyệt đối KHÔNG tạo ra kết quả / số liệu ảo (No Fake / Hallucinated Results)**:
   - Tất cả dữ liệu, số liệu đo đạc (PSNR, SSIM, BD-Rate, Inference Time, HOTA, MOTA, BBox, v.v.) xuất hiện trong Báo cáo và Implementation Plan phải xuất phát từ việc thực thi thực tế các script và trích xuất từ log/output thật.
   - Nghiêm cấm việc bịa đặt, tự suy đoán số liệu hoặc đưa ra báo cáo không có thực chứng từ chạy code.

---

## 3. Cấu Trúc Code & Quy Chuẩn Lập Trình Standard (Coding Standards & Architecture)

1. **Tuân thủ Quy chuẩn Ngôn ngữ (Language Standard Conventions)**:
   - **Python**: Tuân thủ nghiêm ngặt **PEP 8**, bắt buộc sử dụng **Type Hints** (`typing`), Docstrings chuẩn (Google/Sphinx style), sắp xếp import theo đúng thứ tự (Standard Library $\rightarrow$ 3rd-party $\rightarrow$ Local modules).
   - **C++/JavaScript/Khác**: Tuân thủ chuẩn style guide chính thức của từng ngôn ngữ.

2. **Cấu trúc Mã nguồn & Architecture**:
   - Thiết kế mô-đun hóa (Modular Architecture), tách biệt giữa logic xử lý (`src/`), kịch bản thực thi (`scripts/`), cấu hình (`configs/`) và đầu ra (`outputs/`).
   - Xử lý ngoại lệ chặt chẽ (`try-except`), không viết `except: pass` mờ hóa lỗi.
   - Đảm bảo quản lý tài nguyên hệ thống: tối ưu GPU VRAM (`torch.no_grad()`, `torch.cuda.empty_cache()`), tránh tràn RAM bằng generator/streamer khi xử lý video/dataset lớn.

---
