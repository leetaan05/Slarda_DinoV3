# SleepcareAI · Colab notebooks

Folder [`notebooks/`](notebooks/README.md) chứa notebook DINOv3 + SLARDA, script tạo gói Colab và README tổng. Xem mục **[“Chỉ chỗ đã update DINOv3 so với repo gốc”](notebooks/README.md#dinov3-updates)** để tìm từng file/class/hàm mới, điểm khác với SLARDA gốc và tác dụng của thay đổi.

README đối chiếu với commit SLARDA gốc `c0bf93c09a0958dc857db25c71b51c633b1dff58`, trình bày kiến trúc EEG → spectrogram → DINOv3 → SLARDA, loss/teacher EMA, fine-tuning, dữ liệu QC, checkpoint/resume và phạm vi đã kiểm chứng. Các thay đổi engine dùng chung và sửa đổi legacy được phân biệt với phần kiến trúc DINOv3.

Repository này công bố riêng notebooks và tài liệu đi kèm. Checkpoint DINOv3, dữ liệu Sleep-EDF, ZIP được tạo và folder upload Google Drive được giữ cục bộ. Để tạo gói chạy, cần đặt `notebooks/` trong project `SLARDASSC` với **`Autorgressive_Adaptation/` bản đã thêm `dinov3_experiment/`** và `sleep_edf_tools/`, rồi chuẩn bị dữ liệu/checkpoint theo README. Các mã nguồn engine và báo cáo local được nhắc trong tài liệu chưa nằm trong repository này; clone riêng upstream SLARDA cũng chưa có phần tích hợp DINOv3.
