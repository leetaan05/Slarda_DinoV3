# Thay encoder (backbone) bằng DINOv3

Tài liệu: [SLARDA_DINOv3 - English README, source code and setup](SLARDA_DINOv3/README.md) · [Các lệnh Git/GitHub](README_GITHUB_COMMANDS.md).

Đối chiếu CNN 1D của SLARDA với DINOv3 ViT-S/16 trong bản tham chiếu SLARDA_DINOv3. Bài toán vẫn là phân loại đoạn EEG 30 giây thành W, N1, N2, N3 hoặc REM. Thay đổi chính nằm ở cách biểu diễn EEG, nên cả quá trình học và dự đoán đều chịu ảnh hưởng.

## 1. Logic xử lý thay đổi thế nào?

| Thành phần | CNN 1D | DINOv3 + adapter EEG |
| --- | --- | --- |
| Đầu vào | Học trực tiếp từ tín hiệu EEG | EEG → STFT → phổ biên độ log → ảnh 224×224, lặp thành 3 kênh |
| Đặc trưng | Tích chập trên chuỗi thời gian | Backbone đã học từ ảnh, dùng attention giữa các patch của ảnh phổ |
| Kết nối với SLARDA | Đặc trưng 128 kênh, 22 bước | Bỏ CLS/register token; gộp patch theo tần số, chiếu 384→128 chiều và nội suy 14→22 bước |
| Phân loại | Vector 2.816 chiều → 5 logits | Giữ cùng kích thước đầu ra, nhưng đặc trưng và giá trị dự đoán thay đổi |

Adapter giúp giữ giao diện với classifier và bộ phân biệt miền GRU. Tuy nhiên, phổ biên độ bỏ thông tin pha; chuẩn hóa từng đoạn làm thay đổi thông tin về mức biên độ; gộp theo tần số giảm chi tiết phổ. Ba kênh ảnh cùng xuất phát từ một kênh EEG, còn nội suy lên 22 bước không tạo thêm thông tin thời gian.

Luồng thích nghi vẫn là học trên nguồn có nhãn → khởi tạo target và teacher từ nguồn → căn chỉnh miền và học nhãn giả. Phải huấn luyện lại với đặc trưng DINOv3; checkpoint CNN cũ không thể nạp nguyên sang kiến trúc mới. Bản tham chiếu cho phép cập nhật backbone, vì vậy learning rate và bộ nhớ cần được cân nhắc lại.

## 2. Các hàm mất mát thay đổi ra sao?

Mục tiêu của các loss được giữ lại, nhưng đầu vào và gradient thay đổi theo biểu diễn mới:

- CE (cross-entropy) nguồn: vẫn so sánh logits với nhãn thật. Loss cập nhật classifier, lớp chiếu và backbone DINOv3 khi backbone được mở để học.
- CPC ở nhánh tự giám sát: vẫn học dự đoán đặc trưng tương lai bằng loss tương phản, nhưng trên chuỗi đặc trưng ảnh phổ. Bản tham chiếu dùng 10 bước dự đoán cho chuỗi 22 bước. Attention toàn ảnh khiến token sớm có thể chứa thông tin từ thời điểm muộn, nên CPC thấp hơn chưa chứng minh mô hình dự báo tương lai tốt hơn.
- Loss phân biệt miền: vẫn dùng BCEWithLogits, với nguồn = 1 và đích = 0. GRU nhận đặc trưng mới nên độ khó phân biệt hai miền cũng thay đổi.
- Loss căn chỉnh target: vẫn dùng BCEWithLogits để đặc trưng đích được nhận là nguồn. Có thể cần điều chỉnh learning rate và trọng số loss để tránh làm mất thông tin phân lớp.
- CE nhãn giả: vẫn học từ dự đoán teacher vượt ngưỡng tin cậy. DINOv3 làm thay đổi độ tự tin của dự đoán, nên cần kiểm tra lại ngưỡng và trọng số; không có mẫu đạt ngưỡng thì loss này bằng 0.

Trong adaptation, `L_target = L_align + λ · L_pseudo`; loss discriminator được tối ưu riêng. CE nguồn thuộc bước học nguồn; CPC nằm ở nhánh tự giám sát riêng, chưa xác minh việc chuyển checkpoint sang adaptation. Teacher cập nhật bằng trung bình trượt EMA, không có loss riêng. Dùng trọng số DINOv3 đã huấn luyện cũng không yêu cầu thêm các loss tiền huấn luyện ảnh của DINOv3.

## 3. Kết quả có thể thay đổi thế nào?

DINOv3 có thể tận dụng cấu trúc thời gian–tần số trong ảnh phổ, nhưng kiến thức học từ ảnh tự nhiên chưa chắc phù hợp với EEG. Thông tin mất qua adapter cũng có thể làm chất lượng giảm. Vì vậy, chưa thể kết luận DINOv3 tốt hơn CNN chỉ từ việc chạy được mô hình hoặc thấy loss thấp hơn.

Tài liệu tham chiếu mới xác nhận khả năng nạp trọng số và chạy forward/backward; chưa cung cấp kết quả đối chứng đầy đủ trên EEG thật. Cần so sánh trên cùng dữ liệu, chia tập theo người, cùng cặp nguồn–đích và nhiều seed; dùng tập test để đánh giá cuối cùng. Các chỉ số nên gồm accuracy, macro-F1, Cohen’s kappa, F1 từng lớp và chi phí thời gian/bộ nhớ. ViT-S/16 có khoảng 21 triệu tham số, kèm bước tạo ảnh phổ, nên cũng cần đo chi phí thực tế so với CNN.

Tham khảo: README, adapter và các nhánh huấn luyện trong SLARDA_DINOv3; [SLARDA gốc](https://github.com/mohamedr002/SLARDA); [DINOv3 chính thức](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md).
