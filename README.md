# Thay encoder EEG bằng DINOv3: điều gì thay đổi?

## Luồng xử lý

Với CNN ban đầu, mô hình học đặc trưng trực tiếp từ tín hiệu EEG. Khi thay bằng DINOv3, mỗi đoạn EEG được chuyển thành ảnh phổ 3 kênh kích thước 224×224. ViT-S/16 đã huấn luyện trước trên ảnh trích xuất patch token; các token được gộp theo trục tần số để tạo chuỗi đặc trưng theo thời gian. Projection đưa đặc trưng 384 chiều về 128 kênh, rồi head dự đoán 5 giai đoạn ngủ.

Vì vậy, phần encoder đổi từ CNN 1D sang ViT xử lý ảnh phổ. Các phần sau vẫn cần giữ đúng giao diện đặc trưng theo chuỗi: bộ phân biệt miền nhận đặc trưng mới, còn classifier và teacher EMA đi qua cùng pipeline. Thường đóng băng phần lớn DINOv3, chỉ học projection, head và tùy chọn mở các block cuối.

## Hàm mất mát

- **Cross-entropy có nhãn nguồn:** giữ nguyên công thức; chỉ logits đầu vào được tạo từ đặc trưng DINOv3.
- **Loss phân biệt/căn chỉnh miền:** vẫn có thể giữ mục tiêu làm đặc trưng nguồn và đích khó phân biệt. Cần đổi cách sắp xếp hoặc chiếu tensor cho phù hợp với chuỗi 128 kênh theo thời gian.
- **Cross-entropy pseudo-label đích:** vẫn dùng nhãn giả từ teacher EMA trên mẫu đủ tự tin. Teacher giờ là bản EMA của mô hình dùng DINOv3; ngưỡng tự tin và trọng số loss có thể cần hiệu chỉnh vì phân bố logits đã đổi.
- **Loss tiền huấn luyện DINOv3:** không cần cộng lại khi dùng checkpoint làm encoder; loss này thuộc giai đoạn tiền huấn luyện ảnh.

Tóm lại, thay backbone không bắt buộc thay công thức các loss của SLARDA; chủ yếu phải nối lại tensor cho đúng và kiểm tra lại độ cân bằng giữa các loss.

## Kết quả dự kiến và giới hạn

Đặc trưng DINOv3 có thể giúp mô hình tận dụng biểu diễn đã học từ ảnh, nhưng ảnh phổ EEG khác ảnh tự nhiên nên không thể kết luận trước rằng độ chính xác sẽ tăng. Cần so sánh với CNN trên cùng dữ liệu, cách chia tập, seed và protocol; báo cáo macro-F1, kappa và kết quả từng lớp. Nếu chưa chạy đánh giá đầy đủ thì không nên xem mức cải thiện là đã được xác nhận.

DINOv3 thường tốn bộ nhớ và thời gian tính toán hơn CNN, đồng thời thêm bước tạo ảnh phổ và yêu cầu checkpoint tương thích. Chất lượng cuối cùng phụ thuộc vào tiền xử lý, số block được fine-tune và việc hiệu chỉnh các loss.
