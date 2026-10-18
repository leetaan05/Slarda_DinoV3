# Thay encoder (backbone) EEG bằng DINOv3: điều gì thay đổi?

Phân tích đối chiếu SLARDA dùng CNN 1D với biến thể DINOv3 ViT-S/16 trong tài liệu và mã tham chiếu `SLARDA_DINOv3`. Bài toán vẫn là phân loại mỗi đoạn EEG 30 giây thành **W, N1, N2, N3 hoặc REM**; repo này chỉ giữ tài liệu so sánh.

## 1. Logic xử lý

| Thành phần | CNN 1D ban đầu | DINOv3 + adapter EEG |
| --- | --- | --- |
| Đầu vào encoder | Tín hiệu EEG trực tiếp `[B,1,3000]` | EEG → STFT → phổ biên độ log → ảnh 224×224, lặp thành 3 kênh |
| Cách học đặc trưng | Tích chập trên chuỗi thời gian | ViT đã học từ ảnh, xử lý các patch của ảnh phổ |
| Đặc trưng đưa vào SLARDA | Chuỗi `[B,128,22]` ở nhánh adaptation | Bỏ CLS/register token; lưới patch 14×14×384 → trung bình theo tần số → chiếu 384→128 → nội suy 14→22 bước |
| Đầu ra phân loại | 5 logits từ vector 2.816 chiều | Giữ cùng kích thước và cùng 5 lớp, nhưng giá trị dự đoán thay đổi |

Adapter giúp giữ giao diện cho classifier và GRU phân biệt miền, **không giữ nguyên ý nghĩa đặc trưng**. Phổ biên độ bỏ thông tin pha; chuẩn hóa từng đoạn làm mất mức biên độ tuyệt đối; gộp theo tần số làm giảm độ phân giải phổ. Ba kênh ảnh là bản lặp của một phổ, không phải ba kênh EEG; nội suy lên 22 bước không tạo thêm thông tin thời gian.

Luồng thích nghi vẫn là: huấn luyện nguồn có nhãn → khởi tạo target và teacher từ nguồn → căn chỉnh miền và học nhãn giả. Phải huấn luyện lại các tầng phía sau với đặc trưng mới; checkpoint CNN không thể nạp nguyên sang DINOv3. Bản tham chiếu cho phép cập nhật backbone; đóng băng backbone là một lựa chọn thử nghiệm khác.

## 2. Các hàm mất mát

| Loss | Công thức/mục tiêu có đổi không? | Khác biệt khi dùng DINOv3 |
| --- | --- | --- |
| **CE nguồn** | Giữ cross-entropy giữa logits và nhãn thật khi huấn luyện nguồn | Gradient đi qua classifier, projection và phần backbone được mở để học. |
| **CPC tự giám sát** | Giữ loss tương phản: dự đoán đặc trưng tương lai, phân biệt mẫu đúng với mẫu khác trong batch | Dự đoán trên chuỗi đặc trưng ảnh phổ; cần chỉnh số bước dự đoán theo độ dài chuỗi. ViT attention toàn ảnh khiến token sớm có thể chứa thông tin thời điểm muộn, nên CPC giảm chưa chứng minh khả năng dự báo tương lai tốt hơn. |
| **Discriminator** | Giữ BCEWithLogits: nguồn = 1, đích = 0 | GRU vẫn nhận chuỗi, nhưng phân phối đặc trưng mới làm bài toán phân biệt miền thay đổi. |
| **Căn chỉnh target** | Giữ BCEWithLogits, yêu cầu đặc trưng đích được nhận là nguồn (= 1) | Cập nhật encoder target để đánh lừa discriminator; có thể cần chỉnh learning rate và cân bằng với loss nhãn giả. |
| **CE nhãn giả** | Giữ CE trên mẫu target có độ tin cậy teacher vượt ngưỡng | Teacher EMA dùng cùng kiến trúc DINOv3. Độ tự tin thay đổi nên cần hiệu chỉnh ngưỡng/trọng số; batch không có mẫu đạt ngưỡng thì loss này bằng 0. |

Trong vòng adaptation của bản tham chiếu, **`L_target = L_align + λ · L_pseudo`**; discriminator được tối ưu riêng. CE nguồn thuộc bước huấn luyện nguồn, CPC thuộc nhánh tự giám sát, không mặc nhiên cộng tất cả vào một loss. Teacher cập nhật bằng trung bình trượt (EMA), không có loss riêng. Dùng checkpoint DINOv3 cũng không yêu cầu cộng lại các loss tiền huấn luyện ảnh của DINOv3.

## 3. Kết quả sẽ thay đổi thế nào?

**Chưa có cơ sở kết luận DINOv3 tốt hơn CNN trên EEG.** Biểu diễn ảnh có thể giúp nhận dạng cấu trúc thời gian–tần số, nhưng khác biệt giữa ảnh tự nhiên và phổ EEG, cùng thông tin bị mất ở adapter, cũng có thể làm chất lượng giảm. Giữ đúng shape hoặc chạy được forward/backward chỉ xác nhận khả năng kết nối, không xác nhận độ chính xác. Loss miền/CPC thấp hơn cũng không đồng nghĩa phân loại tốt hơn.

Cần so sánh hai mô hình trên cùng dữ liệu, chia tập theo người, cùng cặp source→target và nhiều seed; chỉ dùng nhãn target test để đánh giá. Báo cáo accuracy, macro-F1, Cohen’s kappa, F1 từng lớp và thời gian/bộ nhớ. ViT-S/16 có khoảng 21 triệu tham số; biến thể này còn thêm STFT và xử lý ảnh nên cần đo chi phí thực tế khi so với CNN.

Tham khảo: [SLARDA gốc](https://github.com/mohamedr002/SLARDA), [DINOv3 chính thức](https://github.com/facebookresearch/dinov3), [giao diện DINOv3 trong Transformers](https://huggingface.co/docs/transformers/model_doc/dinov3). Chi tiết adapter và loss ở trên được đối chiếu với thư mục `SLARDA_DINOv3` tại workspace, không phải kết quả thực nghiệm mới.
