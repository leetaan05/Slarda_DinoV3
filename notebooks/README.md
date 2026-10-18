# Notebooks · Những cập nhật DINOv3 so với repo SLARDA gốc

Cập nhật **18/10/2026**. README này giải thích **đã thêm/sửa DINOv3 ở đâu, khác repo gốc như thế nào, phần thuật toán nào được giữ, và cách chạy notebook tương ứng**. Notebook chính: [`colab/train_sleep_dinov3_colab.ipynb`](colab/train_sleep_dinov3_colab.ipynb). Mọi lệnh bên dưới chạy từ **thư mục gốc `SLARDASSC`**, trừ khi ghi rõ khác.

Mục lục:

- [Mốc repo gốc và phạm vi đối chiếu](#baseline)
- [Chỉ chỗ đã update DINOv3 so với repo gốc](#dinov3-updates)
- [Kiến trúc EEG → DINOv3 → SLARDA](#architecture)
- [Loss, tối ưu và các khác biệt protocol](#training)
- [Dữ liệu, checkpoint và kiểm chứng](#evidence)
- [Tổ chức folder và cách chạy](#usage)

<a id="baseline"></a>

## 1. Repo gốc nào được dùng để so sánh?

Mốc đối chiếu là [mohamedr002/SLARDA tại commit `c0bf93c`](https://github.com/mohamedr002/SLARDA/tree/c0bf93c09a0958dc857db25c71b51c633b1dff58), SHA đầy đủ `c0bf93c09a0958dc857db25c71b51c633b1dff58` — commit nền của checkout `SLARDASSC` đang dùng. Nội dung đã được kiểm tra bằng `git show`, `git diff` và đọc các file mới trong workspace; đây không phải khẳng định về nhánh mới nhất của upstream tại mọi thời điểm.

Repo gốc có CNN EEG `EEG_M_SL`, bộ phân biệt miền tự hồi quy và teacher EMA. Trong cây file của commit trên **chưa có** `dinov3_experiment/`, folder `notebooks/` này, hoặc pipeline `sleep_edf_tools/` hiện tại. Bản cập nhật thêm nhánh DINOv3 cho EEG và một engine chung để CNN/DINOv3 dùng cùng cách huấn luyện.

Các file gốc để đọc đối chiếu:

| Thành phần | File tại commit gốc |
|---|---|
| CNN EEG và discriminator | [`Autorgressive_Adaptation/models/models.py`](https://github.com/mohamedr002/SLARDA/blob/c0bf93c09a0958dc857db25c71b51c633b1dff58/Autorgressive_Adaptation/models/models.py) |
| Điều phối huấn luyện | [`Autorgressive_Adaptation/train_CD.py`](https://github.com/mohamedr002/SLARDA/blob/c0bf93c09a0958dc857db25c71b51c633b1dff58/Autorgressive_Adaptation/train_CD.py) |
| SLARDA và teacher EMA | [`Autorgressive_Adaptation/trainer/SLARDA.py`](https://github.com/mohamedr002/SLARDA/blob/c0bf93c09a0958dc857db25c71b51c633b1dff58/Autorgressive_Adaptation/trainer/SLARDA.py) |
| Huấn luyện source | [`Autorgressive_Adaptation/trainer/source_only.py`](https://github.com/mohamedr002/SLARDA/blob/c0bf93c09a0958dc857db25c71b51c633b1dff58/Autorgressive_Adaptation/trainer/source_only.py) |
| Cấu hình EEG | [`Autorgressive_Adaptation/config_files/EEG_Configs.py`](https://github.com/mohamedr002/SLARDA/blob/c0bf93c09a0958dc857db25c71b51c633b1dff58/Autorgressive_Adaptation/config_files/EEG_Configs.py) |

**Phạm vi GitHub:** repository `SleepcareAI` hiện công bố folder `notebooks/` cùng README ở gốc. Các đường dẫn `Autorgressive_Adaptation/...`, `sleep_edf_tools/...`, `tests/...`, `docs/...`, `reports/...` dưới đây là vị trí trong **project local `SLARDASSC` đã được cập nhật**, không phải các file đã được đưa lên repository chỉ chứa notebooks. Tải riêng notebooks hoặc clone nguyên bản upstream sẽ chưa có engine DINOv3 mới này.

<a id="dinov3-updates"></a>

## 2. Chỉ chỗ đã update DINOv3 so với repo gốc

### 2.1. Mã nguồn tích hợp DINOv3

Trong bảng này, `DINO/` viết tắt cho `Autorgressive_Adaptation/dinov3_experiment/`. Số dòng là vị trí tại lần đối chiếu ngày 18/10/2026; tên class/hàm giúp tìm lại nếu file thay đổi.

| Vị trí đã thêm/sửa | Điểm vào chính | Repo gốc | Bản hiện tại và tác dụng |
|---|---|---|---|
| `Autorgressive_Adaptation/train_CD.py:1` | Nhánh kiểm tra `--eeg-backbone` | Đi vào luồng trainer legacy | Có `--eeg-backbone cnn` hoặc `dinov3` thì chuyển sang engine mới trước khi import/parse tham số legacy. Không có cờ này thì vẫn vào luồng legacy hiện tại. |
| `DINO/train_dinov3.py:8` — mới | `main()` | Chưa có CLI DINOv3 | Thêm lựa chọn backbone, backend, checkpoint, số block mở học, source/adapt epochs, resume, giới hạn batch và kiểm tra cấu hình. |
| `DINO/model.py:13` — mới | `EEGSpectrogram` | CNN xử lý trực tiếp EEG 1D | Chuyển EEG thành ảnh phổ 3 kênh 224×224, chuẩn hóa để đưa vào ViT. |
| `DINO/model.py:35` — mới | `DinoEEGBackbone` | Chưa có pretrained ViT | Nạp DINOv3 ViT-S/16, lấy patch token, gộp trục tần số và giữ thứ tự thời gian. |
| `DINO/model.py:66` — mới | `set_trainable_blocks()` | Chưa có chính sách fine-tune ViT | Đóng băng backbone, mở tùy chọn các block cuối và final norm để học. |
| `DINO/model.py:84` — mới | `DinoEEGBackbone.forward()` | Chưa có đường truyền DINO | Loại prefix token, kiểm tra đủ 196 patch token; tôn trọng `torch.no_grad()` của caller. |
| `DINO/model.py:98` — mới | `DinoEEGHead` | Classifier nhận đặc trưng CNN | Chiếu đặc trưng DINO 384 chiều về 128 kênh, đưa về 22 bước và phân loại 5 giai đoạn ngủ. |
| `DINO/model.py:113` — mới | `DomainDiscriminator` | `Discriminator_AR` trong `models/models.py` | GRU 128 → 512 → 1; dùng `transpose(1, 2)` để chuyển đúng trục kênh/thời gian. |
| `DINO/model.py:124` — mới | `DinoEEGModel` | `EEG_M_SL` trả logits và đặc trưng | Ghép backbone/head nhưng giữ giao diện `(logits, (vector, features))` tương thích với ý tưởng SLARDA. |
| `DINO/model.py:136` — mới | `CNNBaseline` | CNN gốc nằm trong mô hình legacy | Bọc đường dự đoán CNN từ `predict_sleep_stage.EEG_M_SL_Inference`, bỏ phần CPC không dùng, để chạy đối chứng bằng cùng engine. |
| `DINO/losses.py` — mới | `discriminator_domain()`, `target_alignment()`, `confident_teacher_pseudo()` | Loss nằm trong trainer legacy | Tách loss miền, target alignment và pseudo-label CE; xử lý batch không có pseudo-label đủ tin cậy. |
| `DINO/engine.py:102` — mới | `adaptation_step()` | Cập nhật trong `trainer/SLARDA.py` | Tách bước discriminator và target; quản lý detach, gradient, chế độ eval và teacher EMA rõ ràng. |
| `DINO/engine.py:141` — mới | `train()` | Luồng legacy riêng | Dùng chung source training, adaptation, validation, export và resume cho CNN/DINOv3. |
| `DINO/preflight.py:15` — mới | `main()` | Chưa có kiểm tra DINO pretrained | Kiểm tra checkpoint, shape và gradient source/target trước khi train dài. |
| `DINO/inference.py:11` — mới | `load_export()` | Chưa có export DINO dạng v2 | Dựng lại đúng backend, xác minh checkpoint khởi tạo và nạp strict source/target weights. |
| `DINO/inference.py:41` — mới | `main()` | Chưa có CLI đánh giá DINO này | Đánh giá held-out test hoặc xuất JSON dự đoán từ EEG NPY. |

`DINO/losses.py` còn có hàm `source_classification()` biểu diễn source CE; vòng source thực tế trong `engine.train()` gọi trực tiếp `torch.nn.functional.cross_entropy`. Hai cách cùng biểu diễn cross-entropy trên nhãn source.

### 2.2. Các file notebooks mới phục vụ nhánh DINOv3

| File trong folder này | Phần cập nhật so với commit gốc |
|---|---|
| [`colab/train_sleep_dinov3_colab.ipynb`](colab/train_sleep_dinov3_colab.ipynb) | Notebook Colab mới: kiểm tra GPU, mount Drive, xác minh checksum, chuẩn bị runtime, audit fit, preflight, train/resume, đánh giá và xuất dự đoán. |
| [`scripts/build_notebook.py`](scripts/build_notebook.py) | Nguồn tạo notebook gồm các cell markdown/code; kiểm tra cú pháp code cell khi sinh file. Sửa nội dung notebook lâu dài tại script này rồi tạo lại `.ipynb`. |
| [`scripts/package_colab.py`](scripts/package_colab.py) | Đóng gói `train_CD.py`, `predict_sleep_stage.py`, các module DINO và công cụ Sleep-EDF thành ZIP kèm `bundle_manifest.json`. |
| [`scripts/prepare_colab_upload.py`](scripts/prepare_colab_upload.py) | Xác minh checkpoint, gom ZIP/notebook/weights/dữ liệu QC và ghi SHA256, kích thước từng file vào `UPLOAD_MANIFEST.json`. |
| [`requirements-colab.txt`](requirements-colab.txt) | Khai báo `timm==1.0.22`, `safetensors==0.5.3`, MNE và Matplotlib; notebook giữ torch/torchvision CUDA có sẵn trong runtime. |

### 2.3. Thay đổi hỗ trợ đã có trong workspace, không phải riêng kiến trúc DINOv3

Những thay đổi sau cũng khác commit gốc, nhưng cần ghi nhận riêng khi giải thích hoặc so sánh thí nghiệm:

| File / thư mục | Khác biệt đã đối chiếu |
|---|---|
| `Autorgressive_Adaptation/args.py` | Thêm `--checkpoint_tag` để phân biệt checkpoint của các cohort. |
| `Autorgressive_Adaptation/config_files/EEG_Configs.py` | Batch legacy trong `Config` đổi từ 512 thành 64. Đây là cấu hình legacy, không phải batch 8 của notebook DINOv3. |
| `Autorgressive_Adaptation/train_CD.py` | Ngoài nhánh dispatch DINO, luồng EEG legacy giới hạn hướng SC (`a`) → ST (`b`) và bổ sung export source/target kèm metadata. |
| `Autorgressive_Adaptation/trainer/source_only.py` | Gắn tag vào tên checkpoint source. |
| `Autorgressive_Adaptation/trainer/SLARDA.py` | Dùng checkpoint tag, xử lý tập pseudo-label rỗng và bỏ đánh giá dựa trên nhãn target validation đã được ẩn. |
| `Autorgressive_Adaptation/trainer/training_evaluation.py` | Tính source/target test accuracy từ toàn bộ dự đoán từng epoch, tránh lệch do trung bình accuracy các batch kích thước khác nhau. |
| `sleep_edf_tools/` | Thêm tải/chọn bản ghi, chuẩn bị EEG, ẩn nhãn target train/val, QC, provenance và audit. |
| `tests/test_eeg_pipeline.py` | Có 6 test về gradient/EMA, pseudo-label rỗng, nhãn target, metric, chuẩn hóa và giao diện DINO; test DINO dùng ViT giả. |

Các khác biệt trên được mô tả theo trạng thái code hiện có; lần sửa README này không tạo ra hay thay đổi các thuật toán đó.

<a id="architecture"></a>

## 3. Kiến trúc thay đổi như thế nào?

Luồng CNN gốc: EEG `[B, 1, 3000]` → các lớp Conv1d/BatchNorm/activation/pooling → đặc trưng `[B, 128, 22]` → classifier 5 lớp. Nhánh DINOv3 thay phần trích đặc trưng bằng luồng sau:

```text
EEG Fpz-Cz, 100 Hz, một epoch 30 giây
  [B, 1, 3000] đã z-score từng epoch
        ↓ EEGSpectrogram
  STFT: Hann window 256, n_fft=256, hop_length=64
  log1p(|STFT|), min-max riêng từng ảnh
  resize bilinear 224×224, lặp 3 kênh, chuẩn hóa ImageNet
        ↓ DinoEEGBackbone
  DINOv3 ViT-S/16 pretrained
  bỏ CLS/register token → 196 patch token, mỗi token 384 chiều
  reshape 14×14, lấy trung bình theo trục tần số
        ↓
  [B, 384, 14]
        ↓ DinoEEGHead
  Conv1d(384→128, kernel=1) + GELU + AdaptiveAvgPool1d(22)
        ↓
  features [B, 128, 22]
        ├── flatten [B, 2816] → Linear(2816→5) → logits [B, 5]
        └── transpose → [B, 22, 128] → GRU(128→512) → Linear(512→1)
```

Năm lớp theo thứ tự `W, N1, N2, N3, REM`. Pool từ 14 lên 22 bước nhằm khớp giao diện đặc trưng; thao tác này không tạo thêm độ phân giải thời gian thực của tín hiệu. GRU ở đây xử lý chuỗi đặc trưng **bên trong một epoch EEG**, không phải chuỗi nhiều epoch ngủ liên tiếp.

`UNFREEZE_LAST_BLOCKS=0` đóng băng ViT, chỉ học projection và classifier. Notebook mặc định `1`: học block cuối, final norm, projection và classifier. Báo cáo preflight cục bộ ghi tổng **21.650.309** tham số của mô hình DINO EEG, trong đó **1.838.213** trainable khi mở một block; không tính discriminator riêng. CLI `train_dinov3.py` mặc định `--unfreeze-last-blocks 0`, nên lệnh local muốn giống notebook cần truyền rõ `--unfreeze-last-blocks 1`.

### Nạp pretrained và backend

- Notebook chọn `timm`, tạo `vit_small_patch16_dinov3` với `pretrained=False` để không tự tải weights, sau đó nạp toàn bộ file safetensors bằng `strict=True`. Không có fallback sang backbone ngẫu nhiên khi nạp lỗi.
- `auto` chọn `timm` nếu đuôi là `.safetensors`, còn lại chọn `meta`. `meta` cần repo DINOv3 local và checkpoint phù hợp với loader của Meta. Đổi đuôi file không chuyển đổi định dạng trọng số.
- SHA256 checkpoint khởi tạo được lưu cùng cấu hình export; inference yêu cầu đúng checkpoint để dựng mô hình rồi nạp source/target state. Vì vậy export có full trained weights nhưng loader hiện tại vẫn cần file pretrained ban đầu ở đường dẫn hợp lệ.
- Đây là **tích hợp và fine-tune DINOv3 pretrained** cho EEG. Chưa bổ sung pretraining DINOv3 từ đầu hoặc giai đoạn CPC từ `Self-supervised training/` vào nhánh mới.

<a id="training"></a>

## 4. Phần nào giữ SLARDA, phần nào đã sửa?

### 4.1. Loss và trình tự tối ưu

1. **Source stage:** dùng nhãn SC để học cross-entropy. Mặc định chọn source của epoch cuối (`source-selection=last`); `best` chọn theo accuracy SC validation và là lựa chọn protocol khác.
2. **Khởi tạo adaptation:** copy source đã học sang target và teacher; source/teacher được đóng băng.
3. **Discriminator:** nhận source/target features đã `detach`; BCE gán source = 1, target = 0; chỉ cập nhật optimizer discriminator ở bước này.
4. **Target:** khóa tham số discriminator, truyền gradient qua nó về target; tối ưu adversarial BCE để target có nhãn miền source, cộng `0.1 ×` CE với pseudo-label teacher có confidence **> 0.9**.
5. **Teacher:** cập nhật tham số `teacher = 0.996 × teacher + 0.004 × target`; copy các buffer từ target. Target train, source và teacher eval trong adaptation.

Batch không có pseudo-label vượt ngưỡng trả pseudo-loss bằng 0 nhưng vẫn có graph hợp lệ. Preflight cố ý dùng confidence = 0 để kiểm tra đầy đủ đường gradient; đây không phải ngưỡng dùng trong lượt train mặc định.

### 4.2. Những khác biệt phải ghi khi so sánh kết quả

| Điểm đối chiếu | Commit gốc | Engine CNN/DINOv3 hiện tại | Hệ quả |
|---|---|---|---|
| Trục input GRU | `Discriminator_AR` dùng `view(B, -1, C)` | `features.transpose(1, 2)` | Giữ đúng thứ tự thời gian thay vì diễn giải lại vùng nhớ thành trục khác. |
| Bước optimizer discriminator | Có thêm `optimizer_disc.step()` sau backward target | Chỉ step ở bước domain classification; khóa tham số khi học target | Tách rõ hai mục tiêu tối ưu. |
| Source/teacher khi adaptation | Có khóa gradient nhưng không gọi eval rõ ràng tại bước adaptation | Gọi eval và đóng băng | Tránh dropout/BN thay đổi ngoài ý định. |
| Teacher EMA | Cập nhật parameters | EMA parameters và copy buffers | Đồng bộ BN statistics cho nhánh CNN. |
| Pseudo-label rỗng | Gọi CE trên tập confident | Zero loss khả vi khi tập rỗng | Tránh loss không hợp lệ ở batch đầu/nhỏ. |
| Batch và batch cuối | Cấu hình EEG gốc 512; loader train/val `drop_last=True` | Mặc định batch 8, `drop_last=False` | Số batch và lượng dữ liệu mỗi epoch khác; cần đồng nhất khi đối chứng. |
| Validation/test | Trainer gốc có target validation theo nhãn | Engine fit đọc `train_a`, `val_a`, `train_b`; không mở target val/test | Không dùng nhãn target val/test để chọn model khi fit. |
| Metric | `model_evaluate` legacy trung bình theo batch | Confusion matrix trên từng epoch; loss cộng theo số mẫu | Accuracy/loss không bị lệch bởi batch cuối; có macro-F1, kappa và metric 5 lớp. |
| Ghép hai miền | `zip(source_loader, target_loader)` | Vẫn dùng `zip` | Mỗi adaptation epoch dừng theo loader ngắn hơn. |

Các sửa engine này áp dụng cho cả `--eeg-backbone cnn` và `dinov3`. Do đó, để đánh giá tác động của DINOv3, cần chạy CNN và DINO bằng **cùng engine, dữ liệu QC, split, batch, seed, số epoch, loss weights và quy tắc chọn source**, rồi báo riêng thay đổi backbone. Không gán mọi chênh lệch accuracy so với kết quả legacy cho DINOv3.

Các thông số mặc định được giữ theo cấu hình SLARDA EEG: Adam betas `(0.5, 0.99)`; source LR `3e-4`; target/discriminator LR `1e-3`, weight decay `3e-4`; StepLR của target mỗi 10 adaptation epoch với gamma `0.1`; source 25 epoch, adaptation 15 epoch. Nhánh mới hiện dùng discriminator GRU cố định; lựa chọn ATT của legacy chưa được tích hợp vào engine mới.

<a id="evidence"></a>

## 5. Dữ liệu, checkpoint và bằng chứng kiểm chứng

### 5.1. Dữ liệu phục vụ thí nghiệm DINOv3

`sleep_edf_tools/prepare_sleep_edf.py:18`, hàm `read_record()`, đọc Fpz-Cz 100 Hz thành cửa sổ 3.000 mẫu. SC dùng nhãn 5 lớp; ST train/val tạo cửa sổ liên tiếp từ PSG và gán `-1`, không mở hypnogram để lọc/cắt theo nhãn. QC bỏ epoch có std dưới `1e-8` volt bằng cùng quy tắc ở các miền/split, tính moments float64 và lưu vị trí mẫu ban đầu trong provenance.

Notebook dùng `sleep_edf_tools/audit_data.py --fit-only` trước train: kiểm tra shape, hữu hạn, nhãn, chất lượng và subject leakage của các split phục vụ fit; không mở test tensor trong bước audit này. File test vẫn nằm trong gói upload và có thể được copy/kiểm tra checksum trước fit; việc này không dùng nhãn test để tối ưu hoặc chọn model. Đánh giá có nhãn `test_b` chỉ chạy sau train ở chế độ full.

`medium_qc` có số epoch SC train/val/test = **32.448 / 7.927 / 8.271**, ST = **6.089 / 1.955 / 2.008**, theo báo cáo audit local. Kết quả legacy trên ST test **2.025** epoch không cùng tập test với QC **2.008** epoch, nên không so sánh trực tiếp như một protocol duy nhất.

### 5.2. Checkpoint và resume đã bổ sung

`DINO/engine.py` thêm `atomic_save()`, `train()` và hàm nội bộ `save_resume()`:

- `model.pt`: export v2 gồm full source/target state, architecture, backend, SHA256 pretrained/data, class names, metadata EEG, history và cờ `smoke_test_only`.
- `model.resume.pt`: thêm teacher/discriminator, toàn bộ optimizer, scheduler, stage/epoch, best source state và RNG Python/NumPy/PyTorch CPU/CUDA. Lưu sau mỗi epoch, nên epoch đang dở được chạy lại sau ngắt.
- Resume kiểm tra các cấu hình như backbone/backend, số block mở học, batch, source epochs/selection, loss weights, seed, SHA256 dữ liệu và weights. Checkpoint được nạp về CPU trước khi khôi phục; CUDA RNG state được chuyển về CPU khi restore.
- CLI chặn ghi đè export nếu chưa chọn resume hoặc `--allow-overwrite`; notebook yêu cầu đổi `RUN_LABEL` khi chạy mới. Không dùng resume để biến smoke run thành full run.

### 5.3. Đã kiểm chứng đến đâu?

| Nội dung | Bằng chứng local đã có | Phạm vi kết luận |
|---|---|---|
| Pretrained thật + forward/backward | `reports/dinov3_preflight.json`: `passed`, RTX 3050 Laptop, logits `[2,5]`, features `[2,128,22]` | Backbone nạp được và tương thích đầu vào EEG. |
| Partial fine-tuning | Báo cáo trên: source/target block cuối có gradient; block đầu, source/teacher trong adaptation được đóng băng | Đường gradient hoạt động ở cấu hình preflight. |
| Smoke train/resume/inference | `docs/DINOV3_COLAB_AUDIT.md`, `docs/dinov3_colab_setup_validation.json` và `reports/dinov3_gpu_smoke_predictions.json` | Báo cáo đã ghi nhận 1+1 epoch giới hạn 2 batch, resume CUDA và xuất dự đoán; không phải thí nghiệm full. |
| Unit test | `tests/test_eeg_pipeline.py`, báo cáo audit ghi 6 test qua | Có kiểm tra contract DINO bằng ViT giả; không thay thế kiểm chứng pretrained thật. |
| Notebook/gói upload | Báo cáo setup: 20 cell, 9 code cell compile, 17 file upload và 13 file code bundle | Kiểm tra cấu trúc, cú pháp và checksum. |
| Train đầy đủ trên Colab | `colab_kernel_executed=false` trong báo cáo setup | Chưa có kết quả full 25+15 epoch hoặc bằng chứng DINOv3 tốt hơn CNN. |

Các báo cáo trên có trước lần cập nhật README này và được đọc để đối chiếu. Lần sửa tài liệu không được coi là một lượt train GPU mới. Các file bằng chứng ở `reports/`, `docs/`, `tests/` thuộc project local, chưa được công bố trong repo chỉ chứa notebooks.

<a id="usage"></a>

## 6. Tổ chức folder và vị trí đã chuyển

| Đường dẫn hiện tại | Nội dung cập nhật |
|---|---|
| [`README.md`](README.md) | README tổng: đối chiếu DINOv3 với commit SLARDA gốc, bản đồ code, kiến trúc, protocol, bằng chứng và cách chạy. |
| [`colab/train_sleep_dinov3_colab.ipynb`](colab/train_sleep_dinov3_colab.ipynb) | Chuyển notebook từ gốc vào `colab/`; nội dung cell giữ nguyên. |
| [`scripts/build_notebook.py`](scripts/build_notebook.py) | Chuyển script vào `scripts/`, tạo notebook tại đường dẫn mới. |
| [`scripts/package_colab.py`](scripts/package_colab.py) | Chuyển script vào `scripts/`, tạo ZIP code tại `generated/`. |
| [`scripts/prepare_colab_upload.py`](scripts/prepare_colab_upload.py) | Chuyển script vào `scripts/`, đọc notebook, ZIP và checkpoint từ vị trí mới. |
| [`requirements-colab.txt`](requirements-colab.txt) | Giữ ở gốc để dễ tìm cấu hình môi trường Colab. |
| `weights/model.safetensors` | Chuyển checkpoint cục bộ vào `weights/`; không đưa lên GitHub. |
| `generated/slarda_colab_code.zip` | Chuyển ZIP được sinh tự động vào `generated/`; không đưa lên GitHub. |
| `colab_upload/SleepCare/` | Giữ folder upload Drive hiện có; không đưa lên GitHub. |

Đường dẫn tham chiếu trong `README_WORKSPACE.md`, tài liệu `docs/` và tài liệu thí nghiệm DINOv3 đã cập nhật tương ứng. `.gitignore` loại `weights/`, `generated/` và `colab_upload/` khỏi Git.

### Cấu trúc folder

```text
notebooks/
├── README.md
├── requirements-colab.txt
├── colab/
│   └── train_sleep_dinov3_colab.ipynb   # Notebook chạy trên GPU
├── scripts/
│   ├── build_notebook.py               # Tạo notebook từ mã nguồn cell
│   ├── package_colab.py                # Đóng gói code Python
│   └── prepare_colab_upload.py         # Chuẩn bị folder cho Google Drive
├── weights/                            # Checkpoint cục bộ, Git bỏ qua
├── generated/                          # ZIP được tạo lại, Git bỏ qua
└── colab_upload/SleepCare/             # Gói Drive, Git bỏ qua
```

Các script cần `Autorgressive_Adaptation/` **bản đã bổ sung `dinov3_experiment/`** và `sleep_edf_tools/` ở cùng project `SLARDASSC`. Script chuẩn bị Drive còn cần dữ liệu tại `../datasets/sleep-edfx-1.0.0/processed/`. Nếu chỉ tải folder `notebooks/` từ GitHub, vẫn xem được notebook nhưng cần bổ sung đúng mã nguồn cập nhật cùng weights/dữ liệu để tạo gói chạy đầy đủ. Bộ code thuần upstream không đáp ứng điều kiện này.

## 7. Chuẩn bị và tạo lại gói Colab

1. Đặt checkpoint timm DINOv3 ViT-S/16 tại `notebooks/weights/model.safetensors`. SHA256 yêu cầu: `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`. Không đổi đuôi thành `.pth`.
2. Chuẩn bị `pilot_qc` và `medium_qc` trong `../datasets/sleep-edfx-1.0.0/processed/`. Mỗi profile cần `EEG/train_a.pt`, `val_a.pt`, `test_a.pt`, `train_b.pt`, `val_b.pt`, `test_b.pt`, `preprocessing.json`.
3. Chạy từ gốc `SLARDASSC`:

   ```powershell
   python notebooks/scripts/build_notebook.py
   python notebooks/scripts/prepare_colab_upload.py
   ```

   Lệnh thứ hai tự chạy `package_colab.py`, xác minh SHA256 checkpoint, tạo `notebooks/generated/slarda_colab_code.zip`, làm mới `notebooks/colab_upload/SleepCare/` và `UPLOAD_MANIFEST.json`. Nếu chỉ cần ZIP code: `python notebooks/scripts/package_colab.py`.

4. Upload **nguyên** `notebooks/colab_upload/SleepCare/` lên `My Drive/SleepCare/`. Mở `notebooks/colab/train_sleep_dinov3_colab.ipynb` bằng Colab hoặc VS Code với kernel Colab, chọn GPU và chạy cell theo thứ tự. Khi sửa code hoặc dữ liệu, chạy lại bước 3 rồi upload lại các file tương ứng; Drive không tự đồng bộ.

```text
My Drive/SleepCare/
├── slarda_colab_code.zip
├── train_sleep_dinov3_colab.ipynb
├── UPLOAD_MANIFEST.json
├── weights/model.safetensors
├── datasets/pilot_qc/EEG/...
├── datasets/medium_qc/EEG/...
└── runs/                         # Sinh sau khi chạy
```

## 8. Cấu hình chạy và đầu ra

### Đọc notebook theo từng bước

Notebook có 20 cell, trong đó 9 code cell tương ứng các bước sau. Trong [`scripts/build_notebook.py`](scripts/build_notebook.py), tìm đúng tiêu đề hoặc biến ở cột cuối để sửa nội dung rồi sinh lại notebook.

| Bước trong notebook | Việc thực hiện | Chỗ cần xem/sửa trong script tạo notebook |
|---|---|---|
| 1. Xác nhận kernel | Kiểm tra Python, PyTorch, CUDA và tên GPU | `torch.cuda.is_available()`, `nvidia-smi` |
| 2. Kết nối Drive, chọn run | Đặt full/smoke, run label, resume, block mở học, batch, source selection | `RUN_MODE`, `RUN_LABEL`, `RESUME`, `UNFREEZE_LAST_BLOCKS`, `FULL_BATCH_SIZE`, `SOURCE_SELECTION`, `WORK` |
| 3. Cài thư viện, kiểm tra code | Cài dependency, kiểm tra ZIP/upload manifest và SHA256 từng file code | `pip install`, `CODE_ZIP`, `bundle_manifest.json` |
| 4. Copy sang runtime | Chép weights và profile đã chọn từ Drive sang SSD Colab, xác minh checksum | `copy_verified()`, `WEIGHTS`, `DATA` |
| 5. Audit fit | Kiểm tra dữ liệu trước khi học, không mở tensor test | Lệnh `audit_data.py --fit-only` |
| 6. Preflight | Nạp pretrained thật, kiểm tra shape và gradient trong subprocess riêng | Lệnh `dinov3_experiment.preflight`, biến `PREFLIGHT` |
| 7. Train / resume | Gọi engine DINOv3, ghi môi trường/lệnh chạy/manifest, lưu model trên Drive | Mảng `command` gọi `train_CD.py --eeg-backbone dinov3` |
| 8. Đánh giá full | Đánh giá cả source và target trên `test_b`, tạo confusion matrix; smoke bỏ qua bước này | `dinov3_experiment.inference --variant both --test`, `METRICS` |
| 9. Xuất dự đoán web | Xuất JSON từ EEG NPY, mặc định minh họa bằng 120 epoch SC validation | `EEG_NPY`, `PREDICTIONS` |

Cell 9 hiện tự ghi dữ liệu minh họa vào `EEG_NPY` bằng `np.save(...)`. Khi dùng NPY riêng, cần thay đường dẫn **và bỏ/đổi đoạn tạo dữ liệu minh họa**, tránh ghi dữ liệu validation lên file riêng của mình.

### Thông số full và smoke

| Tham số | `full` mặc định | `smoke` |
|---|---|---|
| Dữ liệu | `medium_qc`, 28 người | `pilot_qc`, 7 bản ghi |
| Source / adaptation epochs | 25 / 15 | 1 / 1 |
| Batch size | 8 | 2 |
| Giới hạn batch | Không | 2 batch/epoch |
| `UNFREEZE_LAST_BLOCKS` | 1 | 1 |

Notebook dùng `timm==1.0.22`, `safetensors==0.5.3`, strict-load toàn bộ checkpoint và giữ PyTorch CUDA của Colab. Một block cuối, final norm, projection và classifier được cập nhật (1.838.213 tham số). `UNFREEZE_LAST_BLOCKS=0` chỉ học phần head. Luồng học gồm source cross-entropy, temporal discriminator, adversarial target alignment và EMA confident pseudo-label CE.

`full` dùng QC train/val/test: SC 32.448 / 7.927 / 8.271 epoch; ST 6.089 / 1.955 / 2.008 epoch. Nhãn ST train/val là `-1`; ST test chỉ dùng sau fit để đánh giá. QC loại epoch có std `<1e-8` volt theo cùng quy tắc và giữ provenance. Dữ liệu gốc không bị ghi đè.

Checkpoint `model.pt` chứa source/target weights và metadata. `model.resume.pt` chứa thêm teacher, discriminator, optimizer, scheduler và RNG, lưu mỗi epoch. Sau khi reset Colab, chạy lại setup, giữ cấu hình rồi đặt `RESUME=True`; phần chưa lưu của epoch sẽ chạy lại. Không resume `smoke` thành `full`. Đổi `RUN_LABEL` để tạo run mới; notebook chặn ghi đè model đã có.

Run `full` xuất `metrics_test_b.json`, macro-F1, kappa, metric từng lớp, `confusion_matrix.png`, `environment.txt` và `predictions_web.json` lên Drive. `smoke` bỏ qua đánh giá toàn bộ test và gắn cờ `smoke_test_only`. Cell cuối mặc định dùng 120 epoch SC validation để minh họa xuất web; thay `EEG_NPY` khi dùng dữ liệu riêng. Softmax confidence không phải độ chắc chắn y khoa.

### Lệnh train local tương đương cấu hình full

Chỉ chạy khi project local đã có engine DINOv3, dependencies, checkpoint và dữ liệu như mục 7. Từ gốc `SLARDASSC`:

```powershell
python Autorgressive_Adaptation/train_CD.py --eeg-backbone dinov3 --dinov3-backend timm --dinov3-weights notebooks/weights/model.safetensors --data-dir ../datasets/sleep-edfx-1.0.0/processed/medium_qc/EEG --output checkpoints/dinov3_full/model.pt --device cuda --batch-size 8 --source-epochs 25 --adapt-epochs 15 --unfreeze-last-blocks 1 --source-selection last --seed 42
```

Lệnh này chỉ gọi bước train, không tự chạy audit/preflight như notebook. Để có đối chứng CNN, dùng cùng engine với `--eeg-backbone cnn`, cùng profile/seed/batch/số epoch/source selection và output riêng. Kiến trúc CNN không dùng các tùy chọn weights/backend/block của DINOv3.

## 9. Tài liệu liên quan và giới hạn sử dụng

- Đã xác minh SHA256 checkpoint khớp `timm/vit_small_patch16_dinov3.lvd1689m`; đã chạy strict-load, forward/backward, source CE, target adaptation, save/resume CUDA và inference ngắn trên RTX 3050. Đây là kiểm tra vận hành, **chưa phải kết quả train đầy đủ trên Colab**.
- Trong project gốc `SLARDASSC`, xem thêm `docs/DINOV3_COLAB_AUDIT.md`, `docs/dinov3_colab_setup_validation.json` và `docs/COLAB_VSCODE.md`. Những file này không nằm trong gói GitHub chỉ chứa `notebooks/`.
- Trước lần sắp xếp, notebook có 20 cell, 9 code cell biên dịch được; manifest xác nhận 17 file upload và 13 file code trong bundle. Sau khi cập nhật code hoặc dữ liệu, kiểm tra manifest mới.
- Notebook nhắm Python ≥3.11 và torch CUDA ≥2.7 trên Colab. Tài nguyên và quota GPU phụ thuộc tài khoản Colab.
