# Notebooks · DINOv3 + SLARDA trên Google Colab

Cập nhật **18/10/2026**. Đây là README tổng của `notebooks/`. Notebook chính: [`colab/train_sleep_dinov3_colab.ipynb`](colab/train_sleep_dinov3_colab.ipynb). Mọi lệnh bên dưới chạy từ **thư mục gốc `SLARDASSC`**.

## Chỉ chỗ đã update

| Đường dẫn hiện tại | Nội dung cập nhật |
|---|---|
| [`README.md`](README.md) | Viết lại bản đồ folder, cách sử dụng, trạng thái kiểm chứng và danh sách vị trí cập nhật. |
| [`colab/train_sleep_dinov3_colab.ipynb`](colab/train_sleep_dinov3_colab.ipynb) | Chuyển notebook từ gốc vào `colab/`; nội dung cell giữ nguyên. |
| [`scripts/build_notebook.py`](scripts/build_notebook.py) | Chuyển script vào `scripts/`, tạo notebook tại đường dẫn mới. |
| [`scripts/package_colab.py`](scripts/package_colab.py) | Chuyển script vào `scripts/`, tạo ZIP code tại `generated/`. |
| [`scripts/prepare_colab_upload.py`](scripts/prepare_colab_upload.py) | Chuyển script vào `scripts/`, đọc notebook, ZIP và checkpoint từ vị trí mới. |
| [`requirements-colab.txt`](requirements-colab.txt) | Giữ ở gốc để dễ tìm cấu hình môi trường Colab. |
| `weights/model.safetensors` | Chuyển checkpoint cục bộ vào `weights/`; không đưa lên GitHub. |
| `generated/slarda_colab_code.zip` | Chuyển ZIP được sinh tự động vào `generated/`; không đưa lên GitHub. |
| `colab_upload/SleepCare/` | Giữ folder upload Drive hiện có; không đưa lên GitHub. |

Đường dẫn tham chiếu trong `README_WORKSPACE.md`, tài liệu `docs/` và tài liệu thí nghiệm DINOv3 đã cập nhật tương ứng. `.gitignore` loại `weights/`, `generated/` và `colab_upload/` khỏi Git.

## Cấu trúc folder

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

Các script cần `Autorgressive_Adaptation/` và `sleep_edf_tools/` ở **cùng repo `SLARDASSC`**. Script chuẩn bị Drive còn cần dữ liệu tại `../datasets/sleep-edfx-1.0.0/processed/`. Nếu chỉ tải riêng folder `notebooks/` từ GitHub, vẫn xem được notebook nhưng cần bổ sung các thư mục phụ thuộc này để tạo gói chạy đầy đủ.

## Chuẩn bị và tạo lại gói Colab

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

## Cấu hình chạy và đầu ra

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

## Kiểm chứng và giới hạn

- Đã xác minh SHA256 checkpoint khớp `timm/vit_small_patch16_dinov3.lvd1689m`; đã chạy strict-load, forward/backward, source CE, target adaptation, save/resume CUDA và inference ngắn trên RTX 3050. Đây là kiểm tra vận hành, **chưa phải kết quả train đầy đủ trên Colab**.
- Trong project gốc `SLARDASSC`, xem thêm `docs/DINOV3_COLAB_AUDIT.md`, `docs/dinov3_colab_setup_validation.json` và `docs/COLAB_VSCODE.md`. Những file này không nằm trong gói GitHub chỉ chứa `notebooks/`.
- Trước lần sắp xếp, notebook có 20 cell, 9 code cell biên dịch được; manifest xác nhận 17 file upload và 13 file code trong bundle. Sau khi cập nhật code hoặc dữ liệu, kiểm tra manifest mới.
- Notebook nhắm Python ≥3.11 và torch CUDA ≥2.7 trên Colab. Tài nguyên và quota GPU phụ thuộc tài khoản Colab.
