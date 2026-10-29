# Các lệnh Git/GitHub dùng cho tài liệu SLARDA_DINOv3

Cập nhật ngày 2026-10-30 cho repository [leetaan05/SleepcareAI](https://github.com/leetaan05/SleepcareAI).

Tài liệu nghiên cứu và mã `SLARDA_DINOv3` được cập nhật trên **`main`**. Nhánh **`web-deploy`** dành cho ứng dụng web. Vì nhánh mặc định trên GitHub có thể là `web-deploy`, luôn ghi rõ `main` khi clone, đọc nội dung hoặc cập nhật.

## 1. Cách lần cập nhật này được thực hiện

Thư mục làm việc `SleepCare` ban đầu không có `.git` ở thư mục tổng. Một bản clone riêng của `main` được tạo để rà soát tệp và thay đổi; không khởi tạo Git chồng lên các repository con có sẵn trong workspace.

Git CLI được dùng để kiểm tra nhánh, tạo bản clone và rà soát nội dung chuẩn bị cập nhật. **Việc ghi lên GitHub được thực hiện bằng plugin GitHub**, qua một commit mới nối tiếp `main`, không qua lệnh `git push`. Các lệnh push ở mục 4 là hướng dẫn cho những lần cập nhật sau.

Phạm vi tệp gồm README tiếng Anh, mã nguồn của hai nhánh nghiên cứu trong `SLARDA_DINOv3`, requirements, giấy phép hiện hữu, hình kiến trúc từ paper và tài liệu lệnh này. Toàn bộ `QuickRun/`, dữ liệu EEG, model weights, môi trường ảo, checkpoint và log huấn luyện được loại khỏi nội dung đưa lên.

## 2. Các lệnh Git đã dùng

Các lệnh dưới đây dùng trong PowerShell. `SleepcareAI-docs` là tên thư mục minh họa cho bản clone riêng; trong lần thực hiện, bản clone được đặt trong thư mục làm việc tạm của tác vụ.

### Kiểm tra công cụ và hai nhánh trên GitHub

```powershell
git --version
git ls-remote --heads https://github.com/leetaan05/SleepcareAI.git main web-deploy
```

Kết quả lúc bắt đầu:

```text
git version 2.49.0.windows.1
93904ca2f1275d225832afe86fa16cf63cb779b7  refs/heads/main
7f0cc68454a2c226e56a625e9e9add1aee8e76b4  refs/heads/web-deploy
```

`ls-remote` chỉ đọc thông tin nhánh. SHA là mã commit tại thời điểm kiểm tra và có thể thay đổi sau này.

### Clone đúng nhánh nghiên cứu

```powershell
git clone --branch main --single-branch https://github.com/leetaan05/SleepcareAI.git SleepcareAI-docs
cd SleepcareAI-docs
git status --short
git remote -v
git branch -avv
```

Kết quả cần thấy: nhánh hiện tại là `main`, remote `origin` trỏ đúng repository. `--single-branch` giới hạn bản clone vào lịch sử nhánh đã chọn; nó không chỉnh sửa nhánh khác trên GitHub.

### Rà soát và đưa đúng tệp vào vùng chuẩn bị commit

Sau khi đặt tài liệu và các tệp mã nguồn đã chọn vào bản clone:

```powershell
git add -- README.md README_GITHUB_COMMANDS.md SLARDA_DINOv3
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git ls-files -- SLARDA_DINOv3/QuickRun
git status --short
```

Ý nghĩa và kết quả kỳ vọng:

| Lệnh | Mục đích / kết quả |
| --- | --- |
| `git add -- ...` | Chỉ chuẩn bị các đường dẫn được liệt kê; các đường dẫn có dấu cách được truyền như một đối số khi cần. |
| `git diff --cached --check` | Không in lỗi khoảng trắng hoặc dấu conflict. |
| `git diff --cached --stat` | Cho biết số tệp và quy mô thay đổi. |
| `git diff --cached --name-only` | Xem chính xác danh sách đường dẫn sắp đưa lên. |
| `git ls-files -- SLARDA_DINOv3/QuickRun` | Không có kết quả: không có tệp QuickRun trong bản cập nhật. |
| `git status --short` | Xác nhận chỉ có các tài liệu và mã nguồn đã chọn. |

Khi rà toàn bộ mã nguồn sao chép, `git diff --cached --check` đã báo khoảng trắng cuối dòng và dòng trống cuối tệp trong một số tệp Python có sẵn. Nội dung Python được giữ theo bản nguồn. Tài liệu mới được kiểm tra riêng bằng lệnh sau và không có lỗi khoảng trắng:

```powershell
git diff --cached --check -- README.md README_GITHUB_COMMANDS.md SLARDA_DINOv3/README.md SLARDA_DINOv3/.gitignore
```

Các lệnh `git status --short`, `git remote -v`, `git branch -avv` cũng được thử tại thư mục tổng ban đầu và trả `fatal: not a git repository`. Vì vậy lần cập nhật dùng bản clone riêng thay vì coi toàn bộ Google Drive là một Git repository.

## 3. Thao tác plugin GitHub thực sự dùng để xuất bản

Plugin đọc các nhánh, cây tệp và README tham khảo; sau đó thực hiện các thao tác Git data tương đương tạo một commit:

```text
fetch                       Đọc nhánh và cây tệp hiện tại
fetch_file                  Đọc README tham khảo và nội dung cần giữ lại
create_blob                 Tạo nội dung từng tệp đã chọn
create_tree                 Ghép cây mới dựa trên cây main hiện tại
create_commit               Tạo commit có parent là commit main đã kiểm tra
update_ref(branch_name=main) Đưa main đến commit mới
fetch                       Đọc lại để xác nhận kết quả
```

Tên repository được chỉ định là `leetaan05/SleepcareAI`; nhánh ghi được chỉ định là `main`. Cây tệp cũ được dùng làm nền để giữ các tệp ngoài phạm vi cập nhật. Nhánh được kiểm tra lại trước khi đổi ref; nếu main thay đổi đồng thời thì phải đọc trạng thái mới và ghép lại, không ghi đè lịch sử bằng force push.

Sau khi cập nhật, kiểm tra SHA của `web-deploy` vẫn là `7f0cc68454a2c226e56a625e9e9add1aee8e76b4` so với lúc bắt đầu. Đây là phép kiểm tra cho lần cập nhật này, không phải yêu cầu nhánh đó phải giữ nguyên mãi.

## 4. Quy trình Git CLI cho lần cập nhật sau

Đây là hướng dẫn tương đương để bạn tự cập nhật bằng Git. Các lệnh commit/push dưới đây **không phải lệnh đã chạy để xuất bản lần này**.

Trong bản clone đúng repository và sau khi kiểm tra không còn thay đổi chưa xử lý:

```powershell
git switch main
git pull --ff-only origin main
git branch --show-current
```

Kết quả cần thấy là `main`. Sửa README và hình minh họa, sau đó chỉ chọn các tệp thực sự cần cập nhật:

```powershell
git add -- README.md README_GITHUB_COMMANDS.md SLARDA_DINOv3/README.md SLARDA_DINOv3/docs/figures/slarda-framework.png
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "docs: explain SLARDA DINOv3 encoder integration"
git push origin main
```

Đường dẫn trong `git add` cần được điều chỉnh nếu bạn chủ động sửa thêm tệp. Với tệp có khoảng trắng, dùng dấu nháy, ví dụ:

```powershell
git add -- 'SLARDA_DINOv3/Self-supervised training/models/EEG_models.py'
```

`git push origin main` chỉ rõ nhánh nhận thay đổi. Không dùng `git push --all`, `git push --mirror` hoặc force push cho quy trình tài liệu này.

Xác nhận sau khi push:

```powershell
git log -1 --oneline
git status --short
git ls-remote --heads origin main web-deploy
```

Kết quả kỳ vọng: commit mới hiện trong lịch sử `main`, vùng làm việc sạch và chỉ SHA của nhánh được chủ động cập nhật thay đổi. Xem README tại [SLARDA_DINOv3 trên main](https://github.com/leetaan05/SleepcareAI/tree/main/SLARDA_DINOv3).

## 5. Xử lý tình huống thường gặp

| Tình huống | Cách xử lý |
| --- | --- |
| `not a git repository` | Chuyển vào bản clone có `.git`. Thư mục tổng Google Drive ban đầu không phải repo. |
| Đang ở `web-deploy` | Dừng cập nhật tại checkout đó; dùng bản clone riêng với `--branch main --single-branch` như trên. |
| Push bị từ chối do main đã có commit mới | Đọc thay đổi từ remote và tích hợp trước khi push lại; không dùng force để bỏ qua. |
| Git yêu cầu đăng nhập khi push | Đăng nhập tài khoản có quyền ghi qua Git Credential Manager hoặc công cụ xác thực GitHub của bạn. |
| Lỡ chọn một tệp không cần đưa lên | Dùng `git restore --staged -- 'duong/dan/tep'` để bỏ nó khỏi vùng chuẩn bị commit; nội dung tệp vẫn còn. |
| `.gitignore` không loại một tệp đã được theo dõi | Kiểm tra `git ls-files`; ignore chỉ áp dụng tự động cho tệp chưa được theo dõi. |

Tài liệu liên quan: [README tiếng Anh về encoder DINOv3](SLARDA_DINOv3/README.md).
