"""Generate the DINOv3 Colab notebook for the verified timm safetensors checkpoint."""
import json
from pathlib import Path
from textwrap import dedent

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': dedent(text).strip().splitlines(True)})


def code(text):
    source = dedent(text).strip() + '\n'
    compile(source, '<notebook-cell>', 'exec')
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(True)})


md('''
# Train Sleep-EDF · DINOv3 backbone + SLARDA · Colab GPU

Notebook đã cấu hình cho checkpoint **model.safetensors** bạn cung cấp: DINOv3 ViT-S/16, `timm/vit_small_patch16_dinov3.lvd1689m`.
SHA256 khớp file upstream: `2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040`.
Loader dùng **timm==1.0.22**, nạp **strict=True**, không đổi đuôi thành .pth, không thay bằng DINOv2.

**Chuẩn bị một lần:** upload nguyên folder `notebooks/colab_upload/SleepCare` vào **Google Drive / My Drive**.
Chọn runtime **GPU**, chạy các cell theo thứ tự. Mặc định là train đầy đủ: **medium_qc, 25 source epoch + 15 adaptation epoch, mở block cuối DINOv3**.
Để kiểm tra nhanh, đổi RUN_MODE thành `smoke`: pilot_qc, 1 source + 1 adaptation epoch, 2 batch/epoch.

EEG [B,1,3000] → spectrogram → DINOv3 → [B,128,22] → classifier 5 lớp + GRU domain discriminator.
Giữ source cross-entropy, adversarial target alignment và EMA confident pseudo-label CE. ST train/val nhãn -1; test chỉ đánh giá sau fit.
Các sửa so với trainer legacy được mô tả trong README_AUDIT.md. Không bao gồm pretraining DINO từ đầu hay giai đoạn CPC riêng.
''')
md('## 1. Xác nhận kernel Colab có GPU')
code('''
import sys, subprocess, json, shutil, hashlib, zipfile, os, gc
from pathlib import Path
assert sys.version_info >= (3, 11), 'Notebook Colab dùng Python >= 3.11.'
import torch
from packaging.version import Version
assert Version(torch.__version__.split('+')[0]) >= Version('2.7'), 'Chọn runtime Colab mới có PyTorch CUDA >=2.7.'
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
assert torch.cuda.is_available(), 'Chọn Runtime -> Change runtime type -> GPU (hoặc Colab GPU kernel trong VS Code).'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
subprocess.run(['nvidia-smi'], check=True)
''')
md('''
## 2. Kết nối Drive và chọn run

`full` thực sự train toàn bộ các batch với 25/15 epoch; không bị giới hạn ngầm bởi smoke-test.
`smoke` chỉ kiểm tra vận hành. Hai chế độ có folder riêng. Đổi RUN_LABEL để tạo thí nghiệm mới.
Sau ngắt runtime, giữ nguyên cấu hình và đặt RESUME=True. Không resume smoke thành full.
''')
code('''
#@title Cấu hình huấn luyện DINOv3
from google.colab import drive
# Nếu đã mount trong VS Code qua extension, không cần mount lại.
if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')

RUN_MODE = 'full'  #@param ['smoke', 'full']
RUN_LABEL = 'run01'  #@param {type:'string'}
RESUME = False  #@param {type:'boolean'}
UNFREEZE_LAST_BLOCKS = 1  #@param {type:'integer'}
FULL_BATCH_SIZE = 8  #@param {type:'integer'}
SOURCE_SELECTION = 'last'  #@param ['last', 'best']

WORK = Path('/content/drive/MyDrive/SleepCare')
CODE_ZIP = WORK / 'slarda_colab_code.zip'
UPLOAD_MANIFEST = WORK / 'UPLOAD_MANIFEST.json'
WEIGHTS_DRIVE = WORK / 'weights' / 'model.safetensors'
EXPECTED_WEIGHTS_SHA256 = '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'
PROFILE = 'pilot_qc' if RUN_MODE == 'smoke' else 'medium_qc'
SOURCE_EPOCHS, ADAPT_EPOCHS = (1, 1) if RUN_MODE == 'smoke' else (25, 15)
BATCH_SIZE = 2 if RUN_MODE == 'smoke' else FULL_BATCH_SIZE
BACKBONE, BACKEND = 'dinov3', 'timm'
DATA_DRIVE = WORK / 'datasets' / PROFILE / 'EEG'
RUN_NAME = f'{PROFILE}_dinov3_timm_blocks{UNFREEZE_LAST_BLOCKS}_{RUN_MODE}_{RUN_LABEL}'
OUTPUT = WORK / 'runs' / RUN_NAME / 'model.pt'
assert RUN_MODE in ('smoke', 'full')
assert 0 <= UNFREEZE_LAST_BLOCKS <= 12 and BATCH_SIZE > 0
assert RUN_LABEL and '/' not in RUN_LABEL and chr(92) not in RUN_LABEL
missing = [str(p) for p in (CODE_ZIP, UPLOAD_MANIFEST, WEIGHTS_DRIVE) if not p.is_file()]
assert not missing, 'Upload đúng folder SleepCare vào My Drive. Thiếu: ' + ', '.join(missing)
if OUTPUT.exists() and not RESUME:
    raise FileExistsError(f'{OUTPUT} đã có. Đổi RUN_LABEL hoặc RESUME=True, không ghi đè model cũ.')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
print(json.dumps({'mode': RUN_MODE, 'profile': PROFILE, 'backend': BACKEND,
                  'source_epochs': SOURCE_EPOCHS, 'adapt_epochs': ADAPT_EPOCHS,
                  'batch_size': BATCH_SIZE, 'unfreeze_last_blocks': UNFREEZE_LAST_BLOCKS,
                  'output': str(OUTPUT)}, indent=2))
''')
md('''
## 3. Cài đúng thư viện và kiểm tra bundle code

Giữ nguyên torch/torchvision CUDA do Colab cung cấp. Backend timm dùng trực tiếp safetensors đã có, không cần clone repo Meta hoặc đăng nhập Hugging Face.
Không fallback sang random weights khi checkpoint lỗi.
''')
code('''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'timm==1.0.22', 'safetensors==0.5.3', 'mne>=1.7,<2', 'matplotlib>=3.8,<4'], check=True)
import timm, safetensors
print('timm:', timm.__version__, '| safetensors:', safetensors.__version__)

def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

upload_manifest = json.loads(UPLOAD_MANIFEST.read_text())
assert sha256(CODE_ZIP) == upload_manifest['files']['slarda_colab_code.zip']['sha256'], 'Bundle upload thiếu/hỏng.'
CODE = Path('/content') / ('slarda_' + sha256(CODE_ZIP)[:12])
CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(CODE_ZIP) as archive:
    for name in archive.namelist():
        assert (CODE / name).resolve().is_relative_to(CODE.resolve()), 'Unsafe zip path'
    archive.extractall(CODE)
bundle = json.loads((CODE / 'bundle_manifest.json').read_text())
for name, digest in bundle.items():
    path = (CODE / name).resolve()
    assert path.is_relative_to(CODE.resolve()) and sha256(path) == digest, name
TRAIN = CODE / 'Autorgressive_Adaptation'
print('Code verified:', len(bundle), 'files')
''')
md('''
## 4. Copy data và checkpoint từ Drive sang SSD runtime

Chỉ copy tensor đã xử lý, không cần raw EDF 8 GiB. File local được kiểm tra SHA256 theo manifest upload.
`medium_qc` giữ 18 người SC + 10 người ST, chia theo subject. QC dựa trên std tín hiệu, cùng quy tắc cho hai miền.
Epoch hằng đã loại: ST medium train/val/test còn 6.089/1.955/2.008. Bộ gốc vẫn giữ riêng ở workspace.
''')
code('''
DATA = Path('/content/sleepcare_data/processed') / PROFILE / 'EEG'
WEIGHTS = Path('/content/sleepcare_weights/model.safetensors')
DATA.mkdir(parents=True, exist_ok=True)
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

def copy_verified(source, destination, manifest_key):
    expected = upload_manifest['files'][manifest_key]['sha256']
    if not destination.exists() or sha256(destination) != expected:
        shutil.copy2(source, destination)
    assert sha256(destination) == expected, f'Checksum mismatch: {destination}'
    print('Verified:', destination.name)

for name in ['train_a.pt', 'val_a.pt', 'test_a.pt', 'train_b.pt', 'val_b.pt', 'test_b.pt', 'preprocessing.json']:
    source = DATA_DRIVE / name
    assert source.is_file(), f'Thiếu file Drive: {source}'
    copy_verified(source, DATA / name, f'datasets/{PROFILE}/EEG/{name}')
copy_verified(WEIGHTS_DRIVE, WEIGHTS, 'weights/model.safetensors')
assert sha256(WEIGHTS) == EXPECTED_WEIGHTS_SHA256, 'Sai checkpoint DINOv3 ViT-S/16 đã kiểm tra.'
''')
md('''
## 5. Audit dữ liệu dùng cho fit

Không mở tensor test trong bước audit trước train. Kiểm tra train/validation shape, nhãn, finite, chất lượng và subject leakage.
Engine chỉ dùng SC train/validation và ST train; ST validation không dùng nhãn để chọn model.
''')
code('''
subprocess.run([sys.executable, str(CODE/'sleep_edf_tools/audit_data.py'),
                '--root', str(DATA.parents[2]), '--profiles', PROFILE, '--fit-only',
                '--output', str(OUTPUT.parent/'fit_data_audit.json')], check=True)
''')
md('''
## 6. Preflight GPU với pretrained DINOv3 thật

Chạy source CE và một bước SLARDA để xác nhận strict-load, shape, finite và gradient block cuối.
Đây là model tạm trong subprocess, không thay đổi checkpoint gốc hoặc khởi tạo của lượt train chính.
Nếu cell này lỗi thì dừng, không chạy tiếp train đầy đủ.
''')
code('''
PREFLIGHT = OUTPUT.parent / 'preflight.json'
subprocess.run([sys.executable, '-u', '-m', 'dinov3_experiment.preflight',
                '--dinov3-weights', str(WEIGHTS), '--dinov3-backend', BACKEND,
                '--expected-sha256', EXPECTED_WEIGHTS_SHA256,
                '--data-dir', str(DATA), '--device', 'cuda',
                '--unfreeze-last-blocks', str(UNFREEZE_LAST_BLOCKS),
                '--batch-size', str(min(2, BATCH_SIZE)), '--output', str(PREFLIGHT)], cwd=TRAIN, check=True)
preflight = json.loads(PREFLIGHT.read_text())
assert preflight['status'] == 'passed'
print('Preflight passed. Trainable parameters:', preflight['trainable_parameters'])
''')
md('''
## 7. Train / resume mô hình

Full: source 25 epoch, adaptation 15 epoch; Adam/loss weights/EMA theo cấu hình SLARDA.
Log tiến độ mỗi 50 batch. Checkpoint resume lưu mỗi epoch, gồm model/teacher/discriminator, optimizer/scheduler và RNG CPU/CUDA.
Ngắt giữa epoch sẽ chạy lại phần chưa lưu. Mỗi epoch adaptation dừng theo loader ngắn hơn (`zip`) như repo.
Nếu thiếu VRAM, dùng run mới với batch4 hoặc2; không đổi batch khi resume. Blocks=0 chỉ train head, blocks=1 fine-tune cả block cuối.
''')
code('''
assert PREFLIGHT.is_file() and json.loads(PREFLIGHT.read_text())['status'] == 'passed'
command = [sys.executable, '-u', 'train_CD.py', '--eeg-backbone', BACKBONE,
           '--dinov3-backend', BACKEND, '--dinov3-weights', str(WEIGHTS),
           '--data-dir', str(DATA), '--output', str(OUTPUT), '--device', 'cuda',
           '--batch-size', str(BATCH_SIZE), '--source-epochs', str(SOURCE_EPOCHS),
           '--adapt-epochs', str(ADAPT_EPOCHS), '--source-selection', SOURCE_SELECTION,
           '--unfreeze-last-blocks', str(UNFREEZE_LAST_BLOCKS), '--seed', '42', '--log-every', '50']
if RUN_MODE == 'smoke':
    command += ['--max-batches', '2']
if RESUME:
    resume_path = OUTPUT.with_suffix('.resume.pt')
    assert resume_path.is_file(), f'Không có resume checkpoint: {resume_path}'
    command += ['--resume', str(resume_path)]
(OUTPUT.parent/'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
(OUTPUT.parent/'code_manifest.json').write_text(json.dumps(bundle, indent=2))
(OUTPUT.parent/'run_command.json').write_text(json.dumps(command, indent=2))
print('Bắt đầu:', RUN_MODE, '->', OUTPUT.parent)
subprocess.run(command, cwd=TRAIN, check=True)
assert OUTPUT.is_file()
print('Đã lưu model:', OUTPUT)
''')
md('''
## 8. Đánh giá sau fit (full run)

Chỉ sau khi train xong mới dùng test_b. Không chọn hyperparameter hoặc checkpoint bằng ST test.
Smoke không chạy đánh giá toàn bộ test; chỉ kiểm tra xuất dự đoán ở cell sau. Không coi smoke là accuracy của thí nghiệm đầy đủ.
''')
code('''
METRICS = OUTPUT.parent / 'metrics_test_b.json'
if RUN_MODE == 'full':
    subprocess.run([sys.executable, '-u', '-m', 'dinov3_experiment.inference', '--model', str(OUTPUT),
                    '--variant', 'both', '--device', 'cuda', '--test', str(DATA/'test_b.pt'),
                    '--output', str(METRICS), '--dinov3-weights', str(WEIGHTS)], cwd=TRAIN, check=True)
    scores = json.loads(METRICS.read_text())
    for variant in ('source', 'target'):
        print(variant, {k: scores[variant][k] for k in ('accuracy', 'macro_f1', 'kappa', 'epochs')})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, variant in zip(axes, ('source', 'target')):
        matrix = scores[variant]['confusion_matrix']
        ax.imshow(matrix, cmap='Blues')
        ax.set(xticks=range(5), yticks=range(5), xticklabels=scores['class_names'],
               yticklabels=scores['class_names'], xlabel='Predicted', ylabel='True', title=variant)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, matrix[i][j], ha='center', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT.parent/'confusion_matrix.png', dpi=150)
    plt.show()
else:
    print('Smoke completed; bỏ qua đánh giá toàn bộ test. Đổi RUN_MODE=full, RUN_LABEL mới để train thật.')
''')
md('''
## 9. Xuất JSON dự đoán để xem trên web

Mặc định dùng 120 epoch SC validation cho ví dụ xuất file, không đánh giá chất lượng từ ví dụ này.
Thay `EEG_NPY` bằng NPY [N,3000] Fpz-Cz 100 Hz của bạn để phân tích bản ghi riêng.
JSON có tên model/variant và cờ smoke; web không hiển thị như kết quả DINO đã train đầy đủ nếu run chỉ smoke.
''')
code('''
import numpy as np
EEG_NPY = Path('/content/eeg_export_example.npy')
validation = torch.load(DATA/'val_a.pt', map_location='cpu', weights_only=True)
np.save(EEG_NPY, validation['samples'][:120, 0].numpy(), allow_pickle=False)
del validation
PREDICTIONS = OUTPUT.parent / 'predictions_web.json'
subprocess.run([sys.executable, '-u', '-m', 'dinov3_experiment.inference', '--model', str(OUTPUT),
                '--variant', 'target', '--device', 'cuda', '--input', str(EEG_NPY),
                '--output', str(PREDICTIONS), '--dinov3-weights', str(WEIGHTS)], cwd=TRAIN, check=True)
print('Trên SleepCare: Nhập kết quả ->', PREDICTIONS)
print('Tất cả checkpoint/báo cáo đã lưu ở Drive:', OUTPUT.parent)
''')
md('''
## Nếu cần kiểm tra lỗi

- Không thấy GPU: chọn đúng Colab GPU kernel; notebook này không dùng GPU máy Windows khi đang kết nối Colab.
- File .safetensors: giữ đúng tên và vị trí `SleepCare/weights/model.safetensors`; không đổi sang .pth.
- Strict-load lỗi: kiểm tra checkpoint và timm1.0.22. Không dùng strict=False để bỏ qua trọng số thiếu.
- Resume: giữ data/checkpoint/backend/batch/blocks/source_epochs/seed; chỉ được tăng adapt_epochs nếu tiếp tục thêm epoch.
- Sau khi sửa code local: chạy `prepare_colab_upload.py` lại rồi upload lại code bundle/manifest tương ứng, không dùng bundle cũ.

Nguồn checkpoint: https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m
Tài liệu VS Code: https://github.com/googlecolab/colab-vscode/wiki/User-Guide
Colab free có quota và GPU không bảo đảm; model và metrics lưu trên Drive để chịu reset runtime.
''')

notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'}, 'accelerator': 'GPU',
    'colab': {'name': 'train_sleep_dinov3_colab.ipynb', 'provenance': []}}, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'sleepcare-{i:02d}'
path = Path(__file__).resolve().parents[1] / 'colab' / 'train_sleep_dinov3_colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
