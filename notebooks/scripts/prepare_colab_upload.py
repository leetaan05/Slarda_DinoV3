"""Prepare a private local SleepCare folder ready for Google Drive folder upload."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CHECKPOINT_SHA256 = '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'notebooks' / 'weights' / 'model.safetensors')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'datasets' / 'sleep-edfx-1.0.0')
    parser.add_argument('--profiles', nargs='+', default=['pilot_qc', 'medium_qc'])
    parser.add_argument('--output', type=Path, default=ROOT / 'notebooks' / 'colab_upload' / 'SleepCare')
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f'Checkpoint missing: {args.weights}')
    if checksum(args.weights) != EXPECTED_CHECKPOINT_SHA256:
        parser.error('This notebook is configured for the verified timm DINOv3 ViT-S/16 checkpoint; SHA256 mismatch')
    subprocess.run([sys.executable, str(ROOT / 'notebooks' / 'scripts' / 'package_colab.py')], check=True)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {ROOT / 'notebooks' / 'generated' / 'slarda_colab_code.zip': Path('slarda_colab_code.zip'),
             ROOT / 'notebooks' / 'colab' / 'train_sleep_dinov3_colab.ipynb': Path('train_sleep_dinov3_colab.ipynb'),
             args.weights: Path('weights') / 'model.safetensors'}
    for profile in args.profiles:
        folder = args.data_root / 'processed' / profile / 'EEG'
        for name in ('train_a.pt', 'val_a.pt', 'test_a.pt', 'train_b.pt', 'val_b.pt', 'test_b.pt', 'preprocessing.json'):
            path = folder / name
            if not path.is_file():
                parser.error(f'Dataset file missing: {path}')
            files[path] = Path('datasets') / profile / 'EEG' / name
    manifest = {'schema_version': 1, 'intended_drive_folder': '/content/drive/MyDrive/SleepCare',
                'checkpoint': {'path': 'weights/model.safetensors', 'sha256': checksum(args.weights)},
                'profiles': args.profiles, 'files': {}}
    for source, relative in files.items():
        digest = checksum(source)
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Incremental refresh; verify content, not only byte count/mtime.
        if not destination.exists() or checksum(destination) != digest:
            shutil.copy2(source, destination)
        manifest['files'][relative.as_posix()] = {'sha256': digest, 'bytes': source.stat().st_size}
        print(f'Prepared {relative}', flush=True)
    (args.output / 'UPLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    total = sum(value['bytes'] for value in manifest['files'].values())
    (args.output / 'README_UPLOAD.md').write_text(
        '# Upload lên Google Drive\n\n'
        'Upload nguyên thư mục **SleepCare** này vào **My Drive**. Không upload lên web public/Git.\n'
        'Mở train_sleep_dinov3_colab.ipynb, chọn GPU rồi chạy các cell theo thứ tự.\n'
        'Mặc định full: medium_qc, 25 source epoch + 15 adaptation epoch, mở block cuối DINOv3.\n'
        'Để kiểm tra nhanh chọn RUN_MODE="smoke"; notebook tự chọn pilot_qc và chỉ 2 batch/epoch.\n'
        'Checkpoint được nạp trực tiếp từ weights/model.safetensors; không đổi tên thành .pth.\n'
        'UPLOAD_MANIFEST.json dùng để kiểm tra checksum sau copy. Không chứa token/mật khẩu.\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'files': len(files), 'total_mib': round(total / 2**20, 1)}, indent=2))


if __name__ == '__main__':
    main()
