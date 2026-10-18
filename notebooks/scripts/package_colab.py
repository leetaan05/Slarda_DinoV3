"""Create a small code-only Colab bundle from the current working files."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'notebooks' / 'generated' / 'slarda_colab_code.zip')
    args = parser.parse_args()
    files = [ROOT / 'Autorgressive_Adaptation' / 'train_CD.py',
             ROOT / 'Autorgressive_Adaptation' / 'predict_sleep_stage.py']
    files += list((ROOT / 'Autorgressive_Adaptation' / 'dinov3_experiment').glob('*.py'))
    files += list((ROOT / 'sleep_edf_tools').glob('*.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', ZIP_DEFLATED) as archive:
        manifest = {}
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, relative)
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    print(args.output)
    print('SHA256:', hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
