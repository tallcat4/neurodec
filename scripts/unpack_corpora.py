# SPDX-License-Identifier: GPL-3.0-or-later
"""Unpack the bundled corpora offline (works on Windows as well as Linux)."""
import stat
import tarfile
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2] / 'corpora'
    for name in ('jsut_ver1.1.zip', 'jvs_ver1.zip', 'libritts_r_train_clean_100.tar.gz'):
        archive = root / name
        marker = root / (name + '.extracted')
        if marker.exists():
            print(f'Already extracted: {name}', flush=True)
            continue
        print(f'Extracting {name}', flush=True)
        if name.endswith('.zip'):
            with zipfile.ZipFile(archive) as z:
                for item in z.infolist():
                    dest = (root / item.filename).resolve()
                    if not dest.is_relative_to(root.resolve()) or stat.S_ISLNK(item.external_attr >> 16):
                        raise ValueError(f'Unsafe archive member: {item.filename}')
                z.extractall(root)
        else:
            with tarfile.open(archive, 'r|gz') as t:
                for item in t:
                    if not (item.isfile() or item.isdir()):
                        raise ValueError(f'Unexpected archive member: {item.name}')
                    t.extract(item, root, filter='data')
        marker.write_text('Extraction completed\n', encoding='utf-8')
        print(f'Completed {name}', flush=True)


if __name__ == '__main__':
    main()
