# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

"""Download an offline CTranslate2 Whisper model into this project."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['base', 'medium'], default='medium', nargs='?')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    destination = Path(__file__).resolve().parent.parent / 'data' / 'models' / f'whisper-{args.model}'
    print(f'下载 {args.model} 模型到 {destination}', flush=True)
    snapshot_download(
        repo_id=f'Systran/faster-whisper-{args.model}',
        local_dir=str(destination),
        allow_patterns=['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.*', 'preprocessor_config.json'],
    )
    print('模型下载完成', flush=True)


if __name__ == '__main__':
    main()
