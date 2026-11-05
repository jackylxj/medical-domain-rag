"""Strict JSONL validation; no silent decoding errors or medical quality claims."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    with Path(path).open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at line {line_no}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Line {line_no}: expected object')
            for key in ('question', 'answer'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'Line {line_no}: missing/empty {key}')
            yield row


def validate(path):
    path = Path(path)
    count = sum(1 for _ in records(path))
    if count == 0:
        raise ValueError('Dataset is empty')
    return {'path': str(path.resolve()), 'records': count,
            'bytes': path.stat().st_size, 'sha256': sha256(path)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.input), ensure_ascii=False, indent=2))
