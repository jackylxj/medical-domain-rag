"""Prepare smoke data only; this is not a formal train/validation/test split."""
import argparse
import json
import random
from pathlib import Path
from scripts.validate_data import records, sha256, validate


def prepare(source, output, count=200, seed=42):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty directory: {output}')
    if count < 1:
        raise ValueError('count must be positive')
    summary = validate(source)
    if count > summary['records']:
        raise ValueError('Requested more samples than available')
    indices = random.Random(seed).sample(range(summary['records']), count)
    selected = set(indices)
    pool = {i: row for i, row in enumerate(records(source)) if i in selected}
    if sha256(source) != summary['sha256']:
        raise ValueError('Source changed during processing; wait for upload to finish')
    output.mkdir(parents=True, exist_ok=True)
    data = output / 'train.jsonl'
    with data.open('w', encoding='utf-8', newline='\n') as f:
        for i in indices:
            f.write(json.dumps(pool[i], ensure_ascii=False) + '\n')
    info = {'huatuo_smoke': {'file_name': 'train.jsonl', 'formatting': 'alpaca',
                           'columns': {'prompt': 'question', 'response': 'answer'}}}
    manifest = {'purpose': 'smoke_only_not_evaluation', 'source': summary,
                'seed': seed, 'sample_count': count, 'source_record_indices_zero_based': indices,
                'sample_sha256': sha256(data),
                'warning': 'Exclude these samples and near duplicates from future test sets.'}
    for name, obj in [('dataset_info.json', info), ('manifest.json', manifest)]:
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/processed/smoke'))
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.input, args.output, args.count, args.seed)
    print(f"Prepared {result['sample_count']} samples in {args.output}")
    print('SHA256:', result['sample_sha256'])
