"""Deterministic QA subsets with rule-based deduplication (not clinical review)."""
import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from scripts.validate_data import records, sha256, validate


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ''.join(c for c in text if c.isalnum())


def group_questions(rows):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    questions = [normalize(r['question']) for r in rows]
    seen = {}
    blocks = defaultdict(list)
    for i, q in enumerate(questions):
        if q in seen:
            union(i, seen[q])
            continue
        seen[q] = i
        if len(q) >= 12:
            blocks[('prefix', q[:12])].append(i)
            blocks[('suffix', q[-12:])].append(i)
    excluded = set()
    for ids in blocks.values():
        # Highly repetitive blocks are conservatively excluded, not silently skipped.
        if len(ids) > 100:
            excluded.update(ids)
            continue
        for pos, a in enumerate(ids):
            for b in ids[pos + 1:]:
                qa, qb = questions[a], questions[b]
                if min(len(qa), len(qb)) / max(len(qa), len(qb)) < 0.85:
                    continue
                if SequenceMatcher(None, qa, qb, autojunk=False).ratio() >= 0.9:
                    union(a, b)
    groups = [root(i) for i in range(len(rows))]
    excluded_roots = {groups[i] for i in excluded}
    return groups, excluded_roots


def stratified_take(rows, count, rng):
    if count > len(rows) or count < 0:
        raise ValueError(f'Cannot take {count} from {len(rows)} rows')
    if count == 0:
        return [], list(rows)
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row.get('label', 'unknown'))].append(row)
    quotas = {k: len(v) * count // len(rows) for k, v in buckets.items()}
    order = sorted(buckets, key=lambda k: (-(len(buckets[k]) * count % len(rows)), k))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    chosen, rest = [], []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        chosen.extend(bucket[:quotas[key]])
        rest.extend(bucket[quotas[key]:])
    rng.shuffle(chosen)
    return chosen, rest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--smoke-manifest', type=Path, default=Path('data/processed/smoke/manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/medical_v1'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    source = validate(args.input)
    smoke = json.loads(args.smoke_manifest.read_text(encoding='utf-8'))
    if smoke['source']['sha256'] != source['sha256']:
        raise ValueError('Smoke manifest refers to a different source dataset')
    rows = list(records(args.input))
    print(f'Grouping {len(rows):,} questions; please wait...', flush=True)
    groups, blocked = group_questions(rows)
    smoke_roots = {groups[i] for i in smoke['source_record_indices_zero_based']}
    blocked |= smoke_roots
    representatives = {}
    filtered = Counter()
    for i, row in enumerate(rows):
        if groups[i] in blocked:
            filtered['smoke_or_repetitive_group'] += 1
            continue
        q, a = row['question'].strip(), row['answer'].strip()
        if not (8 <= len(q) <= 1000 and 20 <= len(a) <= 2000):
            filtered['character_length'] += 1
            continue
        if re.search(r'1[3-9]\d{9}|https?://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', q + a):
            filtered['contact_or_url_pattern'] += 1
            continue
        if '�' in q + a:
            filtered['replacement_character'] += 1
            continue
        item = dict(row, question=q, answer=a, source_record_index=i, duplicate_group=groups[i])
        old = representatives.get(groups[i])
        if old is None or row.get('score', 0) > old.get('score', 0):
            representatives[groups[i]] = item
    eligible = list(representatives.values())
    rng = random.Random(args.seed)
    test, rest = stratified_take(eligible, 500, rng)
    validation, rest = stratified_take(rest, 500, rng)
    train, _ = stratified_take(rest, 30000, rng)
    pilot, _ = stratified_take(train, 3000, rng)
    # A stable 100-item validation subset for quick generation comparisons; not the test set.
    dev100, _ = stratified_take(validation, 100, rng)
    splits = {'train_30000': train, 'train_3000': pilot, 'validation': validation,
              'test': test, 'dev100': dev100}
    if sha256(args.input) != source['sha256']:
        raise ValueError('Source changed during processing')
    args.output.mkdir(parents=True, exist_ok=True)
    info, files = {}, {}
    for name, selected in splits.items():
        path = args.output / f'{name}.jsonl'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        info[f'huatuo_{name}'] = {'file_name': path.name, 'formatting': 'alpaca',
                                 'columns': {'prompt': 'question', 'response': 'answer'}}
        files[path.name] = {'records': len(selected), 'sha256': sha256(path),
                            'labels': dict(Counter(str(r.get('label', 'unknown')) for r in selected))}
    manifest = {'purpose': 'pilot_and_30k_experiment', 'seed': args.seed, 'source': source,
                'smoke_manifest_sha256': sha256(args.smoke_manifest),
                'eligible_unique_groups': len(eligible), 'excluded_record_counts': dict(filtered),
                'files': files,
                'rules': {'question_chars': [8, 1000], 'answer_chars': [20, 2000],
                          'near_duplicate': 'NFKC alnum; shared prefix/suffix12; length ratio>=.85; SequenceMatcher>=.9',
                          'large_block': '>100 distinct normalized questions excluded',
                          'representative': 'highest source score then first source record'},
                'limitations': ['Rule-based screening, NOT clinical or complete privacy review.',
                               'Near duplicate blocking misses paraphrases with different prefixes and suffixes.',
                               'Source score is not a clinical correctness certificate.',
                               'train_3000 is nested in train_30000; dev100 is nested in validation.',
                               'Do not tune on test or add its QA to the RAG knowledge base.']}
    for filename, content in [('dataset_info.json', info), ('manifest.json', manifest)]:
        (args.output / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'eligible': len(eligible), 'files': files}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
