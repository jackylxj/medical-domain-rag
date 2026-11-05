"""Retrieve fresh questions against the existing Milvus Lite collection."""
import argparse
import json
from pathlib import Path

from experiments.retrieval_v1.run import PREFIX
from scripts.validate_data import sha256

ROOT = Path(__file__).resolve().parents[1]


def load_questions(path):
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty question list')
    questions = [row['question'] for row in rows]
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError('Question must be nonempty text')
    if len(set(questions)) != len(questions):
        raise ValueError('Duplicate questions')
    return questions  # Never pass review notes or expected sources to the model.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--questions', type=Path)
    source.add_argument('--query')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), help='Explicit device; CPU avoids competing with resident vLLM')
    parser.add_argument('--db', type=Path, default=ROOT / 'outputs/retrieval_v1/knowledge.db')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    questions = load_questions(args.questions) if args.questions else [args.query]
    if any(not q.strip() for q in questions):
        parser.error('Query must be nonempty')
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    if not args.db.exists() or not (args.model_dir / 'config.json').is_file():
        raise FileNotFoundError('Existing Milvus database or local BGE config missing')
    from sentence_transformers import SentenceTransformer
    from pymilvus import MilvusClient
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = SentenceTransformer(str(args.model_dir), **({'device':args.device} if args.device else {}))
    vectors = model.encode([PREFIX + q for q in questions], normalize_embeddings=True)
    if vectors.shape != (len(questions), 1024):
        raise ValueError('Expected BGE-Large 1024-dimensional vectors')
    client = MilvusClient(uri=str(args.db.resolve()))
    try:
        client.load_collection('medical_sections')
        hits = client.search('medical_sections', vectors.tolist(), limit=3,
                             search_params={'metric_type': 'COSINE', 'params': {}},
                             output_fields=['source_id', 'title', 'section', 'text', 'source', 'version'])
    finally:
        client.close()
    results = [dict(question=q, hits=[dict(h['entity'], cosine=float(h['distance'])) for h in matches])
               for q, matches in zip(questions, hits, strict=True)]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = dict(questions=len(questions), model_dir=str(args.model_dir.resolve()),
                    database=str(args.db.resolve()), top_k=3, device=str(model.device),
                    questions_sha256=sha256(args.questions) if args.questions else None,
                    note='Fresh retrieval only; cosine score is not an answerability threshold.')
    original_manifest = args.db.parent / 'summary.json'
    if original_manifest.is_file():
        metadata['database_build_summary_sha256'] = sha256(original_manifest)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(results)} freshly retrieved questions: {args.output / "results.json"}')


if __name__ == '__main__':
    main()
