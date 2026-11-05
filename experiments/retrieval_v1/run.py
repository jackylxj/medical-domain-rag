"""Small Chinese medical retrieval experiment; independent of Huatuo QA."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "BAAI/bge-large-zh-v1.5"
REVISION = "79e7739b6ab944e86d6171e44d24c997fc1e0116"
PREFIX = "为这个句子生成表示以用于检索相关文章："


def load_chunks():
    chunks = []
    for path in sorted((HERE / "references").glob("*.md")):
        parts = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)
        title, url, version = parts[0].strip().splitlines()
        for part in parts[1:]:
            section, body = part.strip().split("\n", 1)
            chunks.append(dict(id=len(chunks), source_id=path.stem + "/" + section,
                               title=title.removeprefix("# "), section=section, text=body.strip(),
                               source=url.removeprefix("来源："), version=version.removeprefix("版本：")))
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Local snapshot of the specified BGE checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval_v1")
    parser.add_argument("--query", help="Optional additional query, excluded from metrics")
    parser.add_argument("--search-existing", action="store_true", help="Query the saved database without rebuilding")
    args = parser.parse_args()
    chunks = load_chunks()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    ids = {c["source_id"] for c in chunks}
    if len(ids) != len(chunks) or not all(g in ids for _, g in questions):
        raise ValueError("Invalid or duplicate gold section")
    from sentence_transformers import SentenceTransformer
    from pymilvus import MilvusClient, DataType
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = SentenceTransformer(str(args.model_dir) if args.model_dir else MODEL,
                               **({} if args.model_dir else {"revision": REVISION}))
    if args.search_existing:
        if not args.query or not (args.output / "summary.json").is_file():
            parser.error("--search-existing needs --query and a completed --output")
        db = args.output / "knowledge.db"
        if not db.exists():
            raise FileNotFoundError(db)
        client = MilvusClient(uri=str(db.resolve()))
        try:
            client.load_collection('medical_sections')
            vector = model.encode([PREFIX + args.query], normalize_embeddings=True).tolist()
            matches = client.search("medical_sections", vector, limit=3,
                                    search_params={"metric_type": "COSINE", "params": {}},
                                    output_fields=["title", "section", "source", "text", "version"])
            print(json.dumps([dict(h["entity"], cosine=float(h["distance"])) for h in matches[0]],
                             ensure_ascii=False, indent=2))
        finally:
            client.close()
        return
    texts = [f"{c['title']}｜{c['section']}\n{c['text']}" for c in chunks]
    if any(len(t) > model.max_seq_length for t in model.tokenizer(texts)["input_ids"]):
        raise ValueError("Section exceeds encoder token budget")
    embeddings = model.encode(texts, normalize_embeddings=True)
    queries = [q for q, _ in questions] + ([args.query] if args.query else [])
    query_vectors = model.encode([PREFIX + q for q in queries], normalize_embeddings=True)
    if embeddings.shape != (len(chunks), 1024):
        raise ValueError("Unexpected embedding dimensions")
    args.output.mkdir(parents=True, exist_ok=True)
    db = args.output / "knowledge.db"
    if db.exists() or (args.output / "results.json").exists():
        raise FileExistsError("Specify a new --output directory for a new build")
    client = MilvusClient(uri=str(db.resolve()))
    name = "medical_sections"
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
        for key in ("source_id", "title", "section", "text", "source", "version"):
            schema.add_field(key, DataType.VARCHAR, max_length=4096)
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection(name, schema=schema, index_params=indexes)
        client.insert(name, [dict(c, vector=v.tolist()) for c, v in zip(chunks, embeddings)])
        client.flush(name)
    finally:
        client.close()
    client = MilvusClient(uri=str(db.resolve()))
    try:
        hits = client.search(name, query_vectors.tolist(), limit=3,
                             search_params={"metric_type": "COSINE", "params": {}},
                             output_fields=["source_id", "title", "section", "text", "source", "version"])
    finally:
        client.close()
    results = []
    for i, (q, matches) in enumerate(zip(queries, hits, strict=True)):
        gold = questions[i][1] if i < len(questions) else None
        ranked = [dict(hit["entity"], cosine=float(hit["distance"])) for hit in matches]
        rank = next((n for n, hit in enumerate(ranked, 1) if hit["source_id"] == gold), None)
        results.append(dict(question=q, gold=gold, rank=rank, hits=ranked))
    labeled = results[:len(questions)]
    summary = {"questions": len(labeled), "chunks": len(chunks), "model": MODEL,
               "revision_expected": REVISION, "device": str(model.device),
               "corpus_sha256": hashlib.sha256(json.dumps(chunks, ensure_ascii=False).encode()).hexdigest(),
               "hit_at_1": sum(r["rank"] == 1 for r in labeled)/len(labeled),
               "hit_at_3": sum(r["rank"] is not None for r in labeled)/len(labeled),
               "mrr_at_3": sum(1/r["rank"] for r in labeled if r["rank"])/len(labeled)}
    for file, value in [("results.json", results), ("summary.json", summary)]:
        (args.output / file).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
