"""Versioned text corpus ingestion; no QA data or PDF/image parsing."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split_text(text, size=300, overlap=40):
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < chunk size')
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return spans


def read_document(path):
    parts = re.split(r'^## ', path.read_text(encoding='utf-8-sig'), flags=re.M)
    header = parts[0].strip().splitlines()
    if len(header) != 3 or not header[0].startswith('# ') or not header[1].startswith('来源：') or not header[2].startswith('版本：'):
        raise ValueError(f'{path}: expected title, source URL and version date')
    title, source, version = header[0][2:], header[1][3:], header[2][3:]
    parsed = urlparse(source)
    if not title.strip() or parsed.scheme not in ('https','http') or not parsed.netloc:
        raise ValueError(f'{path}: invalid title/source')
    date.fromisoformat(version)
    sections = []
    for part in parts[1:]:
        section, body = part.strip().split('\n',1)
        body = body.strip()
        if not section.strip() or not body or section in [s for s,_ in sections]:
            raise ValueError(f'{path}: empty or duplicate section')
        sections.append((section,body))
    if not sections:
        raise ValueError(f'{path}: no sections')
    return title,source,version,sections


def prepare(corpus, size, overlap):
    chunks, documents = [], []
    paths = sorted(p for p in corpus.iterdir() if p.suffix.lower() in ('.md','.txt') and p.name != 'README.md')
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Need documents with unique filename stems')
    for path in paths:
        title,source,version,sections = read_document(path)
        file_hash = digest(path.read_bytes())
        documents.append(dict(file=path.name,title=title,source=source,version=version,sha256=file_hash))
        for section,body in sections:
            for n,(start,end) in enumerate(split_text(body,size,overlap),1):
                chunks.append(dict(id=len(chunks),source_id=f'{path.stem}/{section}/{n:04d}',
                    document_id=path.stem,title=title,section=section,text=body[start:end],source=source,
                    version=version,section_char_start=start,section_char_end=end,
                    document_sha256=file_hash,content_sha256=digest(body[start:end].encode())))
    return chunks,documents


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=ROOT/'experiments/retrieval_v2/corpus')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--chunk-size',type=int,default=300)
    parser.add_argument('--overlap',type=int,default=40)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; previous experiments are preserved')
    if not args.prepare_only and (not args.model_dir or not (args.model_dir/'config.json').is_file()):
        parser.error('Building the index requires --model-dir with a local BGE snapshot')
    chunks,documents=prepare(args.corpus,args.chunk_size,args.overlap)
    manifest=dict(documents=documents,document_count=len(documents),chunks=len(chunks),
        chunking=dict(unit='Unicode characters',size=args.chunk_size,overlap=args.overlap,
                      boundary='within sections; fixed windows; offsets in stripped section text'),
        corpus_sha256=digest(json.dumps(chunks,ensure_ascii=False,sort_keys=True).encode()),
        source_content='Project-authored summaries, not full source articles',status='prepared')
    args.output.mkdir(parents=True)
    def save_manifest():
        (args.output/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'chunks.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in chunks),encoding='utf-8')
    save_manifest()
    if not args.prepare_only:
        from sentence_transformers import SentenceTransformer
        from pymilvus import MilvusClient,DataType
        model=SentenceTransformer(str(args.model_dir))
        texts=[f"{c['title']}｜{c['section']}\n{c['text']}" for c in chunks]
        lengths=[len(ids) for ids in model.tokenizer(texts,truncation=False)['input_ids']]
        if max(lengths)>model.max_seq_length:
            raise ValueError('Encoder token limit exceeded: use a smaller --chunk-size; no silent truncation')
        vectors=model.encode(texts,normalize_embeddings=True)
        if vectors.shape!=(len(chunks),1024):
            raise ValueError('Expected BGE-Large embeddings with dimension 1024')
        fields=('source_id','title','section','text','source','version')
        if any(len(c[k].encode())>4096 for c in chunks for k in fields):
            raise ValueError('Milvus text field exceeds 4096 UTF-8 bytes')
        client=MilvusClient(uri=str((args.output/'knowledge.db').resolve()))
        try:
            schema=client.create_schema(auto_id=False,enable_dynamic_field=False)
            schema.add_field('id',DataType.INT64,is_primary=True)
            schema.add_field('vector',DataType.FLOAT_VECTOR,dim=1024)
            for key in fields:schema.add_field(key,DataType.VARCHAR,max_length=4096)
            indexes=client.prepare_index_params()
            indexes.add_index(field_name='vector',index_type='FLAT',metric_type='COSINE')
            client.create_collection('medical_sections',schema=schema,index_params=indexes)
            client.insert('medical_sections',[dict(id=c['id'],vector=v.tolist(),**{k:c[k] for k in fields}) for c,v in zip(chunks,vectors,strict=True)])
            client.flush('medical_sections')
            client.load_collection('medical_sections')
            # Read after insertion, before marking the index ready.
            if not client.search('medical_sections',[vectors[0].tolist()],limit=1,output_fields=['source_id'])[0]:
                raise RuntimeError('Index verification returned no hits')
        finally:client.close()
        manifest.update(status='indexed',model_dir=str(args.model_dir.resolve()),device=str(model.device),
                        embedding_dimension=1024,max_chunk_tokens=max(lengths),encoder_limit=model.max_seq_length)
        save_manifest()
    print(json.dumps({k:v for k,v in manifest.items() if k!='documents'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
