"""Loopback-only resident CPU BGE and Milvus retrieval service."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
from uuid import uuid4

from scripts.validate_data import sha256


class Runtime:
    def __init__(self, model_dir, db):
        model_dir, db = Path(model_dir).resolve(), Path(db).resolve()
        if not (model_dir / 'config.json').is_file() or not db.exists():
            raise FileNotFoundError('Local model config or existing database missing')
        started = time.perf_counter()
        print('Importing retrieval libraries...', flush=True)
        import torch
        from sentence_transformers import SentenceTransformer
        from pymilvus import MilvusClient
        imported = time.perf_counter()
        torch.set_num_threads(min(4, torch.get_num_threads()))
        print('Loading BGE on CPU once...', flush=True)
        self.model = SentenceTransformer(str(model_dir), device='cpu')
        loaded = time.perf_counter()
        print('Opening existing Milvus collection...', flush=True)
        self.client = MilvusClient(uri=str(db))
        self.client.load_collection('medical_sections')
        self.metadata = dict(model_dir=str(model_dir), database=str(db), device='cpu', top_k=3,
                             instance_id=uuid4().hex)
        manifest = db.parent / 'summary.json'
        if manifest.is_file():
            self.metadata['database_build_summary_sha256'] = sha256(manifest)
        self.startup = dict(import_seconds=imported-started, model_load_seconds=loaded-imported,
                            database_load_seconds=time.perf_counter()-loaded)

    def health(self):
        return dict(status='ready', **self.metadata, startup_timings=self.startup)

    def search(self, question):
        started = time.perf_counter()
        vectors = self.model.encode(['为这个句子生成表示以用于检索相关文章：' + question],
                                   normalize_embeddings=True, show_progress_bar=False)
        if vectors.shape != (1, 1024):
            raise ValueError('Expected 1024-dimensional BGE vectors')
        embedded = time.perf_counter()
        matches = self.client.search('medical_sections', vectors.tolist(), limit=3,
            search_params={'metric_type': 'COSINE', 'params': {}},
            output_fields=['source_id', 'title', 'section', 'text', 'source', 'version'])[0]
        ended = time.perf_counter()
        return dict(question=question,
                    hits=[dict(h['entity'], cosine=float(h['distance'])) for h in matches],
                    metadata=self.metadata,
                    timings=dict(embedding_seconds=embedded-started,
                                 vector_search_seconds=ended-embedded,
                                 retrieval_compute_seconds=ended-started))


def handler_for(runtime):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, runtime.health()) if self.path == '/health' else self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/search':
                return self.reply(404, {'error': 'Not found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid body size')
                body = json.loads(self.rfile.read(size))
                question = body.get('question') if isinstance(body, dict) else None
                if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
                    raise ValueError('Question must contain 1 to 1000 characters')
            except (ValueError, UnicodeError) as error:
                return self.reply(400, {'error': str(error)})
            try:
                self.reply(200, runtime.search(question.strip()))
            except Exception as error:
                print(f'Retrieval failed: {error!r}', flush=True)
                self.reply(500, {'error': 'Retrieval failed; inspect server log'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    runtime = Runtime(args.model_dir, args.db)
    try:
        with HTTPServer(('127.0.0.1', args.port), handler_for(runtime)) as server:
            server.timeout = 180
            print(f'READY http://127.0.0.1:{args.port} {json.dumps(runtime.health())}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        runtime.client.close()


if __name__ == '__main__':
    main()
