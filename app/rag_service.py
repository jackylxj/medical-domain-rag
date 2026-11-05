"""Single-worker demo jobs using isolated retrieval and inference subprocesses."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import Request, urlopen
from uuid import uuid4
import hashlib
from app.answer_cache import cache_key

ROOT = Path(__file__).resolve().parents[1]


class BusyError(Exception):
    pass


class JobService:
    def __init__(self, output, retrieval_python, generation_python, bge, qwen, db, runner=subprocess.run,
                 backend='huggingface', vllm_url='http://127.0.0.1:8001/v1', vllm_model='medical-qwen2',
                 retrieval_url=None, answer_cache=None):
        if backend not in ('huggingface','vllm'):
            raise ValueError('RAG_BACKEND must be huggingface or vllm')
        self.backend,self.vllm_url,self.vllm_model=backend,vllm_url,vllm_model
        self.retrieval_url = retrieval_url
        self.answer_cache = answer_cache
        self.deployment = uuid4().hex
        self.output = Path(output)
        self.retrieval_python, self.generation_python = str(retrieval_python), str(generation_python)
        self.bge, self.qwen, self.db = str(bge), str(qwen), str(db)
        self.runner = runner
        self.jobs = {}
        self.lock = threading.Lock()
        self.active = False

    def missing_paths(self):
        paths = [self.retrieval_python, self.generation_python,
                 str(Path(self.bge) / 'config.json'), str(Path(self.qwen) / 'config.json'), self.db]
        return [p for p in paths if not Path(p).exists()]

    def get(self, job_id):
        with self.lock:
            return deepcopy(self.jobs.get(job_id))

    def _cache_context(self):
        if self.backend != 'vllm' or not self.retrieval_url:
            raise ValueError('Answer cache requires resident retrieval and vLLM')
        with urlopen(self.retrieval_url.rstrip('/')+'/health',timeout=2) as response:
            health = json.load(response)
        if health.get('status') != 'ready' or not health.get('instance_id') or not health.get('database_build_summary_sha256'):
            raise ValueError('Retrieval version unavailable')
        if Path(health['database']).resolve() != Path(self.db).resolve():
            raise ValueError('Retrieval database mismatch')
        files = ['scripts/generate_rag_demo.py','scripts/generate_rag_vllm.py','scripts/infer_smoke.py']
        return dict(deployment=self.deployment, backend=self.backend, model=self.vllm_model,
                    endpoint=self.vllm_url, retrieval={k:v for k,v in health.items() if k not in ('startup_timings','status')},
                    source_digests={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def submit(self, question):
        with self.lock:
            if self.active:
                raise BusyError('已有任务运行，请等待完成后再提问。')
            self.active = True
            job_id = uuid4().hex
            self.jobs[job_id] = dict(id=job_id, question=question, status='queued', result=None)
            if len(self.jobs) > 100:
                del self.jobs[next(iter(self.jobs))]
        try:
            threading.Thread(target=self._run, args=(job_id, question), daemon=True).start()
        except Exception:
            with self.lock:
                self.active = False
                self.jobs.pop(job_id, None)
            raise
        return dict(id=job_id, question=question, status='queued', result=None)

    def _run(self, job_id, question):
        started = time.monotonic()
        directory = self.output / job_id
        timings = {}
        cache_state = {'status':'disabled'}
        key = None
        context = None
        try:
            directory.mkdir(parents=True, exist_ok=False)
            self._update(job_id, backend=self.backend)
            if self.answer_cache:
                cache_started = time.monotonic()
                try:
                    context = self._cache_context()
                    key = cache_key(question,context)
                    entry = self.answer_cache.get(key)
                    cache_state = dict(status='hit' if entry else 'miss',key=key)
                    if entry:
                        result = deepcopy(entry['result'])
                        for field in ('request_seconds','usage'):
                            result.pop(field,None)
                        result['generation_executed'] = False
                        cache_state['source_job_id'] = entry['source_job_id']
                        (directory/'cached_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                        self._update(job_id,result=result)
                        final_status = 'completed'
                        return
                except Exception as error:
                    cache_state = dict(status='bypass_error',error_type=type(error).__name__)
                    key = None
                finally:
                    timings['cache_lookup_seconds'] = time.monotonic()-cache_started
            commands = [
                ('retrieving', [self.retrieval_python, '-m', 'scripts.retrieve_questions',
                 '--query', question, '--model-dir', self.bge, '--db', self.db,
                 '--output', str(directory / 'retrieval')]),
                ('generating', [self.generation_python, '-m', 'scripts.generate_rag_demo',
                 '--mode', 'base', '--prompt-version', 'v2', '--model', self.qwen,
                 '--retrieval', str(directory / 'retrieval/results.json'),
                 '--output', str(directory / 'generation')])]
            if self.backend == 'vllm':
                commands[0][1].extend(['--device','cpu'])
                commands[1]=('generating',[self.generation_python,'-m','scripts.generate_rag_vllm',
                    '--base-url',self.vllm_url,'--model',self.vllm_model,
                    '--retrieval',str(directory/'retrieval/results.json'),
                    '--output',str(directory/'generation')])
            self._update(job_id, backend=self.backend)
            for phase, command in commands:
                self._update(job_id, status=phase)
                phase_start = time.monotonic()
                if phase == 'retrieving' and self.retrieval_url:
                    request = Request(self.retrieval_url.rstrip('/') + '/search',
                        data=json.dumps({'question': question}).encode('utf-8'),
                        headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=180) as response:
                        retrieved = json.load(response)
                    if retrieved['question'] != question or Path(retrieved['metadata']['database']).resolve() != Path(self.db).resolve():
                        raise ValueError('Resident retrieval question or database mismatch')
                    folder = directory / 'retrieval'
                    folder.mkdir()
                    (folder / 'results.json').write_text(json.dumps([
                        dict(question=question, hits=retrieved['hits'])], ensure_ascii=False), encoding='utf-8')
                    (folder / 'metadata.json').write_text(json.dumps(retrieved['metadata'], ensure_ascii=False, indent=2), encoding='utf-8')
                    timings.update(retrieved['timings'])
                    if key and retrieved['metadata'] != context.get('retrieval'):
                        cache_state['status'] = 'bypass_version_change'
                        key = None
                else:
                    with (directory / f'{phase}.log').open('w', encoding='utf-8') as log:
                        self.runner(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    check=True, timeout=900)
                timings[phase + '_wall_seconds'] = time.monotonic() - phase_start
            lines = (directory / 'generation/answers.jsonl').read_text(encoding='utf-8').splitlines()
            result = json.loads(next(line for line in lines if line.strip()))
            if not isinstance(result.get('answer'), str) or not result['answer'].strip():
                raise ValueError('Empty answer')
            if 'request_seconds' in result:
                timings['vllm_request_seconds'] = result['request_seconds']
            result['generation_executed'] = True
            if key and result.get('finish_reason') == 'stop':
                cache_started = time.monotonic()
                try:
                    self.answer_cache.put(key,dict(result=result,source_job_id=job_id))
                    cache_state['stored'] = True
                except Exception as error:
                    cache_state.update(stored=False,write_error_type=type(error).__name__)
                timings['cache_write_seconds'] = time.monotonic()-cache_started
            self._update(job_id, result=result)
            final_status = 'completed'
        except Exception as error:
            final_status = 'failed'
            self._update(job_id, error='执行失败，请查看该任务目录中的日志。', error_type=type(error).__name__)
            if directory.is_dir():
                (directory / 'error.txt').write_text(repr(error), encoding='utf-8')
        finally:
            with self.lock:
                self.jobs[job_id].update(status=locals().get('final_status', 'failed'),
                                        timings=timings,
                                        cache=cache_state,
                                        retrieval_backend='resident_http' if self.retrieval_url else 'subprocess',
                                        elapsed_seconds=round(time.monotonic() - started, 4))
                self.active = False
                record = deepcopy(self.jobs[job_id])
            if directory.is_dir():
                (directory / 'job.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
