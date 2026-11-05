"""Chinese RAG demo API; run one uvicorn worker only."""

import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag_service import JobService, BusyError, ROOT
from app.answer_cache import AnswerCache

settings = get_settings()
app = FastAPI(
    title=settings.app_title,
    version="0.2.0",
    description="BGE + Milvus Lite + Qwen2 医学知识科普演示；单任务执行。",
)


service = JobService(
    ROOT / 'outputs/web_demo',
    os.getenv('RAG_RETRIEVAL_PYTHON', '/mnt/workspace/envs/rag-retrieval-v1/bin/python'),
    os.getenv('RAG_GENERATION_PYTHON', sys.executable),
    os.getenv('RAG_BGE_PATH', '/mnt/workspace/models/bge-large-zh-v1.5'),
    os.getenv('RAG_QWEN_PATH', '/mnt/workspace/models/Qwen2-7B-Instruct'),
    os.getenv('RAG_DB_PATH', str(ROOT / 'outputs/retrieval_v1/knowledge.db')),
    backend=os.getenv('RAG_BACKEND','huggingface'),
    vllm_url=os.getenv('RAG_VLLM_URL','http://127.0.0.1:8001/v1'),
    vllm_model=os.getenv('RAG_VLLM_MODEL','medical-qwen2'),
    retrieval_url=os.getenv('RAG_RETRIEVAL_URL'),
    answer_cache=AnswerCache.from_url(os.environ['RAG_REDIS_URL'],int(os.getenv('RAG_CACHE_TTL','3600')))
    if os.getenv('RAG_REDIS_URL') else None)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name('index.html'))


@app.get('/health')
def health():
    missing = service.missing_paths()
    return dict(status='ok', stage='rag_web_demo', paths_ready=not missing,
                missing_paths=missing, backend=service.backend,
                retrieval_backend='resident_http' if service.retrieval_url else 'subprocess',
                answer_cache_enabled=service.answer_cache is not None,
                message='仅检查本地路径；vLLM连接和推理需单独验证。' if service.backend=='vllm'
                else '路径检查不代表模型推理已通过；每次请求重新加载模型。')


@app.post('/api/jobs', status_code=202)
def submit(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(422, '请输入问题。')
    if service.missing_paths():
        raise HTTPException(503, '模型、数据库或Python环境路径缺失，请检查 /health。')
    try:
        return service.submit(question)
    except BusyError as error:
        raise HTTPException(409, str(error)) from error


@app.get('/api/jobs/{job_id}')
def result(job_id: str):
    job = service.get(job_id)
    if job is None:
        raise HTTPException(404, '任务不存在或服务已重启；已保存结果可在 outputs/web_demo 查看。')
    return job
