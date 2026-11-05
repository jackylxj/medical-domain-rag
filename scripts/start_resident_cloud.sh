#!/usr/bin/env bash
# Run from an SSH shell or the ModelScope terminal. Existing listeners are left alone.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG="$ROOT/outputs/resident_services"
mkdir -p "$LOG"
# This image ships CUDA forward-compatibility libraries; SSH does not inherit the notebook environment.
if [ -d /usr/local/cuda/compat ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
listening() { ss -ltnH "sport = :$1" | grep -q .; }
if [ "${RAG_ENABLE_CACHE:-0}" = 1 ]; then
  if ! listening 6379; then
    test -s "$ROOT/configs/redis-demo.conf"
    nohup redis-server "$ROOT/configs/redis-demo.conf" --bind 127.0.0.1 > "$LOG/redis.log" 2>&1 < /dev/null &
    echo $! > "$LOG/redis.pid"
  fi
  export RAG_REDIS_URL=redis://127.0.0.1:6379/0
  export RAG_CACHE_TTL=3600
fi
if ! listening 8001; then
  nohup /usr/local/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/workspace/models/Qwen2-7B-Instruct --served-model-name medical-qwen2 \
    --host 127.0.0.1 --port 8001 --dtype bfloat16 --max-model-len 2048 \
    --max-num-seqs 1 --gpu-memory-utilization 0.85 --enforce-eager \
    > "$LOG/vllm.log" 2>&1 < /dev/null &
  echo $! > "$LOG/vllm.pid"
fi
if ! listening 8002; then
  nohup /mnt/workspace/envs/rag-retrieval-v1/bin/python -m scripts.serve_retrieval \
    --model-dir /mnt/workspace/models/bge-large-zh-v1.5 \
    --db "$ROOT/outputs/retrieval_v2/knowledge.db" --port 8002 \
    > "$LOG/retrieval.log" 2>&1 < /dev/null &
  echo $! > "$LOG/retrieval.pid"
fi
if ! listening 8000; then
  export RAG_BACKEND=vllm RAG_VLLM_URL=http://127.0.0.1:8001/v1
  export RAG_DB_PATH="$ROOT/outputs/retrieval_v2/knowledge.db"
  export RAG_RETRIEVAL_URL=http://127.0.0.1:8002
  nohup /usr/local/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 \
    > "$LOG/web.log" 2>&1 < /dev/null &
  echo $! > "$LOG/web.pid"
fi
echo "Launch requested. Wait for startup logs and check health; do not rerun while startup is pending."
