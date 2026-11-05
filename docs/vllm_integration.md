# 常驻vLLM接入

用户云端vLLM 0.27.1已成功返回模型列表和实际生成。A10，BF16，2048上下文，单并发，gpu-memory-utilization 0.85，enforce-eager。运行端口8001仅绑定127.0.0.1。当前启用的是eager模式，不能将后续速度当作CUDA Graph优化结果。

新增generate_rag_vllm通过标准库HTTP调用，不在客户端加载torch/Qwen。复用同一个SYSTEM和prepare_prompt(v2)，显式设置temperature=0、top_p=1、max_tokens=512、seed=42、repetition_penalty=1.0。保存服务模型信息、响应fingerprint、usage和每请求耗时。后端、聊天模板及默认参数可能影响结果，不能宣称与HF逐token一致。

## 先做固定证据回归

```bash
cd /mnt/workspace/RAGDemo
python -m scripts.generate_rag_vllm \
  --retrieval outputs/rag_corpus_v2_validation/retrieval/results.json \
  --output outputs/rag_vllm_corpus_v2_v1
```

这次只访问vLLM，不启动第二份Qwen、不运行BGE。报告耗时仅为HTTP生成请求，不含检索、模型启动或网页轮询，不是端到端性能。失败保留已生成答案，新实验另用目录。

## 回归确认后切换网页

上传app/main.py、app/rag_service.py、app/index.html和scripts/retrieve_questions.py、scripts/generate_rag_vllm.py。先等旧任务结束、停止旧8000端口API，保留8001 vLLM终端运行。在启动网页的终端设置：

```bash
export RAG_BACKEND=vllm
export RAG_VLLM_URL=http://127.0.0.1:8001/v1
export RAG_DB_PATH=/mnt/workspace/RAGDemo/outputs/retrieval_v2/knowledge.db
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

/health应显示backend=vllm。后端没有自动降级到HF，避免失败时再加载模型。网页每次检索仍会启动BGE子进程，但强制CPU，不与vLLM争抢GPU；BGE尚未常驻，因此完整链路仍存在加载开销。先实测再决定常驻检索优化。导出环境变量在同一终端生效，不放入现有严格校验.env。

保持单worker、不同时启动训练或旧HF生成。回滚到HF前先停止vLLM释放GPU，再unset RAG_BACKEND重启API。尚未加入Redis、LangChain。图形界面的延迟不当成纯生成耗时。
