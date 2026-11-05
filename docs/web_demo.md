# 中文网页问答演示

应用入口 app/main.py；页面 app/index.html；任务执行 app/rag_service.py。复用 scripts/retrieve_questions.py 和 scripts/generate_rag_demo.py。使用 base Qwen2、v2 提示词、BGE + Milvus Lite；未接入 LangChain、vLLM、Redis。每次重新加载模型，不能把本轮耗时报告为常驻服务延迟。

## 启动

在已通过 Qwen 推理的云端 Python 环境运行，不要切到 rag-retrieval-v1：

```bash
cd /mnt/workspace/RAGDemo
python -c "import fastapi, uvicorn, pydantic_settings; from llamafactory.chat import ChatModel; print('imports OK')"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

若 API 库缺失，仅补齐缺失项，不重新安装训练或检索依赖。检索环境默认 /mnt/workspace/envs/rag-retrieval-v1/bin/python，生成环境默认启动API的Python。通过shell export RAG_RETRIEVAL_PYTHON、RAG_GENERATION_PYTHON、RAG_BGE_PATH、RAG_QWEN_PATH、RAG_DB_PATH 可覆盖路径（这些配置通过进程环境读取，不写进现有严格校验的 .env）。

必须单 worker、不使用 --reload，且不要同时用命令行启动其他GPU问答。只通过平台提供的受控端口访问用于演示；此版本没有用户认证、跨进程任务锁或多用户隔离。启动成功不代表模型验证成功。

第二个终端检查：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/jobs -H 'Content-Type: application/json' -d '{"question":"没有头痛为什么也要测血压？"}'
```

POST返回202和id。用 GET /api/jobs/{id} 查询状态 queued/retrieving/generating/completed/failed。已有任务返回409；路径缺失503；空白输入422。页面自动轮询并展示引用片段。平台页面端口代理需将请求转发到此8000端口；浏览器根URL使用末尾斜杠，以便相对API路径正确解析。

## 日志和结果

outputs/web_demo/{id}/ 下保存 retrieving.log、generating.log、job.json、检索结果和生成结果；失败另有 error.txt。每阶段超时900秒。重启丢失内存任务索引，但已有文件保留；不要在任务运行中重启，退出前等任务结束。页面刷新在同一标签可恢复轮询。内存仅保留最近100个任务。

## 验证范围

本地通过模拟子进程验证任务互斥、结果读取和失败释放；真实BGE检索和Qwen生成已分别验证，但网页到GPU的完整链路仍需云端执行确认。
