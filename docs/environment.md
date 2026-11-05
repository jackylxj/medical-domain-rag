# 第一步环境与验收记录

日期：2026-11-03（Asia/Shanghai）。

## 已检查环境

- 项目目录：D:/dev/pythonProject/RAGDemo
- 复用项目虚拟环境：.venv/Scripts/python.exe
- 项目Python：3.12.15
- 系统Python：3.13.12，未修改
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- nvidia-smi报告显存：8188 MiB
- NVIDIA驱动：581.57
- 没有验证CUDA/PyTorch兼容性，没有安装GPU依赖。
- 原有.idea和.venv保留。未初始化Git仓库或推送GitHub。

## 安装范围

只在项目.venv安装FastAPI、Uvicorn、pydantic-settings及其依赖。
直接依赖：FastAPI 0.162.0、Uvicorn 0.54.0、pydantic-settings 2.16.2。
完整解析版本保存在requirements.txt。pip check通过。

## 实际验收

临时使用空闲本机端口启动真实Uvicorn进程：

- GET /：200，正确返回被环境变量覆盖的应用标题。
- GET /health：200，status=ok且rag_ready=false。
- GET /openapi.json：200，包含/health定义。
- GET /docs：200。
- APP_PORT=70000：配置校验拒绝启动。
- 移除临时覆盖后：从项目.env读取默认中文标题与8000端口。
- 验收后临时服务已停止，无后台问答进程遗留。

未运行：BGE、Milvus、Qwen、LoRA、Redis；没有使用ModelScope密钥。
健康检查只报告API存活，不代表RAG可用。

## 下一阶段建议

等待用户指示后完成文本解析与分块小闭环，先把样本文档处理成带来源的JSONL。
8GB GPU下的7B推理与LoRA训练将独立评估，不预先承诺本机可运行完整训练/服务组合。
