# 基于领域适配大模型与 RAG 的医疗问答系统

**Medical Question Answering with a Domain-Adapted LLM and RAG**

基于 Qwen2-7B-Instruct 的中文医学知识问答项目，覆盖医疗 QA 数据处理、LoRA 领域适配、知识库检索、来源引用和 Web 服务。通过固定数据划分衡量微调效果，通过可追溯证据组织回答，通过常驻服务与 Redis 缓存改善交互体验。

## 项目成果

- **30,000 条医疗 QA 微调**：使用 LLaMA-Factory 和 LoRA 完成单卡训练，在同一 500 题保留测试集上，字符级 ROUGE-L F1 从 **16.81 提升至 26.91（+10.10 分）**。
- **可追溯的 RAG 问答**：实现 Markdown/TXT 解析、章节与字符窗口切分、BGE-Large 向量化、Milvus Lite 检索、回答引用与来源展示。
- **可交互的部署链路**：FastAPI、CPU 常驻检索、GPU vLLM 推理；Redis 支持完全相同问题的答案复用、TTL 和故障回退。
- **有记录的工程优化**：五次串行服务测试中，首个任务 7.97 秒，其余 0.75–1.49 秒；另三组缓存配对测试中，命中任务耗时 **38.3–52.8 毫秒**。

ROUGE-L 为逐题字符 LCS F1 均值，衡量文本重合度；延迟为服务端任务耗时，不含服务启动、浏览器轮询及外部网络。完整条件见[实验报告](docs/experiment_report.md)。

## 系统架构

![云端发布验收的真实回答快照](docs/demo.png)

[离线查看这组真实运行样例](docs/demo.html)（下载后用浏览器打开）。截图展示保存的验收回答；运行 Web 服务后可实时提问。

```mermaid
flowchart LR
  D[Markdown / TXT 资料摘要] --> C[章节与字符窗口切分]
  C --> E[BGE-large-zh-v1.5 · 1024维]
  E --> M[Milvus Lite · FLAT / COSINE]
  U[用户问题] --> A[FastAPI 任务接口]
  A --> K{Redis 完全匹配缓存}
  K -->|命中| O[回答与来源展示]
  K -->|未命中或不可用| R[CPU 常驻 BGE 检索]
  M --> R
  R --> P[Top-3 证据 + RAG Prompt v2]
  P --> V[vLLM · Qwen2-7B-Instruct]
  V --> O
  V -->|写入答案与证据| K
  T[Huatuo26M-Lite · 固定划分] --> L[LLaMA-Factory · LoRA]
  L --> Q[原始模型 / LoRA 离线对照]
```

在线演示采用 **原始 Qwen2-7B-Instruct + RAG**；30k LoRA 作为领域适配与离线对照模型。该选择依据开发题中的回答完整性与引用表现，微调产物和评估流程均保留。

## 技术实现

| 模块 | 实现 |
|---|---|
| 数据与微调 | 固定种子、规则去重、30k / 500 / 500 划分；LLaMA-Factory、LoRA、BF16 |
| 文本处理 | Markdown / TXT；按章节切分，长章节 300 字符窗口、40 字符重叠；来源、位置与哈希 |
| 向量检索 | SentenceTransformers、BGE-large-zh-v1.5、1024 维；Milvus Lite、COSINE、Top-3 |
| 生成 | Qwen2-7B-Instruct、Prompt v2、vLLM OpenAI-compatible API |
| Web | FastAPI、任务状态轮询、中文网页、答案与检索证据展示 |
| 缓存 | Redis、问题与版本上下文哈希键、TTL、故障回退 |
| 验证 | 字符级 ROUGE-L F1、Hit@K / MRR、逐题证据核对、串行任务计时、自动化测试 |

当前演示知识库为 **6 篇项目编写的公开资料摘要、24 个片段**。系统面向医学知识科普与技术实验，不用于个人诊断。

## 快速运行

已验证环境：Linux、Python 3.12、NVIDIA A10。训练、检索与生成采用分离环境。首次运行先按[复现指南](docs/reproduction.md)准备依赖、模型和索引；启动脚本默认使用 /mnt/workspace 布局。

```bash
cd /mnt/workspace/RAGDemo
RAG_ENABLE_CACHE=1 bash scripts/start_resident_cloud.sh
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8002/health
curl -fsS http://127.0.0.1:8000/health
```

通过服务器端口入口打开 8000/。提问“没有头痛为什么仍需要实际测量血压？”查看回答及支持片段；提问“资料给出的哮喘吸入药商品名和每次剂量是多少？”查看资料不足处理；再次提交完全相同的问题查看缓存复用。

```bash
curl -sS http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"question":"没有头痛为什么仍需要实际测量血压？"}'
# 使用返回的 id 查询：GET /api/jobs/{id}
```

本地源码测试无需下载模型：

```bash
python -m unittest discover -s tests
```

## 阅读导航

| 内容 | 入口 |
|---|---|
| 实验设计、结果与模型选择 | [实验报告](docs/experiment_report.md) |
| 环境、数据、训练与评估 | [复现指南](docs/reproduction.md) |
| 解析与入库 | [文本知识库](docs/text_corpus_v2.md) |
| 常驻服务 | [部署说明](docs/resident_services.md) |
| 缓存键、TTL 与回退 | [Redis 缓存](docs/answer_cache.md) |
| 冻结版本与验收 | [发布说明](docs/release_v1.md) |
| 已有诊断实验档案 | [实验索引](docs/experiments/README.md) |
| 数据与模型来源 | [来源说明](docs/sources_and_release.md) |

代码在 app/、scripts/，配置在 configs/，复现材料在 experiments/，测试在 tests/。模型权重、原始医疗 QA、运行输出和凭据单独保存；仓库提供源码、配置、摘要语料与评估汇总。
