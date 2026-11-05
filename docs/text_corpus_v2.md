> 历史阶段记录。当前实现与启动入口请以项目 README 和 docs/reproduction.md 为准。

# 文本知识库 v2

本轮保持模型、base/v2生成参数、Top-3及索引类型不变，仅扩充语料并建立版本化入库流程。6篇项目摘要、24个章节，不是原文全文库。新增WHO减钠、身体活动、健康饮食，来源与日期见各文件。旧的3篇摘要原样保留。

## 切分

按章节切分；超过300个Unicode字符的章节再以300字符、40字符重叠滑窗划分，窗口不跨章节。字符数不是token数：编码前用真实BGE tokenizer检查标题和文本总长度，超过模型上限立即报错，不静默截断。当前短摘要均不足300字符，因此没有人为增加块数；长文路径已用单元测试验证覆盖和重叠。每块保留文档ID、来源URL、页面日期、文件哈希、片段哈希及章节内字符起止位置（左闭右开）。这些位置基于去除首尾空白后的章节正文。

build_text_index输出chunks.jsonl和summary.json；实际索引写knowledge.db。summary的status=prepared表示只准备文本或构建未结束；status=indexed才表示通过索引读回检查。Milvus保留生成所需字段，完整审计信息在chunks.jsonl中，以source_id连接。独立目录拒绝覆盖，可切回v1。

## 云端步骤

先等待网页问答结束；停止API进程避免同时运行GPU任务。用已有检索环境的绝对Python路径构建，无需重装依赖：

```bash
cd /mnt/workspace/RAGDemo
/mnt/workspace/envs/rag-retrieval-v1/bin/python -m scripts.build_text_index \
  --corpus experiments/retrieval_v2/corpus \
  --model-dir /mnt/workspace/models/bge-large-zh-v1.5 \
  --output outputs/retrieval_v2
cat outputs/retrieval_v2/summary.json
```

在已验证LLaMA-Factory的生成环境运行（无须激活检索环境）：

```bash
python -m scripts.ask_rag \
  --questions experiments/retrieval_v2/questions.json \
  --db outputs/retrieval_v2/knowledge.db \
  --output outputs/rag_corpus_v2_validation
```

审核generation/answers.jsonl，重点看新增知识能否回答、旧知识是否仍召回、资料外题是否拒答。新题是面向当前语料设计的开发检查，不是外部盲测。v1食盐克数题现在已可回答，不应计为拒答失败；旧标签不修改，分别报告。

## 网页切换

仅在v2回答审核后：停止旧API，设置 `export RAG_DB_PATH=/mnt/workspace/RAGDemo/outputs/retrieval_v2/knowledge.db` 再按原方式启动uvicorn单worker。回滚时unset RAG_DB_PATH并重启即可。网页健康检查只检查路径，不证明知识库质量。页面的3篇12片段说明改为提示当前版本需查看构建记录，防止切换后误报。

## 当前局限

仍是小规模摘要，字符滑窗可能切断句子；先选用章节边界并检查token长度，后续长文语料可对比分句/分段策略。未引入PDF、LangChain、vLLM、Redis。本轮不把开发题通过率描述为医学准确率。
