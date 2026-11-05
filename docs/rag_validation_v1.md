# 新问题检索与生成验证

沿用已有 Milvus Lite 的12个摘要片段。validation_v1.json 为项目作者新增的12道验证题：改写4题、跨片段4题、摘要无法回答4题。它不是外部盲测或临床认证测试，使用之后不能继续称作未见验证题；若据此调参，应另备后续测试题。

原始 Qwen2 + v2 提示词固定。审核标注仅供人工对照，retrieve_questions 只读取 question；模型收到真实检索到的前三个片段，不注入 expected_sources 或 review。跨片段题未召回全部证据时，分别记录检索遗漏和生成对已提供证据的处理。

ask_rag 先用检索环境执行 BGE + 已有 Milvus 查询，进程退出释放资源后，用当前 Python 环境执行 Qwen 生成。文件保留两阶段结果，不重新构建索引。每次命令重新加载模型，是命令行按需问答，尚非常驻 API，不用于服务延迟报告。

在已能运行 LLaMA-Factory 推理的环境、RAGDemo 根目录执行：

```bash
python -m scripts.ask_rag --questions experiments/retrieval_v1/validation_v1.json --output outputs/rag_validation_v1
python -m scripts.ask_rag --query "没有头痛就能排除血压升高吗？" --output outputs/rag_query_v1
```

检索 Python 默认为 /mnt/workspace/envs/rag-retrieval-v1/bin/python，可用 --retrieval-python 指定。--dry-run 仅打印两条命令，不加载模型、不创建目录。失败保留已产生文件，请使用新目录重试，或直接用 generate_rag_demo 的 --retrieval 复用成功的检索结果。

人工审核每题：召回所需片段是否完整、是否直接回答问题、陈述是否由所引片段支持、是否存在无依据扩展、应拒答时是否拒答。不能把有引用编号直接算成正确答案，也不能把余弦相似度当作已校准的拒答阈值。
