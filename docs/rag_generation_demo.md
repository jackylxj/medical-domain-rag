# 检索结果接入 LoRA 生成

检索阶段输出 outputs/retrieval_v1/results.json，生成阶段在已验证的 LLaMA-Factory 环境中执行 scripts/generate_rag_demo.py。通过文件把两个独立环境连接起来。

输入只使用用户问题和前三个检索片段。gold、rank 等开发题标注不进入提示词。片段保留章节、版本、URL，引用编号为每题独立的[1]到[3]。来源文本是项目编写的 WHO 资料摘要。

云端将本地 scripts/generate_rag_demo.py 复制到同一路径，沿用已有 scripts/infer_smoke.py 和 scripts/validate_data.py。

```bash
deactivate
cd /mnt/workspace/RAGDemo
python -c "from llamafactory.chat import ChatModel; import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
python -m scripts.generate_rag_demo --output outputs/rag_generation_v1 --dry-run
python -m scripts.generate_rag_demo --output outputs/rag_generation_v1
```

deactivate 用于退出当前 rag-retrieval-v1 虚拟环境，返回此前安装 LLaMA-Factory 的系统环境。如该实例没有恢复训练环境，先恢复已验证依赖。不要把检索 requirements 安装进训练环境。

成功后读取 summary.json 和 answers.jsonl。程序检查回答是否非空、是否达到长度上限、引用编号是否存在；引用是否支持具体陈述需人工检查。失败时保留部分输出并换新的 output 重试。

可在同一批问题上生成无检索对照，等第一个进程退出后执行：

```bash
python -m scripts.generate_rag_demo --without-rag --output outputs/lora_reference_questions_v1
```

两组使用同一30k adapter、system、贪心解码和512个新token上限，仅改变提供的证据上下文。此处没有人工标准答案，故不报告ROUGE或医学准确率。只有12道按资料设计的开发题，不能与Huatuo500题测试分数直接比较。

当前通过 LLaMA-Factory HF 后端与 Python 提示词组装完成离线 RAG 生成，LangChain、vLLM、在线 API 尚待集成。
