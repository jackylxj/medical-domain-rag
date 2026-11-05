# 单题错误拒答诊断

固定原始Qwen2、v2提示词与生成参数，只使用已有检索结果。A/B对比Top-3和Top-1；B/C对比原问法和中性问法；A/D补齐另一种上下文下的问法对照。E/F为资料外问题的Top-3/Top-1对照。全部六次调用在同一个模型进程内顺序执行，每题独立消息，不串联聊天历史。

中性问法改变了表达及问题意图的呈现方式，不是严格的单词级消融。选择Top-1是针对已知正确首片段的诊断，不是通用筛选算法；不能据此把生产检索永久改为Top-1。

云端更新scripts/diagnose_rag_refusal.py后，在已经能运行Qwen的环境执行：

```bash
cd /mnt/workspace/RAGDemo
python -m scripts.diagnose_rag_refusal
```

默认输出outputs/rag_refusal_diagnostic_v1，拒绝覆盖。--prepare-only只构建输入；--report-only读取完成的生成并打印报告。若准备后要继续运行，可手动调用generate_rag_demo，使用该目录results.json、base/v2、输出该目录generation，然后--report-only。推理失败保留输出；不要删除旧记录来反复尝试。

判读：A先看是否复现原拒答；若没有，先比较环境/权重和运行条件，不能继续声称复现成功。A/B变化支持上下文数量敏感；B/C与A/D变化支持问法敏感。E/F检查无依据内容，但一个控制问题不能证明所有拒答能力。出现回答仍需检查内容与引用，不能仅按是否拒答判定优劣。

本地仅验证输入构造、隔离变量及报告逻辑，云端生成尚待执行。此次诊断之后先整理已知局限并推进部署，避免无限针对单题调参。
