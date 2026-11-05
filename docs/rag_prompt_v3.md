# v3 提示词开发回归

依据corpus v2上的错误拒答（题2、5）、引用遗漏（题7）和信息遗漏（题8），新增v3提示词。允许资料明确支持的同义表达、否定澄清和简单换算；分部分回答并逐句引用；仍不允许外部知识补全。未将具体测试题答案写入提示词。

保留v1/v2、默认行为与网页配置不变。直接使用上轮保存的retrieval/results.json，避免重新检索引入变化。base Qwen2、BF16、贪心解码、512新token上限、system和adapter配置不变。这一批题已经用于调提示词，只能称开发回归题，不是独立盲测。

```bash
python -m scripts.generate_rag_demo --mode base --prompt-version v3 \
  --retrieval outputs/rag_corpus_v2_validation/retrieval/results.json \
  --output outputs/rag_corpus_v2_prompt_v3
python -m scripts.compare_rag_runs \
  --before outputs/rag_corpus_v2_validation/generation \
  --after outputs/rag_corpus_v2_prompt_v3
```

验收：题2应按资料换算而非拒答；题5保留选择冷冻蔬果的条件；题7分别核对久坐和活动不足，不把它们等同，引用覆盖每项陈述；题8给出有依据的食品选择；题10—12保持资料不足。其余题检查退步，不仅统计引用出现率。

云端结果未出之前不宣称修复成功，不切换网页。若改善，另写未用于修改提示词的新题验证；若没有改善，保留失败案例，先分析而非持续针对这12题叠加规则。
