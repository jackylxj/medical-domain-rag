# 实验记录索引

当前发布版见[发布说明](../release_v1.md)，整体结论见[实验报告](../experiment_report.md)。本目录保存 2026 年 11 月的实验设计、结果与技术取舍。

| 阶段 | 记录 |
|---|---|
| 医疗 QA 领域适配与测试集对照 | [30k LoRA / test500](2026-11-05-lora-test500.md) |
| 常驻检索与生成 | [串行服务计时](resident_benchmark_v1.md) |
| Redis 缓存与回退 | [缓存实测](redis_cache_v1.md) |
| 先定位证据、再生成与核验 | [证据流程](evidence_pipeline_v1.md) |
| 规则检查探索 | [规则验证](evidence_rules_v2.md)、[挑战集](evidence_rules_challenge_v1.md) |
| 微调复盘 | [训练复盘](training_retrospective_2026-11-05.md) |
| 原始模型与 LoRA 开发对照 | [候选开发题基线](evidence_candidate_baseline_v1.md) |
| 固定证据的提示词对照 | [short / strict](evidence_candidate_prompt_v1.md) |

每次训练保存运行配置、数据清单、环境和日志；原始问答与逐题预测保留在私有 outputs 备份，不随源码分发。开发检查、保留测试集和工程计时分别报告。历史报告里的后续方案保留为研究记录；本次发布不启动第二轮训练。
