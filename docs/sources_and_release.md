# 来源与发布说明

- 问答数据：[FreedomIntelligence/Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)。本仓库提供处理脚本，不再分发原始问答。
- 生成模型：[Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)。模型权重不随源码包提供。
- 向量模型：[BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)。模型权重不随源码包提供。
- 微调工具：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。上游项目独立安装，不冒充自研训练框架。
- RAG 文本：experiments/retrieval_v1 与 retrieval_v2 中的项目编写摘要；各语料文件保存 WHO 页面链接与页面日期。摘要不是 WHO 官方完整文档，也不是临床知识库。

使用、分发模型、数据、第三方代码时，应遵循各原始提供方条款。本项目不重新授予这些材料的权利。仓库尚未选择自己的代码许可证；公开可见不等于已授予开源使用许可。正式开源前由作者决定许可证。

## 源码发布范围

scripts/package_source.py 使用显式目录、扩展名和文件白名单，仅打包源码、配置模板、开发语料与文档。它排除 .env、虚拟环境、模型、原始 QA、运行输出和权重。它不代替发布前检查。

源码包是传输产物。GitHub 仓库目标为 jackylxj/medical-domain-rag，提交源文件、README、记录与测试，不只上传 ZIP。个人 CV 和内部计划不随源码发布。

## 评估口径

领域适配报告字符级 ROUGE-L F1，服务实验报告串行任务耗时，RAG 开发检查报告证据支持与回答完整性；三者分开解释。当前实验日期为 2026 年 11 月，具体实现与统计条件见 experiment_report.md。
