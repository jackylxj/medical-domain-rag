# Sources and distribution / 来源与发布说明

- Medical QA / 医疗问答数据：[FreedomIntelligence/Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite). The project used this public dataset; it is not hospital-internal data. 项目使用该公开数据集，不属于医院内部数据。
- Generation model / 生成模型：[Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct).
- Embedding model / 向量模型：[BGE-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5).
- Training framework / 训练框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- The development corpus used project-authored summaries of public health information. 开发语料使用基于公开健康资料编写的摘要。

The current source tree does not distribute datasets, corpus text, recorded answers, or evaluation outputs. Prepare authorized inputs separately and follow each upstream provider's terms. Model weights and credentials are not included.

当前源码目录不分发数据集、语料正文、历史回答或评测输出。使用者需自行准备有权使用的输入，并遵循上游数据、模型和依赖的使用条款。模型权重与凭据不随仓库提供。
