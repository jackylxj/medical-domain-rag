# Dependency environments / 依赖环境

- **Web API:** requirements.txt; requirements/local.txt references this file. / 网页 API 依赖位于 requirements.txt，local.txt 引用该文件。
- **Redis client:** requirements/cache.txt. Install Redis server separately. / Redis 客户端依赖见 cache.txt，服务端另行安装。
- **Data and training utilities:** requirements/train-tools.txt. / 数据与训练启动工具依赖见 train-tools.txt。
- **GPU training:** install LLaMA-Factory in a compatible CUDA/PyTorch environment. The launcher records pip freeze and the framework commit for each run. / GPU 训练使用兼容的 LLaMA-Factory 环境，启动器记录运行时依赖及框架提交。
- **GPU inference:** vLLM serves Qwen2 through an OpenAI-compatible API; scripts/start_resident_cloud.sh provides a launch entry. / vLLM 提供 Qwen2 推理服务，启动入口见该脚本。
- **CPU retrieval:** sentence-transformers, pymilvus, and milvus-lite in a separate environment. / CPU 检索使用独立环境安装这些依赖。

Review interpreter paths in scripts/start_resident_cloud.sh before deployment. GPU training and inference dependencies must match the target CUDA/PyTorch environment; this directory is not a complete lockfile for all services.

部署前检查启动脚本中的解释器路径。GPU 训练与推理依赖需要匹配目标 CUDA/PyTorch 环境，本目录不是所有服务的完整依赖锁定文件。
