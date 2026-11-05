# 复现入口与环境记录

这是已有 ModelScope A10 环境上验证的项目，不是经过全新机器安装验收的容器镜像。启动脚本包含当前云端绝对路径，换机器须相应修改。不要把本地 API requirements.txt 当成 GPU 全环境锁文件。

## 环境分离

| 用途 | 已记录版本 |
|---|---|
| 30k 训练保存的 pip-freeze | torch 2.13.0、transformers 5.8.0、peft 0.18.1、accelerate 1.11.0、datasets 4.0.0、trl 0.24.0 |
| 当前在线生成 / 网页 | torch 2.13.0、transformers 5.16.1、vllm 0.27.1、fastapi 0.136.3、uvicorn 0.52.4、redis 5.2.1 |
| 独立检索环境 | torch 2.8.0、transformers 4.57.1、sentence-transformers 5.1.2、pymilvus 3.0.2、milvus-lite 3.2.2 |

上游 LLaMA-Factory 当前 commit：9997aa078b4da35e04bb492a8a27d45141f4b80b。训练原始输出保留环境清单；当前 commit 不单独证明历史训练源码未改变。当前在线 transformers 已不同，不要直接在服务环境里重跑训练或为了训练降级正在运行的服务依赖。

检索解释器：/mnt/workspace/envs/rag-retrieval-v1/bin/python；当前网页/vLLM解释器：/usr/local/bin/python。Python 3.12，GPU A10。环境表只是关键版本记录，不是完整依赖锁。

## 从数据到结果

1. 准备源码、单独训练环境与 LLaMA-Factory；先验证 CUDA、BF16 及训练模块导入。
2. 从原始提供方获取 Huatuo26M-Lite。原始文件放 /mnt/workspace/medical_data/format_data.jsonl。模型 Qwen2-7B-Instruct 放 /mnt/workspace/models/Qwen2-7B-Instruct。scripts.download_model 可下载模型；默认未锁远端 revision，首次下载后保存来源版本与文件哈希。
3. 先 prepare_data，再 prepare_experiment：后者依赖前者的 smoke manifest，以排除试跑样本。已有产物不要覆盖。

```bash
python -m scripts.prepare_data --input /mnt/workspace/medical_data/format_data.jsonl
python -m scripts.prepare_experiment --input /mnt/workspace/medical_data/format_data.jsonl
python -m scripts.train --config configs/train_30k.yaml --dry-run
python -m scripts.train --config configs/train_30k.yaml
python -m scripts.evaluate --mode base --data data/processed/medical_v1/test.jsonl --output outputs/eval_base_test500_new
python -m scripts.evaluate --mode lora --adapter outputs/qwen2_medical_30k_v1 --data data/processed/medical_v1/test.jsonl --output outputs/eval_30k_test500_new
```

4. 完成训练后释放 GPU，再运行在线 vLLM；不要同时训练和启动服务。保留 adapter、运行配置、数据 manifest、环境与预测结果的私有备份。
5. 安装独立检索环境的 experiments/retrieval_v1/requirements.txt，并取得 BAAI/bge-large-zh-v1.5 放入 /mnt/workspace/models/bge-large-zh-v1.5。使用 docs/text_corpus_v2.md 中的构建命令建立 outputs/retrieval_v2；先查看 summary.status=indexed。已有索引不重复构建；重建前停止常驻检索，避免 Milvus Lite 文件冲突。
6. 按 docs/resident_services.md 和 docs/answer_cache.md 安装 Redis 服务与客户端，启动常驻服务。脚本不安装系统 Redis，云镜像重启后需检查依赖和进程。
7. 三个端口健康检查通过后，通过网页完成一次有证据问题、一次资料外问题、一次重复问题。核对答案、来源、拒答和缓存状态，不只看 HTTP 200。

## 验收与备份

源码测试：python -m unittest discover -s tests。
缓存复测：python -m scripts.verify_answer_cache --output outputs/redis_cache_new；仅在空闲且新缓存环境运行，已有输出目录拒绝覆盖。
不要为重新打包再次训练。GitHub 发布源码与汇总；模型、原始数据和逐题预测单独备份并先检查再分发。

已记录的测试集 SHA256：537f93646f394009f0a458570b8116d55b83d5f2a1e224743475f591401da6a7。
训练原始数据 SHA256：84b849f07ab3b9517443f5efeb6c7230dcadf1f71ccaf41f3bf889e53618d4e8。
