> 历史阶段记录。当前实现与启动入口请以项目 README 和 docs/reproduction.md 为准。

# 魔搭云端执行：200 条 / 20 步

## 当前状态

用户已验证 A10、PyTorch 2.13.0+cu130、CUDA 可用、BF16 支持；云端 LLaMA-Factory 0.9.6.dev0。
模型位于 /mnt/workspace/models/Qwen2-7B-Instruct；原始数据位于 /mnt/workspace/medical_data/format_data.jsonl。
基础文件检查通过不等于权重/GPU 训练成功，本步骤才验证实际训练。

## 1. 上传源码

将本地 dist/RAGDemo-source.zip 上传至云端 /mnt/workspace，然后在终端执行：

```bash
cd /mnt/workspace
unzip RAGDemo-source.zip
cd /mnt/workspace/RAGDemo
```

首次上传使用此步骤。如果 RAGDemo 已存在，先核对远端是否有未同步修改，不要盲目覆盖。
不需要上传 .venv、数据或模型；它们已在云端。不会修改 LLaMA-Factory 源码。

## 2. 确认当前训练环境

```bash
which python
which llamafactory-cli
llamafactory-cli version
python -c "import torch, yaml; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.is_bf16_supported())"
```

使用之前成功识别 GPU 的同一个终端环境。若只有 yaml 缺失，执行 python -m pip install 'PyYAML>=6,<7'。已有模型无需再安装下载工具或重新下载。
不要在此环境安装本地 API 的 requirements.txt，避免无关依赖影响训练栈。

## 3. 校验并抽样

```bash
python -m scripts.validate_data --input /mnt/workspace/medical_data/format_data.jsonl
python -m scripts.prepare_data --input /mnt/workspace/medical_data/format_data.jsonl
```

生成项目内 data/processed/smoke/ 下的 train.jsonl、dataset_info.json、manifest.json。
默认抽样 200 条，seed=42；原始文件不变。源记录索引和哈希用于追溯，未来测试集应排除这些样本及近似重复项。
该步骤不做临床质量筛选，不代表正式数据清洗/划分。重复执行会拒绝覆盖：已成功生成就直接继续。
之前手工创建的 /mnt/workspace/medical_data/smoke 不用于本流程，也不会删除。

## 4. 预览配置（不训练）

```bash
python -m scripts.train --dry-run
```

确认 model_name_or_path、dataset_dir、output_dir；dry-run 只解析路径，不验证 GPU 或模型权重。

## 5. 训练

```bash
bash scripts/train.sh
```

也可以执行 python -m scripts.train。默认记录上游目录 /mnt/workspace/LLaMA-Factory 的 commit；若不同用 --factory-dir 指定。
尽量保持终端连接；当前入口不是后台任务。另开终端 nvidia-smi 查看显存。
默认输出 outputs/qwen2_medical_smoke_v1；已有非空目录会被拒绝。重试请在 configs/train_smoke.yaml 中更改 output_dir 为 v2，保留失败日志。
若 OOM，先保留 console.log 和显存信息，再诊断；不要同时启动 vLLM 或盲目安装量化依赖。

## 6. 验收和保留结果

```bash
cat outputs/qwen2_medical_smoke_v1/run_result.json
ls -lh outputs/qwen2_medical_smoke_v1
```

成功标准：returncode=0，adapter_config_exists=true，adapter_weights_exist=true；训练日志达到 20 步且 loss 没有 NaN/Inf。
这些只证明训练流程与保存成功，下一步仍要重新加载 adapter 并推理。不得把 smoke loss 当作医疗效果提升。
200 条是可用数据池；单卡 batch1 × 累积4 × 20步约处理80条样本，并非200条全部训练一轮。截断长度512只用于调试。

输出保存：run_config.yaml、data_manifest.json、environment.json、pip-freeze.txt、console.log、run_result.json 和 adapter。云实例关闭前自行下载备份，不假设平台目录永久保留。
分享排错信息优先提供日志末尾和 run_result.json，不需要发送原始问答。环境文件公开前检查路径及包来源是否包含私人信息。

## 后续

先验证 adapter 加载；再做清洗、去重、固定 train/validation/test 划分，正式微调与统一评估。RAG 的 BGE/Milvus 和服务链单独实施。
