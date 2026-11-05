# Adapter 重新加载与生成检查

前提：20 步训练成功，adapter 位于 outputs/qwen2_medical_smoke_v1；保持同一个训练环境，使用 LLaMA-Factory 的 ChatModel 和 qwen 模板，无需安装 vLLM。

只需把本地 scripts/infer_smoke.py 复制到云端 /mnt/workspace/RAGDemo/scripts/infer_smoke.py。不会覆盖训练结果。

```bash
cd /mnt/workspace/RAGDemo
python -m scripts.infer_smoke --mode lora
```

本命令从本地路径加载基座与 adapter，使用 HF 后端生成一条科普回答，并将答案、提示词、生成配置、adapter 哈希保存到 outputs/inference_smoke/lora_时间戳.json。
LLaMA-Factory 可能在内存中合并 adapter，这是正常的加载策略，不会改写磁盘上的基座。

成功后再运行基座作为观察对照：

```bash
python -m scripts.infer_smoke --mode base
```

两个命令必须依次运行、等待前一个进程退出，避免同时占用两份模型显存。base 模式不传 adapter 路径；两次使用同一个问题、system、qwen 模板、BF16、贪心解码和256个最大新token。回答相同不代表 adapter 没加载，回答不同不代表能力提升。

成功标准：没有加载/生成异常，输出非空，生成 JSON 保存成功。finish_reason=length 表示到达生成长度上限，回答可能被截断；不是完整回答质量通过。正式评估尚未实现。
当前科普问题是手工调试问题，未验证与训练数据去重，不用于测试集或成绩报告。
若报错，提供完整错误末尾；不要重新训练或覆盖 adapter。云端重建后仍需要先恢复 LLaMA-Factory 环境。

本地支持 --dry-run 查看配置，不需要 torch、模型或 GPU；仅验证配置拼装，不代表云端执行成功。
