# Training configuration / 训练配置

| Configuration | Purpose / 用途 |
|---|---|
| train_smoke.yaml | Validate the training pipeline with a short run / 短程验证训练链路 |
| train_pilot.yaml | Pilot LoRA training / 小规模 LoRA 训练 |
| train_30k.yaml | Main domain-adaptation training configuration / 领域适配主训练配置 |

All three configurations use Qwen2-7B-Instruct and LLaMA-Factory LoRA. Inspect the YAML for dataset identifiers, hyperparameters, and output directories. Configuration values describe intended runs, not evidence of completed training.

三份配置均使用 Qwen2-7B-Instruct 与 LLaMA-Factory LoRA。数据标识、超参数和输出位置以 YAML 为准。配置描述预期运行方式，本身不代表训练完成。

Preview configuration without loading a model / 不加载模型即可检查配置：

```bash
python -m scripts.train --config configs/train_30k.yaml --dry-run
```

Launch after preparing model weights, data manifests, and the cloud training environment / 准备好模型、数据清单与训练环境后启动：

```bash
bash scripts/train.sh --config configs/train_30k.yaml
```

The launcher resolves relative paths against the repository root, checks training and validation data hashes, verifies CUDA/BF16, and refuses a nonempty output directory. Use a new output_dir for a new run.

启动器以项目根目录解析相对路径，核对训练和验证数据哈希，检查 CUDA/BF16，并拒绝覆盖非空输出目录。新运行应使用新的 output_dir。

## Training artifacts / 训练产物

scripts/train.py writes the following files to output_dir during execution; these are not bundled in the repository:

scripts/train.py 在运行时向 output_dir 写入以下文件，仓库不随附这些文件：

| File | Contents / 内容 |
|---|---|
| run_config.yaml | Resolved configuration / 实际生效配置 |
| data_manifest.json | Input manifest / 输入数据清单 |
| environment.json | GPU, library metadata, commits, and hashes / GPU、版本、提交与哈希 |
| pip-freeze.txt | Python dependency snapshot / Python 依赖快照 |
| console.log | Training subprocess stdout and stderr / 训练进程输出 |
| run_result.json | Exit status and adapter-file checks / 退出状态与适配器文件检查 |

LLaMA-Factory may also produce trainer state, step logs, and loss plots depending on its version and configuration. Verify those files in the actual run directory. A successful exit plus adapter files is followed by separate reload and inference verification.

LLaMA-Factory 还可能按版本与配置生成训练状态、逐步日志和损失曲线，具体以实际运行目录为准。进程成功退出且生成适配器后，仍应另行验证重载与推理。
