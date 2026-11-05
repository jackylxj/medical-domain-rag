# 训练配置

train_smoke.yaml 是唯一已实现的训练配置：Qwen2-7B-Instruct，BF16 LoRA，q_proj/v_proj，rank 8，最大序列长度 512，单卡 batch 1 / 累积 4，20 次更新。
所有相对路径由 scripts.train 按项目根目录解析；模型默认使用云端绝对路径。
修改 output_dir 可创建新实验，脚本拒绝覆盖非空输出。
当前不提供 train_lora.yaml：正式实验需先确定质量筛选、去重、独立数据划分及长度分布，避免把调试参数误作正式方案。
