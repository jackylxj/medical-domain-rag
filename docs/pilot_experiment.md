# 第一轮独立验证实验

## 同步文件

在既有云端 RAGDemo 中新增 scripts/prepare_experiment.py、scripts/evaluate.py、configs/train_pilot.yaml、configs/train_30k.yaml；更新 scripts/train.py。保留已有 scripts/validate_data.py、scripts/infer_smoke.py 等模块。不要覆盖 LLaMA-Factory 源码。

## 执行顺序

```bash
cd /mnt/workspace/RAGDemo
python -m scripts.prepare_experiment --input /mnt/workspace/medical_data/format_data.jsonl
python -m scripts.train --config configs/train_pilot.yaml --dry-run
python -m scripts.train --config configs/train_pilot.yaml
```

数据准备依赖之前保留的 data/processed/smoke/manifest.json。全部输出位于 data/processed/medical_v1。目录非空拒绝覆盖，成功后跳过重复准备。
正式子集 30000、试验子集 3000、验证 500、测试 500，另有验证内部的 dev100 用于快速生成比较。3000 是 30000 的子集，不能把两者合计成33000条独立训练数据。
数据未做专家医学审核；精确归一化去重与前/后缀分桶近似比较只能减少部分泄漏，不能保证语义完全去重。稀有科室可能未被500条划分覆盖，不能宣称各科室性能均获验证。

3k 配置从原基座重新训练，不接着20步 adapter 训练。1epoch，约750次参数更新，q/v LoRA、BF16、512截断沿用已验证条件；500条验证集在epoch末计算loss。512可能截掉较长答案，需记录该限制，正式扩大前检查 token 长度分布。

## 比较基座和第一轮 adapter

训练成功后依次运行，不能同时占用GPU：

```bash
python -m scripts.evaluate --mode base --output outputs/eval_base_dev100_v1
python -m scripts.evaluate --mode lora --output outputs/eval_pilot_dev100_v1
```

两者读取同一dev100和同一提示词、贪心生成、512个最大输出token。输出预测全文、来源索引、配置和summary。错误后保留日志，重跑使用新的output路径，不能把部分预测当完整结果。
当前指标为字符级 ROUGE-L F1（去空白、保留标点，逐条平均，0—100），不是医学准确率，不可与未知分词定义的CV数字直接对比。BLEU和人工评价后续补充；引用答案不等于医学金标准。检查空回答和长度截断比例，并手工核查失败案例。
测试集留到配置确定后再评估，不能根据test分数反复调参。

## 30k 配置（待第一轮检查后执行）

```bash
python -m scripts.train --config configs/train_30k.yaml
```

同样从原基座重新训练，1epoch约7500次更新。时间以第一轮稳定吞吐估算，不保证7小时实例时限内完成。先确认剩余时长、磁盘空间和备份方案。
新结果属于当前公开复现实验，不用于倒填无原始记录的历史指标。

## GitHub

可以发布代码与经过检查的汇总指标，当前不自动建仓库或推送：用户尚未提供目标仓库。
不要发布 predictions.jsonl（含原始QA）、原始数据、adapter、.env 或未检查的环境日志。GitHub代码依赖LLaMA-Factory，明确标注上游与数据来源。
