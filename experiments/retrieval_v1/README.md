# 第一轮医疗参考资料检索

从项目根目录运行。输入为 references 下的三篇中文 Markdown 摘要，各按 WHO 同名章节整理。每个二级标题生成一个片段，正文与版本、章节、源链接一起存入 Milvus Lite。各题只有一个预先指定的目标章节。题目和参考资料专为开发检查编写，与 Huatuo 测试集分别统计。

在单独的 Python 3.12 环境中安装 requirements.txt。以魔搭 Linux 终端为例，只需将本目录复制到 /mnt/workspace/RAGDemo/experiments/retrieval_v1/。

```bash
cd /mnt/workspace/RAGDemo
python -m venv /mnt/workspace/envs/rag-retrieval-v1
source /mnt/workspace/envs/rag-retrieval-v1/bin/activate
python -m pip install -r experiments/retrieval_v1/requirements.txt
python experiments/retrieval_v1/run.py
```

首次默认从 Hugging Face 下载固定 revision 的 BAAI/bge-large-zh-v1.5。若云端无法联网，提前放置该版本的完整模型目录，并在命令末尾添加 `--model-dir /mnt/workspace/models/bge-large-zh-v1.5`。模型可在 CPU 上运行。若希望节省 Linux GPU 依赖的下载，可先从 PyTorch CPU wheel 源安装对应版本的 torch。

输出位于 outputs/retrieval_v1。summary.json 记录命中率，results.json 保存每题前三名、余弦相似度、证据摘要和出处。数据库重新打开后才执行检索。重建需要新的 --output；保留原实验用于比较。

```bash
python experiments/retrieval_v1/run.py --search-existing --query '怀孕期间如何发现血糖异常？'
```

模型与建库时保持一致。若使用本地模型目录，查询时同样传入 --model-dir。每个来源的版本是网页发布日期。检索结果里的 text 是项目自写摘要，核查原文请打开 source。

指标按目标章节的名次计算，Hit@1 和 Hit@3 是命中题数除以总题数，MRR@3 对前三名中的目标取排名倒数并求平均。12 道题只用于开发冒烟检查；同一份语料生成的题目不能代表独立医疗问答测试。
