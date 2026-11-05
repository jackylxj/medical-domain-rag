# 第二轮新增来源登记与预先划分

本批12个新WHO来源在QA编写前已划分；旧6来源全部排除。采集于2026-11-05。

| 用途 | 主题 | 来源 | 快照字符数 |
|---|---|---|---:|
| train | 口腔卫生 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/oral-health) | 2475 |
| train | 耳聋和听力损失 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/deafness-and-hearing-loss) | 2300 |
| train | 视力损害和盲症 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/blindness-and-visual-impairment) | 2225 |
| train | 贫血症 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/anaemia) | 2442 |
| train | 癫痫 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/epilepsy) | 2544 |
| train | 季节性流感 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/influenza-(seasonal)) | 2690 |
| development | 抑郁障碍（抑郁症） | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/depression) | 2250 |
| development | 乙型肝炎 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/hepatitis-b) | 2976 |
| development | 儿童肺炎 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/pneumonia) | 1561 |
| test_reserved | 痴呆症 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/dementia) | 3021 |
| test_reserved | 结核病 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/tuberculosis) | 3644 |
| test_reserved | 腹泻病 | [原文](https://www.who.int/zh/news-room/fact-sheets/detail/diarrhoeal-disease) | 2091 |

## 版本与内容处理

registry.json记录原定划分；snapshot_manifest.json记录原始/清理后哈希、来源、采集时间。页面发布日期不能可靠提取，保留null；版本由内容SHA256标识。第三方Reader可能存在缓存或提取遗漏，不声称源站逐字实时快照。WHO原始页面已另外确认可访问。

私有原始快照：data/source_snapshots/evidence_sources_v2；清理正文：data/source_snapshots/evidence_sources_v2_clean。原始快照不覆盖；清理保留从重要事实到WHO应对章节之前的内容，排除机构行动与公共导航。全部正文不进入源码包；后续使用和分发另行核对提供方条款。

本批清理后共30219字符；跨划分规范化后长度至少80字符的完全相同段落为0。只做精确段落检查，不能证明不存在语义近重复、训练预料污染或同一事实跨主题出现。

## 来源隔离约束

- 同一WHO页面的中英文版本、查询参数、修订版本归为同一source_group，不得另分到其他集合。
- train仅用于训练样本；development用于选择方案；test_reserved只用于未来保留测试题标注和最终评估。
- 测试来源不能作为训练/开发的干扰片段；不要将全部目录直接混合构建训练数据。
- 来源预留不等于已有测试题，也不保证未出现在Qwen预训练数据中。当前12来源尚未生成问答。
- 3个测试来源是首批预留，不足以宣称200题具有广泛主题代表性；扩大测试题前还需补充更多独立主题。

## 本轮验证

新增6项来源/正文清理测试；全项目60项测试通过。抓取发现儿童肺炎页面混入导航，清理后重新检查，保留初始快照及原始重复记录供审计。未调用GPU模型，未启动训练，未更改网页。

## 下一步

先从6个train来源提取带原文位置和条件的事实单元，制作少量带依据、引用与缺失信息的问答，逐条检查。3个development来源另行编写评估题。test_reserved保持隔离，不参与提示词调整。经近重复和质量审阅后再扩展数量，不能只对少数段落反复改写以达到1000条。

## 重建命令

```bash
python -m scripts.collect_evidence_sources --output data/source_snapshots/evidence_sources_new
python -m scripts.clean_evidence_sources --input data/source_snapshots/evidence_sources_new --output data/source_snapshots/evidence_sources_new_clean
```

重新采集可能获得新版本，不能覆盖本次快照或悄悄变更已冻结测试来源。
