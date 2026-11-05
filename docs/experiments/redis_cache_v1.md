# Redis 缓存实测（2026-11-05）

环境：ModelScope A10，常驻 CPU BGE + Milvus Lite、vLLM Qwen2-7B-Instruct，FastAPI 单 worker。
Redis 6.0.16，redis-py 5.2.1。最终配置核验：127.0.0.1:6379，128 MiB，allkeys-lru，关闭磁盘持久化。

最终运行 `outputs/redis_cache_validation_v2`，6 个串行任务全部完成。

| 问题 | 首次未命中/秒 | 重复命中/秒 |
|---|---:|---:|
| 成年人全天食盐建议 | 1.0122 | 0.0402 |
| 家务和步行是否为活动 | 1.0214 | 0.0528 |
| 资料外的哮喘药商品名与剂量 | 0.7601 | 0.0383 |

三组命中结果与原任务的答案和证据完全一致；资料外问题复用“当前资料不足以回答该问题”。
缓存命中没有调用检索 search 或 vLLM 生成，仍调用检索 health 确认版本。
表中是服务端任务耗时，不含浏览器网络、轮询或服务启动；不是并发压测或医学准确率结果。
新问题仍需正常检索生成。不能把命中耗时宣传为模型生成速度。

验收通过：真实 Redis 存取；一秒测试 TTL 到期；合成 corpus/model/prompt/deployment 变化导致未命中；关闭端口模拟 Redis 不可用时，真实 BGE+vLLM 仍完成问答。
模型与知识库版本失效由版本上下文与服务重启协议实现，未实际更换模型权重进行端到端版本迁移测试。

首次部署核验发现 Redis 没有加载预期 bind 配置；已重新启动并用 ss 和 CONFIG GET 确认回环监听，再清空该临时缓存并重新完成上表的最终实验。原始首轮记录保留为 validation_v1，不作为最终部署配置的验收证据。

源码：app/answer_cache.py、app/rag_service.py、scripts/verify_answer_cache.py。
运行说明：docs/answer_cache.md。原始 jobs.json、summary.json、outage_result.json 已下载本地 outputs 归档，默认不加入源码版本控制。
