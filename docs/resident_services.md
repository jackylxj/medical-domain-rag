# 常驻检索与生成服务

服务结构：网页 FastAPI（8000）→ CPU BGE + Milvus Lite（8002）→ vLLM Qwen2（8001）。
8001、8002 仅监听回环地址。网页通过平台认证代理访问。只运行一个网页 worker。

## 启动

在 ModelScope 中执行 `bash scripts/start_resident_cloud.sh`。脚本依赖已有模型、索引与环境，不负责安装。
不要在上一次启动尚未完成时重复执行；已有监听端口会被保留，需要升级代码时应先停止对应旧服务。
日志位于 `outputs/resident_services/`。脚本输出 launch requested 不表示就绪。

分别检查 `http://127.0.0.1:8001/v1/models`、`http://127.0.0.1:8002/health`、`http://127.0.0.1:8000/health`。
网页 health 只检查路径与配置，必须提交实际问题验证整条链路。

通过 SSH 启动时可能缺少 Notebook 默认环境中的 CUDA 兼容库路径。
本次镜像在 `/usr/local/cuda/compat` 提供该库；脚本优先加入动态库搜索路径。
这不是驱动升级，换镜像后应重新验证 GPU 运算与 vLLM 启动。

## 计时实验

运行 `python -m scripts.benchmark_resident_web --output outputs/resident_benchmark_v1`。
输出目录必须不存在。五条问题顺序执行，其中第一条重复一次，包含一条资料外问题。
完整任务与回答保存在 `jobs.json`，汇总在 `summary.json`。

- `elapsed_seconds`：网页服务端任务耗时，包含检索、生成子进程和中间文件处理，不含浏览器网络与轮询。
- `embedding_seconds`、`vector_search_seconds`：常驻检索内部耗时。
- `retrieving_wall_seconds`：检索 HTTP 请求与保存结果的耗时。
- `generating_wall_seconds`：生成客户端子进程整体耗时。
- `vllm_request_seconds`：仅 vLLM 生成 HTTP 请求耗时。
- `client_observed_seconds`：测试客户端观察到完成的时间，包含轮询等待。

服务启动耗时另外记录在检索 health 的 `startup_timings`，不能混入稳定请求延迟。
本轮没有应用层答案缓存或 Redis；vLLM 自身可能启用前缀缓存。重复问题不等于 Redis 命中。
五条样本只用于功能与延迟观察，不是并发压测，也不能据此声称医学准确率或稳定的 P95。

## 边界

Milvus Lite 由常驻检索进程持有。重建索引前先停止该服务，不要让另一个进程同时打开该数据库。
重建完成后重启服务以加载新索引。实例关闭会终止所有服务，nohup 只能应对终端断开。
当前保持 base Qwen2 + v2 提示词；LoRA 的离线问答实验独立保存。已知错误拒答未因后端提速而解决。
