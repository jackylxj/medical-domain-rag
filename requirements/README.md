# 环境隔离

- local.txt：已有 FastAPI 环境，引用根目录现有锁定版本。
- train-tools.txt：数据/启动工具额外依赖。数据校验和抽样仅需 Python 标准库；下载才需要 modelscope。
- 训练使用云端已安装的 LLaMA-Factory，不向 Windows .venv 安装 GPU 栈。每次运行保存 pip-freeze.txt 与上游 commit，当前云端完整依赖尚未拿到，不能声称已锁定复现环境。
- 推理服务（vLLM 等）的依赖在实施该阶段时单独确定，目前未实现。
