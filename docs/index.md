# knowlyr-gym

**Agent 轨迹工具链 — 环境、录制、奖励、训练**

knowlyr-gym 是一个 monorepo，包含 6 个独立 Python 包，覆盖 Agent 轨迹数据的完整生命周期：

```
环境执行 → 轨迹录制 → 奖励评分 → 数据编排 → 模型训练
```

## 子包一览

| 包 | PyPI | 说明 |
|---|---|---|
| **[Core](api/core.md)** | `knowlyr-core` | 共享数据模型、Gym 协议、环境注册表 |
| **[Sandbox](api/sandbox.md)** | `knowlyr-sandbox` | Docker 代码沙箱 + 对话环境 |
| **[Recorder](api/recorder.md)** | `knowlyr-recorder` | Agent 轨迹录制与标准化 |
| **[Reward](api/reward.md)** | `knowlyr-reward` | 过程级 Reward 计算引擎 |
| **[Hub](api/hub.md)** | `knowlyr-hub` | 轨迹收集 Pipeline 编排 |
| **[Trainer](api/trainer.md)** | `knowlyr-trainer` | SFT / DPO / GRPO 训练 |

## 快速开始

```bash
# 安装全部包（开发模式）
make install-dev

# 运行测试
make test

# 启动文档服务
make docs-serve
```

## API 文档

前往 [API Reference](api/index.md) 查看各包的完整 API 文档。
