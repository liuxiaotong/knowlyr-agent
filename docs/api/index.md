# API Reference

knowlyr-agent 的 6 个子包 API 文档，从 docstring 自动生成。

## 包速查

| 包 | 核心类 / 函数 | 用途 |
|---|---|---|
| [Core](core.md) | `AgentEnv` · `TimeStep` · `register` · `make` · `DomainProfile` | Gym 协议 + 注册表 + 领域配置 |
| [Sandbox](sandbox.md) | `SandboxEnv` · `ConversationEnv` · `SandboxConfig` | 执行环境 |
| [Recorder](recorder.md) | `Recorder` · `Trajectory` · `Step` · `ToolCall` | 轨迹录制 |
| [Reward](reward.md) | `RewardEngine` · `Rubric` · `PreferencePair` | 奖励计算 |
| [Hub](hub.md) | `collect` · `Pipeline` · `DatasetExporter` · `create_model_agent` | 数据编排 |
| [Trainer](trainer.md) | `TrainConfig` · `SFTConfig` · `AgentInference` · `evaluate_agent` | 模型训练 |

## 核心概念

### Gym 风格环境协议

借鉴 Gymnasium / BrowserGym 设计，所有环境实现统一接口：

```python
env = make("knowlyr/sandbox")
timestep = env.reset(task=my_task)

while not timestep.terminated:
    action = agent(timestep.observation)
    timestep = env.step(action)

env.close()
```

### 数据流

```
AgentEnv.step() → TimeStep → Recorder → Trajectory → RewardEngine → Hub → Trainer
```

### 环境注册表

```python
from knowlyrcore import register, make, list_envs

register("my/env", MyEnvClass, domain="coding")
env = make("my/env")
print(list_envs())  # ["knowlyr/sandbox", "knowlyr/conversation", ...]
```
