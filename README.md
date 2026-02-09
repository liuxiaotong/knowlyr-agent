# knowlyr-agent

knowlyr 生态的 Agent 工具链 — 从代码执行到轨迹数据生产的完整 Pipeline。

## 架构

```
Task (JSONL / SWE-bench)
  │
  ▼
┌──────────────────┐
│  knowlyr-sandbox │  Docker 隔离执行环境
│  (执行层)         │  pip install knowlyr-sandbox
└────────┬─────────┘
         │ raw log
         ▼
┌──────────────────┐
│ knowlyr-recorder │  Agent 日志 → 标准化轨迹
│  (录制层)         │  pip install knowlyr-recorder
└────────┬─────────┘
         │ Trajectory
         ▼
┌──────────────────┐
│  knowlyr-reward  │  过程级 Reward 评分
│  (评分层)         │  pip install knowlyr-reward
└────────┬─────────┘
         │ scored trajectory
         ▼
┌──────────────────┐
│  knowlyr-hub     │  Pipeline 编排 + 数据导出
│  (编排层)         │  pip install knowlyr-hub
└──────────────────┘
         │
         ▼
  SFT / DPO / HuggingFace
```

## 四个子包

| 包名 | 功能 | CLI 命令 | MCP Server |
|------|------|---------|------------|
| [knowlyr-sandbox](packages/sandbox/) | Docker 沙箱执行环境 | `knowlyr-sandbox` | 4 Tools |
| [knowlyr-recorder](packages/recorder/) | Agent 轨迹录制与格式转换 | `knowlyr-recorder` | 3 Tools |
| [knowlyr-reward](packages/reward/) | 过程级 Rubric Reward 计算 | `knowlyr-reward` | 4 Tools |
| [knowlyr-hub](packages/hub/) | Pipeline 编排与数据集导出 | `knowlyr-hub` | 3 Tools |

每个包**独立安装、独立使用**，也可以通过 Hub 串联为完整 Pipeline。

## 快速安装

```bash
# 按需安装单个包
pip install knowlyr-sandbox
pip install knowlyr-recorder
pip install knowlyr-reward
pip install knowlyr-hub

# 或者安装 Hub 并拉取全部依赖
pip install knowlyr-hub[all]
```

## 开发

```bash
git clone https://github.com/liuxiaotong/knowlyr-agent.git
cd knowlyr-agent

# 开发模式安装全部包
make install-dev

# 运行全部测试
make test

# Lint 检查
make lint
```

## 所属生态

本项目是 [knowlyr 数据工程生态](https://github.com/liuxiaotong) 的 Agent 工具链部分。

## License

MIT
