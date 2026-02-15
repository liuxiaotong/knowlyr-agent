# knowlyr-trainer

纯 PyTorch Agent 轨迹训练工具 — SFT / DPO / GRPO，无缝对接 `knowlyr-hub` 导出的数据集。

**Agent 训练增强**: 多轮对话格式、观察 token 遮蔽、步骤级 reward 加权、长轨迹分块、课程学习。

## 安装

```bash
pip install knowlyr-trainer

# 可选
pip install knowlyr-trainer[peft]    # LoRA 微调
pip install knowlyr-trainer[wandb]   # wandb 日志
pip install knowlyr-trainer[all]     # 全部
```

## 快速开始

### CLI

```bash
# SFT 训练
knowlyr-trainer sft --train-file sft.jsonl --model Qwen/Qwen2.5-Coder-7B

# DPO 偏好学习
knowlyr-trainer dpo --train-file dpo.jsonl --model ./output/sft/final --beta 0.1

# GRPO 组内相对策略优化
knowlyr-trainer grpo --train-file grpo.jsonl --model ./output/sft/final

# 模型评估
knowlyr-trainer eval --model ./output/sft/final --eval-file eval.jsonl
```

### YAML 配置

```bash
knowlyr-trainer sft --config train_config.yaml
```

```yaml
# train_config.yaml
model_name_or_path: Qwen/Qwen2.5-Coder-7B
train_file: sft_data.jsonl
output_dir: ./output/sft
num_epochs: 3
batch_size: 4
learning_rate: 2e-5
max_length: 4096
bf16: true
use_lora: true
agent_format: true          # 启用 Agent 多轮格式
mask_observations: true     # 遮蔽观察 token
step_weighted_loss: true    # 步骤级 reward 加权
curriculum: true            # 课程学习
```

### Python API

```python
from agenttrainer import SFTConfig
from agenttrainer.trainers.sft import SFTTrainer

config = SFTConfig(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B",
    train_file="sft_data.jsonl",
    output_dir="./output",
    agent_format=True,
    mask_observations=True,
)
trainer = SFTTrainer(config)
trainer.train()
```

## 数据格式

无缝对接 `knowlyr-hub export` 导出的 JSONL：

```bash
knowlyr-hub export --format sft  -t trajectories.jsonl -o sft_train.jsonl
knowlyr-hub export --format dpo  -t trajectories.jsonl -p preferences.jsonl -o dpo_train.jsonl
knowlyr-hub export --format grpo -t trajectories.jsonl -o grpo_train.jsonl
```

### Agent 增强数据格式

启用 `agent_format=True` 时，支持结构化步骤数据：

```json
{
  "instruction": "Fix the off-by-one bug in sort function",
  "input": "{\"repo\": \"owner/repo\"}",
  "steps": [
    {"thought": "Read the file", "action": "read_file /sort.py", "observation": "def sort(arr): ...", "reward": 0.7},
    {"thought": "Fix the bug",   "action": "edit_file /sort.py", "observation": "File edited",       "reward": 0.9}
  ],
  "task_id": "task-001",
  "reward": 0.85
}
```

也兼容平文本 `response` 字段（自动解析 `Step N: / Thought: / Action: / Observation:` 格式）。

## Agent 训练增强

标准 SFT/DPO/GRPO 之外，针对 Agent 长程任务的 6 项增强：

### 1. 多轮对话格式 (`agent_format`)

将轨迹从平文本转为结构化多轮对话：

```
user:       Fix the bug in sort.py        ← 任务描述（不参与 loss）
assistant:  Thought: Read the file        ← 模型输出（参与 loss ✓）
            Action: read_file /sort.py
user:       Observation: def sort(arr)... ← 环境反馈（不参与 loss）
assistant:  Thought: Fix the comparison   ← 模型输出（参与 loss ✓）
            Action: edit_file /sort.py
user:       Observation: File edited      ← 环境反馈（不参与 loss）
```

### 2. 观察遮蔽 (`mask_observations`)

只对模型生成的 thought + action token 计算 loss，环境返回的 observation token 设为 `labels=-100`。避免模型学习「预测环境行为」，专注于「学习决策」。

### 3. 步骤级 reward 加权 (`step_weighted_loss`)

使用 `knowlyr-reward` 的步骤级 process reward 加权每个 token 的 CE loss：

```
loss_token = CE(token) × (step_reward / mean_reward)
```

好的步骤获得更高权重，差的步骤被降权。

### 4. 长轨迹分块 (`chunk_long_trajectories`)

超过 `max_length` 的轨迹按步骤边界拆分为多个训练样本。每个 chunk 包含任务描述 + 上下文步骤 + 当前段，不在步骤中间断开。

### 5. 课程学习 (`curriculum`)

从简单（短轨迹/高 reward）到困难（长轨迹/低 reward）渐进式训练：

```yaml
curriculum: true
curriculum_start_ratio: 0.3    # 初始使用 30% 最简单样本
curriculum_warmup_epochs: 2    # 2 个 epoch 后使用全部数据
```

### 6. 步骤级 GRPO (`step_level_advantage`)

在 GRPO 的轨迹级 advantage 基础上，用步骤 reward 进一步加权：

```
A_{i,j} = A_trajectory_i × (r_{step_j} / mean(r_steps))
```

好的轨迹中的好步骤获得更大正梯度，差的轨迹中的差步骤受到更大惩罚。

## 训练方法

| 方法 | 用途 | 数据格式 | CLI |
|------|------|---------|-----|
| **SFT** | 监督微调 | instruction/response JSONL | `knowlyr-trainer sft` |
| **DPO** | 偏好对齐 | prompt/chosen/rejected JSONL | `knowlyr-trainer dpo` |
| **GRPO** | 组内策略优化 | prompt + 多条轨迹 JSONL | `knowlyr-trainer grpo` |

## 功能矩阵

| 功能 | SFT | DPO | GRPO |
|------|-----|-----|------|
| 多轮对话格式 | ✅ | — | — |
| 观察遮蔽 | ✅ | — | — |
| 步骤加权 loss | ✅ | — | — |
| 长轨迹分块 | ✅ | — | — |
| 课程学习 | ✅ | — | — |
| 步骤级 advantage | — | — | ✅ |
| LoRA | ✅ | ✅ | ✅ |
| bf16 混合精度 | ✅ | ✅ | ✅ |
| Checkpoint 保存 | ✅ | ✅ | ✅ |
| wandb 日志 | ✅ | ✅ | ✅ |

## License

MIT
