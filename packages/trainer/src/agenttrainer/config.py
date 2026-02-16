"""训练配置 - Pydantic 模型."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """共享训练配置基类.

    所有训练方式（SFT、DPO、GRPO）的公共配置。支持从 YAML 文件加载和 CLI 参数覆盖。

    Example::

        from agenttrainer import TrainConfig

        # 从 YAML 加载
        config = TrainConfig.from_yaml("train_config.yaml")

        # CLI 参数覆盖
        config = config.merge_cli(learning_rate=1e-5, num_epochs=5)

        # 直接构造
        config = TrainConfig(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B",
            train_file="./data/train.jsonl",
            agent_format=True,
            mask_observations=True,
            output_dir="./checkpoints",
        )
    """

    # 模型
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B"
    tokenizer_name: str | None = None  # 默认同 model_name_or_path

    # 数据
    train_file: str = ""
    eval_file: str | None = None
    max_length: int = 2048

    # 训练超参
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # 精度
    bf16: bool = True

    # 内存优化
    gradient_checkpointing: bool = False

    # 日志 & 保存
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42

    # 恢复训练 — 从 checkpoint 目录继续 (包含 training_state.pt)
    resume_from_checkpoint: str | None = None

    # wandb (需要 knowlyr-trainer[wandb])
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # LoRA (需要 knowlyr-trainer[peft])
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    # ── Agent 训练增强 ──────────────────────────────────
    # 使用多轮 agent 格式（step → thought+action / observation），而非平文本
    agent_format: bool = False
    # 遮蔽环境观察 token（labels=-100），只对 thought+action 计算 loss
    mask_observations: bool = True
    # 使用步骤级 process reward 加权 loss
    step_weighted_loss: bool = False

    # ── 长轨迹分块 ──────────────────────────────────────
    chunk_long_trajectories: bool = False
    chunk_overlap: int = 128  # 块之间重叠的 token 数

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """从 YAML 文件加载配置."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def merge_cli(self, **kwargs: Any) -> "TrainConfig":
        """用 CLI 参数覆盖配置（忽略 None 值）."""
        updates = {k: v for k, v in kwargs.items() if v is not None}
        return self.model_copy(update=updates)


class SFTConfig(TrainConfig):
    """SFT 训练配置.

    在 TrainConfig 基础上增加课程学习（Curriculum Learning）支持。

    Example::

        from agenttrainer import SFTConfig

        config = SFTConfig(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B",
            train_file="./data/sft_train.jsonl",
            agent_format=True,
            mask_observations=True,
            curriculum=True,              # 启用课程学习
            curriculum_start_ratio=0.3,   # 初始只用 30% 简单样本
            curriculum_warmup_epochs=1,   # 1 epoch 后使用全部数据
            output_dir="./checkpoints/sft",
        )
    """

    # Curriculum learning — 从简单样本逐步过渡到困难样本
    curriculum: bool = False
    curriculum_start_ratio: float = 0.3  # 初始阶段使用数据的比例
    curriculum_warmup_epochs: int = 1  # 几个 epoch 后使用全部数据


class DPOConfig(TrainConfig):
    """DPO 训练配置.

    Direct Preference Optimization 训练。需要偏好对数据（chosen/rejected）。

    Example::

        from agenttrainer import DPOConfig

        config = DPOConfig(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B",
            train_file="./data/dpo_train.jsonl",
            beta=0.1,              # KL 惩罚系数（越大越保守）
            label_smoothing=0.0,   # 标签平滑
            output_dir="./checkpoints/dpo",
        )
    """

    beta: float = 0.1
    label_smoothing: float = 0.0


class GRPOConfig(TrainConfig):
    """GRPO 训练配置.

    Group Relative Policy Optimization 训练。需要分组轨迹数据。

    Example::

        from agenttrainer import GRPOConfig

        config = GRPOConfig(
            model_name_or_path="Qwen/Qwen2.5-Coder-7B",
            train_file="./data/grpo_train.jsonl",
            group_size=8,              # 每组轨迹数
            clip_epsilon=0.2,          # PPO clip 范围
            kl_coef=0.01,             # KL 惩罚系数
            step_level_advantage=True, # 使用步骤级 advantage
            output_dir="./checkpoints/grpo",
        )
    """

    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    # 步骤级 advantage — 在轨迹级 advantage 基础上用步骤 reward 加权
    step_level_advantage: bool = False
