"""测试训练配置 — TrainConfig / SFTConfig / DPOConfig / GRPOConfig."""

import tempfile
from pathlib import Path

import yaml

from agenttrainer.config import DPOConfig, GRPOConfig, SFTConfig, TrainConfig


# ── TrainConfig 基类 ───────────────────────────────────────────────


class TestTrainConfig:
    """测试 TrainConfig 共享基类."""

    def test_defaults(self):
        """默认值应正确."""
        config = TrainConfig()
        assert config.model_name_or_path == "Qwen/Qwen2.5-Coder-7B"
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-5
        assert config.lr_scheduler == "cosine"
        assert config.bf16 is True
        assert config.seed == 42
        assert config.agent_format is False
        assert config.mask_observations is True

    def test_custom_values(self):
        """自定义值应生效."""
        config = TrainConfig(
            model_name_or_path="meta-llama/Llama-3-8B",
            num_epochs=5,
            learning_rate=1e-4,
            output_dir="./custom",
        )
        assert config.model_name_or_path == "meta-llama/Llama-3-8B"
        assert config.num_epochs == 5
        assert config.learning_rate == 1e-4
        assert config.output_dir == "./custom"

    def test_from_yaml(self):
        """应能从 YAML 文件加载配置."""
        data = {
            "model_name_or_path": "test-model",
            "num_epochs": 10,
            "batch_size": 8,
            "learning_rate": 3e-5,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        try:
            config = TrainConfig.from_yaml(yaml_path)
            assert config.model_name_or_path == "test-model"
            assert config.num_epochs == 10
            assert config.batch_size == 8
            assert config.learning_rate == 3e-5
            # 未指定的字段保持默认
            assert config.seed == 42
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_partial(self):
        """YAML 文件只需包含部分字段."""
        data = {"num_epochs": 20}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        try:
            config = TrainConfig.from_yaml(yaml_path)
            assert config.num_epochs == 20
            assert config.model_name_or_path == "Qwen/Qwen2.5-Coder-7B"  # 默认值
        finally:
            Path(yaml_path).unlink()

    def test_merge_cli(self):
        """CLI 参数应覆盖配置."""
        config = TrainConfig(num_epochs=3, learning_rate=2e-5)
        merged = config.merge_cli(num_epochs=10, learning_rate=None)

        assert merged.num_epochs == 10
        assert merged.learning_rate == 2e-5  # None 被忽略

    def test_merge_cli_returns_new_instance(self):
        """merge_cli 应返回新实例，不修改原配置."""
        config = TrainConfig(num_epochs=3)
        merged = config.merge_cli(num_epochs=10)

        assert config.num_epochs == 3  # 原值不变
        assert merged.num_epochs == 10

    def test_lora_defaults(self):
        """LoRA 默认配置应正确."""
        config = TrainConfig()
        assert config.use_lora is False
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.lora_target_modules == ["q_proj", "v_proj"]

    def test_agent_format_fields(self):
        """Agent 训练增强字段应有正确默认值."""
        config = TrainConfig()
        assert config.agent_format is False
        assert config.mask_observations is True
        assert config.step_weighted_loss is False
        assert config.chunk_long_trajectories is False
        assert config.chunk_overlap == 1

    def test_wandb_fields(self):
        """wandb 字段默认应为 None."""
        config = TrainConfig()
        assert config.wandb_project is None
        assert config.wandb_run_name is None

    def test_resume_checkpoint(self):
        """resume_from_checkpoint 默认应为 None."""
        config = TrainConfig()
        assert config.resume_from_checkpoint is None

        config2 = TrainConfig(resume_from_checkpoint="./ckpt/step-500")
        assert config2.resume_from_checkpoint == "./ckpt/step-500"

    def test_early_stopping_defaults(self):
        """early stopping 默认应为禁用."""
        config = TrainConfig()
        assert config.early_stopping_patience is None
        assert config.save_best_model is False

    def test_early_stopping_custom(self):
        """自定义 early stopping 参数应生效."""
        config = TrainConfig(
            early_stopping_patience=3,
            save_best_model=True,
        )
        assert config.early_stopping_patience == 3
        assert config.save_best_model is True


# ── SFTConfig ─────────────────────────────────────────────────────


class TestSFTConfig:
    """测试 SFTConfig."""

    def test_inherits_train_config(self):
        """应继承 TrainConfig 所有字段."""
        config = SFTConfig(model_name_or_path="test-model")
        assert config.model_name_or_path == "test-model"
        assert config.num_epochs == 3  # 继承默认值

    def test_curriculum_defaults(self):
        """课程学习字段默认值应正确."""
        config = SFTConfig()
        assert config.curriculum is False
        assert config.curriculum_start_ratio == 0.3
        assert config.curriculum_warmup_epochs == 1

    def test_curriculum_enabled(self):
        """启用课程学习应生效."""
        config = SFTConfig(
            curriculum=True,
            curriculum_start_ratio=0.5,
            curriculum_warmup_epochs=2,
        )
        assert config.curriculum is True
        assert config.curriculum_start_ratio == 0.5
        assert config.curriculum_warmup_epochs == 2

    def test_from_yaml(self):
        """SFTConfig 应能从 YAML 加载."""
        data = {
            "model_name_or_path": "sft-model",
            "curriculum": True,
            "curriculum_start_ratio": 0.4,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            yaml_path = f.name

        try:
            config = SFTConfig.from_yaml(yaml_path)
            assert config.curriculum is True
            assert config.curriculum_start_ratio == 0.4
        finally:
            Path(yaml_path).unlink()


# ── DPOConfig ─────────────────────────────────────────────────────


class TestDPOConfig:
    """测试 DPOConfig."""

    def test_defaults(self):
        """DPO 特有字段默认值应正确."""
        config = DPOConfig()
        assert config.beta == 0.1
        assert config.label_smoothing == 0.0

    def test_custom_beta(self):
        """自定义 beta 应生效."""
        config = DPOConfig(beta=0.3, label_smoothing=0.1)
        assert config.beta == 0.3
        assert config.label_smoothing == 0.1

    def test_inherits_train_config(self):
        """应继承 TrainConfig 所有字段."""
        config = DPOConfig(model_name_or_path="dpo-model")
        assert config.model_name_or_path == "dpo-model"
        assert config.agent_format is False  # 继承默认值


# ── GRPOConfig ────────────────────────────────────────────────────


class TestGRPOConfig:
    """测试 GRPOConfig."""

    def test_defaults(self):
        """GRPO 特有字段默认值应正确."""
        config = GRPOConfig()
        assert config.group_size == 8
        assert config.clip_epsilon == 0.2
        assert config.kl_coef == 0.01
        assert config.step_level_advantage is False

    def test_custom_values(self):
        """自定义 GRPO 参数应生效."""
        config = GRPOConfig(
            group_size=16,
            clip_epsilon=0.1,
            kl_coef=0.05,
            step_level_advantage=True,
        )
        assert config.group_size == 16
        assert config.clip_epsilon == 0.1
        assert config.kl_coef == 0.05
        assert config.step_level_advantage is True

    def test_inherits_train_config(self):
        """应继承 TrainConfig 所有字段."""
        config = GRPOConfig(learning_rate=1e-6)
        assert config.learning_rate == 1e-6
        assert config.num_epochs == 3  # 继承默认值
