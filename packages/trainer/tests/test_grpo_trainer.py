"""GRPO Trainer 集成测试（使用 tiny-gpt2）."""

import pytest

from agenttrainer.config import GRPOConfig
from agenttrainer.trainers.grpo import GRPOTrainer


class TestGRPOTrainer:
    @pytest.mark.slow
    def test_train_completes(self, grpo_sample_file, small_model_name, tmp_path):
        """完整 GRPO 训练应能跑通（1 epoch, 小模型）."""
        config = GRPOConfig(
            model_name_or_path=small_model_name,
            train_file=str(grpo_sample_file),
            output_dir=str(tmp_path / "grpo_output"),
            num_epochs=1,
            batch_size=1,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
            group_size=4,
            clip_epsilon=0.2,
            kl_coef=0.01,
        )

        trainer = GRPOTrainer(config)
        trainer.train()

        # 检查最终模型输出
        final_dir = tmp_path / "grpo_output" / "final"
        assert final_dir.exists()
