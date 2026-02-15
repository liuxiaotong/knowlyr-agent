"""DPO Trainer 集成测试（使用 tiny-gpt2）."""

import pytest

from agenttrainer.config import DPOConfig
from agenttrainer.trainers.dpo import DPOTrainer


class TestDPOTrainer:
    @pytest.mark.slow
    def test_train_completes(self, dpo_sample_file, small_model_name, tmp_path):
        """完整 DPO 训练应能跑通（1 epoch, 小模型）."""
        config = DPOConfig(
            model_name_or_path=small_model_name,
            train_file=str(dpo_sample_file),
            output_dir=str(tmp_path / "dpo_output"),
            num_epochs=1,
            batch_size=1,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
            beta=0.1,
        )

        trainer = DPOTrainer(config)
        trainer.train()

        # 检查最终模型输出
        final_dir = tmp_path / "dpo_output" / "final"
        assert final_dir.exists()
