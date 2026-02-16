"""GRPO Trainer 集成测试（使用 tiny-gpt2）."""

import json

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

    @pytest.mark.slow
    def test_step_level_advantage(self, tmp_path, small_model_name):
        """step_level_advantage=True 应能跑通."""
        # 创建带 step_rewards 的 GRPO 数据
        data_file = tmp_path / "grpo_step.jsonl"
        groups = [
            {
                "task_id": "task-001",
                "prompt": "Fix the bug",
                "trajectories": [
                    {
                        "response": "Read then edit",
                        "reward": 0.9,
                        "step_rewards": [0.4, 0.9],
                    },
                    {
                        "response": "Just bash it",
                        "reward": 0.3,
                        "step_rewards": [0.3, 0.1],
                    },
                    {
                        "response": "Read read read",
                        "reward": 0.5,
                        "step_rewards": [0.5, 0.5],
                    },
                ],
            },
        ]
        with open(data_file, "w") as f:
            for grp in groups:
                f.write(json.dumps(grp) + "\n")

        config = GRPOConfig(
            model_name_or_path=small_model_name,
            train_file=str(data_file),
            output_dir=str(tmp_path / "grpo_step_output"),
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
            step_level_advantage=True,
        )

        trainer = GRPOTrainer(config)
        trainer.train()

        final_dir = tmp_path / "grpo_step_output" / "final"
        assert final_dir.exists()

    @pytest.mark.slow
    def test_eval_file(self, grpo_sample_file, small_model_name, tmp_path):
        """eval_file 非空时应执行验证评估."""
        config = GRPOConfig(
            model_name_or_path=small_model_name,
            train_file=str(grpo_sample_file),
            eval_file=str(grpo_sample_file),
            output_dir=str(tmp_path / "grpo_eval_output"),
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

        final_dir = tmp_path / "grpo_eval_output" / "final"
        assert final_dir.exists()
