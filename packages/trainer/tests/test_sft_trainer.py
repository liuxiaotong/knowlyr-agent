"""SFT Trainer 集成测试（使用 tiny-gpt2）."""

import json

import pytest

from agenttrainer.config import SFTConfig
from agenttrainer.trainers.sft import SFTTrainer


class TestSFTTrainer:
    @pytest.mark.slow
    def test_train_completes(self, sft_sample_file, small_model_name, tmp_path):
        """完整 SFT 训练应能跑通（1 epoch, 小模型）."""
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            output_dir=str(tmp_path / "sft_output"),
            num_epochs=1,
            batch_size=2,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
        )

        trainer = SFTTrainer(config)
        trainer.train()

        # 检查最终模型输出
        final_dir = tmp_path / "sft_output" / "final"
        assert final_dir.exists()

    @pytest.mark.slow
    def test_checkpoint_saved(self, sft_sample_file, small_model_name, tmp_path):
        """每 save_steps 应保存 checkpoint."""
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            output_dir=str(tmp_path / "sft_output"),
            num_epochs=1,
            batch_size=1,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=1,  # 每步保存
            bf16=False,
        )

        trainer = SFTTrainer(config)
        trainer.train()

        # 应该有 checkpoint 目录
        output_dir = tmp_path / "sft_output"
        checkpoints = list(output_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0


class TestAgentSFTTrainer:
    """Agent 模式 SFT 训练集成测试."""

    @pytest.mark.slow
    def test_agent_format_train(self, sft_sample_file, small_model_name, tmp_path):
        """agent_format=True 模式应能跑通."""
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            output_dir=str(tmp_path / "agent_sft_output"),
            num_epochs=1,
            batch_size=2,
            max_length=128,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
            agent_format=True,
            mask_observations=True,
        )

        trainer = SFTTrainer(config)
        trainer.train()

        final_dir = tmp_path / "agent_sft_output" / "final"
        assert final_dir.exists()

    @pytest.mark.slow
    def test_step_weighted_loss(self, tmp_path, small_model_name):
        """step_weighted_loss=True 应能跑通."""
        # 创建带 steps 的数据
        data_file = tmp_path / "agent_sft.jsonl"
        records = [
            {
                "instruction": "Fix the bug",
                "input": "",
                "steps": [
                    {"thought": "Read the code", "action": "read_file /a.py",
                     "observation": "def foo(): pass", "reward": 0.5},
                    {"thought": "Fix it", "action": "edit_file /a.py",
                     "observation": "File edited", "reward": 0.9},
                ],
                "task_id": "t-1",
                "reward": 0.7,
            },
            {
                "instruction": "Add tests",
                "input": "",
                "steps": [
                    {"thought": "Check existing tests", "action": "read_file /tests.py",
                     "observation": "No tests", "reward": 0.6},
                ],
                "task_id": "t-2",
                "reward": 0.6,
            },
        ]
        with open(data_file, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(data_file),
            output_dir=str(tmp_path / "weighted_output"),
            num_epochs=1,
            batch_size=2,
            max_length=128,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
            agent_format=True,
            mask_observations=True,
            step_weighted_loss=True,
        )

        trainer = SFTTrainer(config)
        trainer.train()

        final_dir = tmp_path / "weighted_output" / "final"
        assert final_dir.exists()
