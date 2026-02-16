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


class TestSFTTrainerEval:
    """SFT 验证评估测试."""

    @pytest.mark.slow
    def test_eval_file_runs(self, sft_sample_file, small_model_name, tmp_path):
        """eval_file 非空时应执行验证评估."""
        # 用同一文件做 train 和 eval
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            eval_file=str(sft_sample_file),
            output_dir=str(tmp_path / "sft_eval_output"),
            num_epochs=1,
            batch_size=2,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
        )

        trainer = SFTTrainer(config)
        trainer.train()  # 不应报错

        final_dir = tmp_path / "sft_eval_output" / "final"
        assert final_dir.exists()

    @pytest.mark.slow
    def test_no_eval_file_skips(self, sft_sample_file, small_model_name, tmp_path):
        """eval_file 为空时不应执行验证评估."""
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            eval_file=None,
            output_dir=str(tmp_path / "sft_no_eval"),
            num_epochs=1,
            batch_size=2,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
        )

        trainer = SFTTrainer(config)
        trainer.train()  # 不应报错

        final_dir = tmp_path / "sft_no_eval" / "final"
        assert final_dir.exists()

    @pytest.mark.slow
    def test_save_best_model(self, sft_sample_file, small_model_name, tmp_path):
        """save_best_model=True 应保存 best 目录."""
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            eval_file=str(sft_sample_file),
            output_dir=str(tmp_path / "sft_best"),
            num_epochs=2,
            batch_size=2,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
            save_best_model=True,
        )

        trainer = SFTTrainer(config)
        trainer.train()

        best_dir = tmp_path / "sft_best" / "best"
        assert best_dir.exists()

    @pytest.mark.slow
    def test_early_stopping(self, sft_sample_file, small_model_name, tmp_path):
        """early_stopping_patience=1 应在验证 loss 不改善时提前停止."""
        config = SFTConfig(
            model_name_or_path=small_model_name,
            train_file=str(sft_sample_file),
            eval_file=str(sft_sample_file),
            output_dir=str(tmp_path / "sft_early"),
            num_epochs=10,  # 设大，看是否提前停止
            batch_size=2,
            max_length=64,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=0,
            bf16=False,
            early_stopping_patience=1,
        )

        trainer = SFTTrainer(config)
        trainer.train()

        # 应该正常完成（提前停止或跑完）
        final_dir = tmp_path / "sft_early" / "final"
        assert final_dir.exists()
