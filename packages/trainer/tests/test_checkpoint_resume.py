"""测试 checkpoint 恢复功能 — load_training_state + _maybe_resume."""

from pathlib import Path

import pytest
import torch

from agenttrainer.models.checkpoint import load_training_state, save_checkpoint


# ── load_training_state 测试 ─────────────────────────────────────


class TestLoadTrainingState:
    """load_training_state() 单元测试."""

    def test_load_valid_state(self, tmp_path):
        """有效 checkpoint 应正确加载 global_step."""
        ckpt_dir = tmp_path / "checkpoint-10"
        ckpt_dir.mkdir()
        torch.save(
            {"optimizer": None, "scheduler": None, "global_step": 10},
            ckpt_dir / "training_state.pt",
        )

        state = load_training_state(str(ckpt_dir))
        assert state is not None
        assert state["global_step"] == 10

    def test_missing_state_file(self, tmp_path):
        """无 training_state.pt 时返回 None."""
        ckpt_dir = tmp_path / "checkpoint-empty"
        ckpt_dir.mkdir()

        state = load_training_state(str(ckpt_dir))
        assert state is None

    def test_nonexistent_dir(self, tmp_path):
        """不存在的目录返回 None."""
        state = load_training_state(str(tmp_path / "nonexistent"))
        assert state is None

    def test_state_contains_optimizer(self, tmp_path):
        """state 应包含 optimizer dict."""
        ckpt_dir = tmp_path / "checkpoint-5"
        ckpt_dir.mkdir()

        # 模拟 optimizer state
        fake_opt_state = {"state": {}, "param_groups": [{"lr": 1e-5}]}
        torch.save(
            {"optimizer": fake_opt_state, "scheduler": None, "global_step": 5},
            ckpt_dir / "training_state.pt",
        )

        state = load_training_state(str(ckpt_dir))
        assert state["optimizer"] == fake_opt_state
        assert state["global_step"] == 5


# ── _maybe_resume 测试 ───────────────────────────────────────────


class TestMaybeResume:
    """BaseTrainer._maybe_resume() 单元测试."""

    def test_no_resume_path(self):
        """resume_from_checkpoint 为空时返回 0."""
        from agenttrainer.config import TrainConfig
        from agenttrainer.trainers.base import BaseTrainer

        # 使用匿名子类来实例化
        class _DummyTrainer(BaseTrainer):
            def _train_loop(self):
                pass

        config = TrainConfig(resume_from_checkpoint=None)
        trainer = _DummyTrainer(config)

        # 创建 dummy optimizer 和 scheduler
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        step = trainer._maybe_resume(optimizer, scheduler=None)
        assert step == 0

    def test_resume_from_valid_checkpoint(self, tmp_path):
        """从有效 checkpoint 恢复 global_step."""
        from agenttrainer.config import TrainConfig
        from agenttrainer.trainers.base import BaseTrainer

        class _DummyTrainer(BaseTrainer):
            def _train_loop(self):
                pass

        # 准备 checkpoint
        ckpt_dir = tmp_path / "checkpoint-42"
        ckpt_dir.mkdir()

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # 保存 optimizer state
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": None,
                "global_step": 42,
            },
            ckpt_dir / "training_state.pt",
        )

        config = TrainConfig(resume_from_checkpoint=str(ckpt_dir))
        trainer = _DummyTrainer(config)

        # 新 optimizer
        new_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        step = trainer._maybe_resume(new_optimizer, scheduler=None)

        assert step == 42

    def test_resume_missing_state_returns_zero(self, tmp_path):
        """checkpoint 目录存在但无 training_state.pt 时返回 0."""
        from agenttrainer.config import TrainConfig
        from agenttrainer.trainers.base import BaseTrainer

        class _DummyTrainer(BaseTrainer):
            def _train_loop(self):
                pass

        ckpt_dir = tmp_path / "empty-checkpoint"
        ckpt_dir.mkdir()

        config = TrainConfig(resume_from_checkpoint=str(ckpt_dir))
        trainer = _DummyTrainer(config)

        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        step = trainer._maybe_resume(optimizer, scheduler=None)

        assert step == 0


# ── config 字段测试 ───────────────────────────────────────────────


class TestResumeConfig:
    """TrainConfig.resume_from_checkpoint 字段测试."""

    def test_default_none(self):
        """默认值为 None."""
        from agenttrainer.config import TrainConfig
        config = TrainConfig()
        assert config.resume_from_checkpoint is None

    def test_set_path(self, tmp_path):
        """可以设置 checkpoint 路径."""
        from agenttrainer.config import TrainConfig
        path = str(tmp_path / "ckpt")
        config = TrainConfig(resume_from_checkpoint=path)
        assert config.resume_from_checkpoint == path

    def test_sft_config_inherits(self):
        """SFTConfig 应继承 resume_from_checkpoint."""
        from agenttrainer.config import SFTConfig
        config = SFTConfig(resume_from_checkpoint="/some/path")
        assert config.resume_from_checkpoint == "/some/path"
