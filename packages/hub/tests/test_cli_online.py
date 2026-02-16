"""测试 CLI evaluate / online 命令."""

import json
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from trajectoryhub.cli import main


# ── Mock 组件 ─────────────────────────────────────────────────────


class _MockEnv(AgentEnv):
    """测试用环境."""

    domain = "test"

    def __init__(self):
        self._step_count = 0

    def reset(self, *, task=None, seed=None) -> TimeStep:
        self._step_count = 0
        return TimeStep(observation="ready")

    def step(self, action: dict) -> TimeStep:
        self._step_count += 1
        tool = action.get("tool", "noop")
        return TimeStep(
            observation=f"{tool}-result",
            terminated=(tool == "submit"),
            info={"success": True} if tool == "submit" else {},
        )

    def close(self):
        pass

    @property
    def available_tools(self):
        return ["bash", "submit"]


def _make_mock_agent(obs: str) -> dict[str, Any]:
    """简单 1 步 submit agent."""
    return {"tool": "submit", "params": {}}


# ── evaluate 命令测试 ─────────────────────────────────────────────


class TestEvaluateCommand:
    """测试 evaluate CLI 命令."""

    def test_evaluate_no_model(self):
        """不提供 --model 应报错."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "evaluate", "--env", "knowlyr/conversation",
        ])
        assert result.exit_code != 0
        assert "请通过 --model 指定模型路径" in result.output

    def test_evaluate_with_mock(self, tmp_path):
        """使用 mock 数据测试评估命令完整流程."""
        # 模拟 evaluate_agent 返回结果
        mock_result = {
            "success_rate": 0.8,
            "avg_reward": 0.65,
            "std_reward": 0.12,
            "avg_steps": 3.5,
            "std_steps": 1.2,
            "min_reward": 0.1,
            "max_reward": 0.9,
            "reward_distribution": {"<0.25": 1, "0.25-0.5": 2, "0.5-0.75": 3, ">=0.75": 4},
            "n_episodes": 10,
            "episodes": [],
        }

        with patch("trajectoryhub.evaluate.evaluate_agent", return_value=mock_result), \
             patch("trajectoryhub.collect.make_reward_fn", return_value=lambda s, a: 0.5):
            runner = CliRunner()
            out_file = tmp_path / "eval.json"
            result = runner.invoke(main, [
                "evaluate",
                "--env", "knowlyr/conversation",
                "--model", "./my-model",
                "-n", "10",
                "-o", str(out_file),
            ])

            assert result.exit_code == 0
            assert "评估结果" in result.output
            assert "80.0%" in result.output
            assert "0.650" in result.output

            # 检查输出文件
            assert out_file.exists()
            saved = json.loads(out_file.read_text())
            assert saved["success_rate"] == 0.8
            assert "episodes" not in saved  # episodes 应被移除

    def test_evaluate_help(self):
        """--help 应正常显示."""
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "evaluate_agent" in result.output or "评估" in result.output


# ── online 命令测试 ──────────────────────────────────────────────


class TestOnlineCommand:
    """测试 online CLI 命令."""

    def test_online_no_model(self):
        """不提供 --model 应报错."""
        runner = CliRunner()
        result = runner.invoke(main, ["online"])
        assert result.exit_code != 0
        assert "请通过 --model 指定模型路径" in result.output

    def test_online_with_mock(self, tmp_path):
        """使用 mock 数据测试 online 命令."""
        from trajectoryhub.online import IterationStats

        mock_results = [
            IterationStats(iteration=0, n_episodes=5, success_rate=0.6, avg_reward=0.5, avg_steps=3),
            IterationStats(iteration=1, n_episodes=5, success_rate=0.8, avg_reward=0.7, avg_steps=2.5),
        ]

        with patch("trajectoryhub.online.online_training_loop", return_value=mock_results):
            runner = CliRunner()
            result = runner.invoke(main, [
                "online",
                "--model", "my-model",
                "--env", "knowlyr/conversation",
                "-n", "2",
                "-o", str(tmp_path / "loop"),
            ])

            assert result.exit_code == 0
            assert "训练循环完成" in result.output
            assert "Iter 0" in result.output
            assert "Iter 1" in result.output

    def test_online_help(self):
        """--help 应正常显示."""
        runner = CliRunner()
        result = runner.invoke(main, ["online", "--help"])
        assert result.exit_code == 0
        assert "在线训练循环" in result.output
