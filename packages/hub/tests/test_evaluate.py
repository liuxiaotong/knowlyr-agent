"""测试 evaluate.py — Hub 层评估桥接."""

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep


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


def _mock_agent(obs: str) -> dict[str, Any]:
    """简单 submit agent."""
    return {"tool": "submit", "params": {}}


def _fake_trainer_modules(*, evaluate_return=None, compare_return=None):
    """构造 fake agenttrainer 模块层级，无需实际安装 knowlyr-trainer.

    Returns:
        (modules_dict, mock_evaluate, mock_compare) 元组.
    """
    mock_evaluate = MagicMock(return_value=evaluate_return or {})
    mock_compare = MagicMock(return_value=compare_return or {})

    agent_eval = types.ModuleType("agenttrainer.eval.agent_eval")
    agent_eval.evaluate_agent = mock_evaluate
    agent_eval.compare_agents = mock_compare

    eval_pkg = types.ModuleType("agenttrainer.eval")
    eval_pkg.agent_eval = agent_eval

    trainer_pkg = types.ModuleType("agenttrainer")
    trainer_pkg.eval = eval_pkg

    modules = {
        "agenttrainer": trainer_pkg,
        "agenttrainer.eval": eval_pkg,
        "agenttrainer.eval.agent_eval": agent_eval,
    }
    return modules, mock_evaluate, mock_compare


# ── evaluate_agent 测试 ──────────────────────────────────────────


class TestEvaluateAgent:
    """evaluate_agent() 桥接层测试."""

    def test_with_agent_fn_and_env(self):
        """传入 agent_fn + env 实例应调用底层 evaluate."""
        mock_result = {
            "success_rate": 0.8,
            "avg_reward": 0.5,
            "n_episodes": 5,
        }
        modules, mock_eval, _ = _fake_trainer_modules(evaluate_return=mock_result)

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import evaluate_agent

            result = evaluate_agent(
                agent_fn=_mock_agent,
                env=_MockEnv(),
                n_episodes=5,
                max_steps=10,
            )

            assert result["success_rate"] == 0.8
            assert result["avg_reward"] == 0.5
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["agent_fn"] is _mock_agent
            assert call_kwargs["n_episodes"] == 5
            assert call_kwargs["max_steps"] == 10

    def test_with_model_path(self):
        """传入 model_path 应转发到底层."""
        mock_result = {"success_rate": 0.6, "avg_reward": 0.3}
        modules, mock_eval, _ = _fake_trainer_modules(evaluate_return=mock_result)

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import evaluate_agent

            evaluate_agent(
                model_path="./my-model",
                env_id="knowlyr/conversation",
                system_prompt="你好",
            )

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["model_path"] == "./my-model"
            assert call_kwargs["env_id"] == "knowlyr/conversation"
            assert call_kwargs["system_prompt"] == "你好"

    def test_with_reward_fn(self):
        """reward_fn 应被传递."""
        def my_reward(steps, action):
            return 0.5

        modules, mock_eval, _ = _fake_trainer_modules(
            evaluate_return={"success_rate": 0.5},
        )

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import evaluate_agent

            evaluate_agent(
                agent_fn=_mock_agent,
                env=_MockEnv(),
                reward_fn=my_reward,
            )

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["reward_fn"] is my_reward

    def test_with_tasks(self):
        """tasks 参数应被传递."""
        tasks = [{"id": "t1"}, {"id": "t2"}]
        modules, mock_eval, _ = _fake_trainer_modules(
            evaluate_return={"success_rate": 0.5},
        )

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import evaluate_agent

            evaluate_agent(
                agent_fn=_mock_agent,
                env=_MockEnv(),
                tasks=tasks,
            )

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["tasks"] is tasks

    def test_trainer_not_installed(self):
        """knowlyr-trainer 未安装时应抛 RuntimeError."""
        with patch.dict("sys.modules", {
            "agenttrainer": None,
            "agenttrainer.eval": None,
            "agenttrainer.eval.agent_eval": None,
        }):
            from trajectoryhub.evaluate import evaluate_agent

            with pytest.raises(RuntimeError, match="knowlyr-trainer"):
                evaluate_agent(agent_fn=_mock_agent, env=_MockEnv())

    def test_default_parameters(self):
        """默认参数应正确传递."""
        modules, mock_eval, _ = _fake_trainer_modules(
            evaluate_return={"success_rate": 0.0},
        )

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import evaluate_agent

            evaluate_agent(agent_fn=_mock_agent, env=_MockEnv())

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["n_episodes"] == 10
            assert call_kwargs["max_steps"] == 30
            assert call_kwargs["system_prompt"] == ""
            assert call_kwargs["reward_fn"] is None
            assert call_kwargs["tasks"] is None


# ── compare_agents 测试 ──────────────────────────────────────────


class TestCompareAgents:
    """compare_agents() 桥接层测试."""

    def test_basic_compare(self):
        """多 agent 对比应转发到底层."""
        def agent_a(obs):
            return {"tool": "submit"}

        def agent_b(obs):
            return {"tool": "submit"}

        mock_result = {
            "agent_a": {"success_rate": 0.8},
            "agent_b": {"success_rate": 0.6},
        }
        modules, _, mock_compare = _fake_trainer_modules(compare_return=mock_result)

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import compare_agents

            agents = {"agent_a": agent_a, "agent_b": agent_b}
            result = compare_agents(
                agents=agents,
                env=_MockEnv(),
                n_episodes=5,
            )

            assert result["agent_a"]["success_rate"] == 0.8
            assert result["agent_b"]["success_rate"] == 0.6
            call_kwargs = mock_compare.call_args[1]
            assert call_kwargs["agents"] is agents
            assert call_kwargs["n_episodes"] == 5

    def test_compare_with_env_id(self):
        """env_id 参数应被传递."""
        modules, _, mock_compare = _fake_trainer_modules(compare_return={})

        with patch.dict("sys.modules", modules):
            from trajectoryhub.evaluate import compare_agents

            compare_agents(
                agents={"a": _mock_agent},
                env_id="knowlyr/conversation",
                max_steps=20,
            )

            call_kwargs = mock_compare.call_args[1]
            assert call_kwargs["env_id"] == "knowlyr/conversation"
            assert call_kwargs["max_steps"] == 20

    def test_compare_trainer_not_installed(self):
        """knowlyr-trainer 未安装时应抛 RuntimeError."""
        with patch.dict("sys.modules", {
            "agenttrainer": None,
            "agenttrainer.eval": None,
            "agenttrainer.eval.agent_eval": None,
        }):
            from trajectoryhub.evaluate import compare_agents

            with pytest.raises(RuntimeError, match="knowlyr-trainer"):
                compare_agents(agents={"a": _mock_agent}, env=_MockEnv())
