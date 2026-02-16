"""测试 agent 级别评估 — evaluate_agent + compare_agents."""

import pytest

from agenttrainer.eval.agent_eval import evaluate_agent, compare_agents, _compute_stats


# ── Mock 环境和 agent ──────────────────────────────────────────


class _MockEnv:
    """用于测试的 mock 环境."""

    domain = "conversation"

    def __init__(self, *, success_at_step: int = 2):
        self._success_at = success_at_step
        self._step_count = 0

    def reset(self, *, task=None, seed=None):
        from knowlyrcore.timestep import TimeStep

        self._step_count = 0
        msg = task or "测试任务"
        return TimeStep(observation=msg)

    def step(self, action):
        from knowlyrcore.timestep import TimeStep

        self._step_count += 1
        tool = action.get("tool", "think")

        if tool in ("respond", "submit") or self._step_count >= self._success_at:
            return TimeStep(
                observation="完成",
                terminated=True,
                info={"success": True},
            )

        return TimeStep(
            observation=f"步骤 {self._step_count}",
            info={"step": self._step_count},
        )

    @property
    def available_tools(self):
        return ["think", "respond"]

    def close(self):
        pass

    def get_trajectory(self):
        """RecorderWrapper 需要此方法，但 mock 不使用."""
        return {"steps": [], "outcome": {"success": True}}


def _make_simple_agent(respond_at: int = 2):
    """创建简单 agent: 在第 respond_at 步回复."""
    call_count = [0]

    def agent(observation: str) -> dict:
        call_count[0] += 1
        if call_count[0] >= respond_at:
            call_count[0] = 0
            return {"tool": "respond", "params": {"message": "完成"}}
        return {"tool": "think", "params": {"thought": "思考中"}}

    return agent


# ── _compute_stats 测试 ──────────────────────────────────────────


class TestComputeStats:
    """测试统计计算."""

    def test_empty_episodes(self):
        """空 episodes 返回零值."""
        result = _compute_stats([], 0)
        assert result["success_rate"] == 0.0
        assert result["avg_reward"] == 0.0
        assert result["n_episodes"] == 0
        assert result["episodes"] == []

    def test_all_success(self):
        """全部成功."""
        episodes = [
            {"episode": 0, "success": True, "total_reward": 0.8, "n_steps": 3, "outcome": {}},
            {"episode": 1, "success": True, "total_reward": 0.9, "n_steps": 2, "outcome": {}},
        ]
        result = _compute_stats(episodes, 2)
        assert result["success_rate"] == 1.0
        assert result["avg_reward"] == pytest.approx(0.85)
        assert result["avg_steps"] == pytest.approx(2.5)
        assert result["n_episodes"] == 2

    def test_mixed_success(self):
        """部分成功."""
        episodes = [
            {"episode": 0, "success": True, "total_reward": 0.9, "n_steps": 2, "outcome": {}},
            {"episode": 1, "success": False, "total_reward": 0.1, "n_steps": 5, "outcome": {}},
            {"episode": 2, "success": True, "total_reward": 0.7, "n_steps": 3, "outcome": {}},
            {"episode": 3, "success": False, "total_reward": 0.2, "n_steps": 6, "outcome": {}},
        ]
        result = _compute_stats(episodes, 4)
        assert result["success_rate"] == 0.5
        assert result["min_reward"] == pytest.approx(0.1)
        assert result["max_reward"] == pytest.approx(0.9)
        assert result["std_reward"] > 0

    def test_reward_distribution(self):
        """reward 分布计算."""
        episodes = [
            {"episode": 0, "success": True, "total_reward": 0.1, "n_steps": 5, "outcome": {}},
            {"episode": 1, "success": True, "total_reward": 0.3, "n_steps": 4, "outcome": {}},
            {"episode": 2, "success": True, "total_reward": 0.6, "n_steps": 3, "outcome": {}},
            {"episode": 3, "success": True, "total_reward": 0.9, "n_steps": 2, "outcome": {}},
        ]
        result = _compute_stats(episodes, 4)
        dist = result["reward_distribution"]
        assert dist["<0.25"] == 1
        assert dist["0.25-0.5"] == 1
        assert dist["0.5-0.75"] == 1
        assert dist[">=0.75"] == 1

    def test_single_episode(self):
        """单个 episode 不崩溃 (std=0)."""
        episodes = [
            {"episode": 0, "success": True, "total_reward": 0.5, "n_steps": 3, "outcome": {}},
        ]
        result = _compute_stats(episodes, 1)
        assert result["std_reward"] == 0.0
        assert result["std_steps"] == 0.0


# ── evaluate_agent 测试 ──────────────────────────────────────────


class TestEvaluateAgent:
    """测试 evaluate_agent 函数."""

    def test_missing_both_args(self):
        """agent_fn 和 model_path 都未提供时报错."""
        with pytest.raises(ValueError, match="必须提供"):
            evaluate_agent()

    def test_missing_env(self):
        """env 和 env_id 都未提供时报错."""
        with pytest.raises(ValueError, match="必须提供 env_id 或 env"):
            evaluate_agent(agent_fn=lambda obs: {"tool": "respond"})

    def test_basic_evaluation(self):
        """基本评估: mock env + simple agent."""
        agent = _make_simple_agent(respond_at=2)
        env = _MockEnv(success_at_step=3)

        result = evaluate_agent(
            agent_fn=agent,
            env=env,
            n_episodes=5,
            max_steps=10,
        )

        assert "success_rate" in result
        assert "avg_reward" in result
        assert "avg_steps" in result
        assert "reward_distribution" in result
        assert result["n_episodes"] == 5
        assert len(result["episodes"]) == 5

    def test_evaluation_with_tasks(self):
        """带 tasks 列表的评估."""
        agent = _make_simple_agent(respond_at=1)
        env = _MockEnv(success_at_step=2)

        tasks = ["任务A", "任务B", "任务C"]
        result = evaluate_agent(
            agent_fn=agent,
            env=env,
            n_episodes=6,
            max_steps=10,
            tasks=tasks,
        )

        # 6 episodes 使用 3 个 tasks 循环
        assert result["n_episodes"] == 6
        assert len(result["episodes"]) == 6

    def test_evaluation_with_reward_fn(self):
        """带 reward_fn 的评估."""
        agent = _make_simple_agent(respond_at=2)
        env = _MockEnv(success_at_step=3)

        def my_reward(steps, action):
            if action.get("tool") == "respond":
                return 1.0
            return 0.1

        result = evaluate_agent(
            agent_fn=agent,
            env=env,
            n_episodes=3,
            max_steps=10,
            reward_fn=my_reward,
        )

        # 有 reward_fn 时 reward 应非零
        assert result["n_episodes"] == 3

    def test_all_episodes_succeed(self):
        """所有 episode 都成功时 success_rate=1.0."""
        # 直接 respond 的 agent
        agent = _make_simple_agent(respond_at=1)
        env = _MockEnv(success_at_step=2)

        result = evaluate_agent(
            agent_fn=agent,
            env=env,
            n_episodes=3,
            max_steps=10,
        )

        assert result["success_rate"] == 1.0


# ── compare_agents 测试 ──────────────────────────────────────────


class TestCompareAgents:
    """测试 compare_agents 函数."""

    def test_compare_two_agents(self):
        """对比两个 agent."""
        fast_agent = _make_simple_agent(respond_at=1)
        slow_agent = _make_simple_agent(respond_at=3)
        env = _MockEnv(success_at_step=5)

        results = compare_agents(
            agents={"fast": fast_agent, "slow": slow_agent},
            env=env,
            n_episodes=3,
            max_steps=10,
        )

        assert "fast" in results
        assert "slow" in results
        assert results["fast"]["n_episodes"] == 3
        assert results["slow"]["n_episodes"] == 3
        # fast agent 步数应少于 slow agent
        assert results["fast"]["avg_steps"] <= results["slow"]["avg_steps"]

    def test_compare_single_agent(self):
        """只有一个 agent 时也能正常工作."""
        agent = _make_simple_agent(respond_at=1)
        env = _MockEnv(success_at_step=2)

        results = compare_agents(
            agents={"only_one": agent},
            env=env,
            n_episodes=2,
        )

        assert "only_one" in results
        assert len(results) == 1
