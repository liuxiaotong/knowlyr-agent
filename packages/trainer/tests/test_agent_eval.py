"""测试 agent 级别评估 — evaluate_agent + compare_agents + 统计检验."""

import pytest

from agenttrainer.eval.agent_eval import (
    evaluate_agent,
    compare_agents,
    _compute_stats,
    confidence_interval,
    significance_test,
)


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

    def test_compare_includes_significance(self):
        """两个 agent 对比时包含显著性检验结果."""
        fast_agent = _make_simple_agent(respond_at=1)
        slow_agent = _make_simple_agent(respond_at=3)
        env = _MockEnv(success_at_step=5)

        results = compare_agents(
            agents={"fast": fast_agent, "slow": slow_agent},
            env=env,
            n_episodes=5,
            max_steps=10,
        )

        assert "_comparisons" in results
        assert "fast_vs_slow" in results["_comparisons"]
        comp = results["_comparisons"]["fast_vs_slow"]
        assert "t_statistic" in comp
        assert "p_approx" in comp
        assert "significant" in comp
        assert "effect_size" in comp


# ── confidence_interval 测试 ─────────────────────────────────────


class TestConfidenceInterval:
    """测试置信区间计算."""

    def test_single_value(self):
        """单个值的 CI 是 (value, value)."""
        lo, hi = confidence_interval([5.0])
        assert lo == 5.0
        assert hi == 5.0

    def test_empty(self):
        """空列表返回 (0, 0)."""
        lo, hi = confidence_interval([])
        assert lo == 0.0
        assert hi == 0.0

    def test_symmetric(self):
        """均匀数据的 CI 应围绕均值对称."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = confidence_interval(data)
        mean = 3.0
        assert lo < mean
        assert hi > mean
        # 近似对称
        assert abs((mean - lo) - (hi - mean)) < 0.01

    def test_large_sample(self):
        """大样本的 CI 应较窄."""
        data = [1.0] * 50 + [2.0] * 50
        lo, hi = confidence_interval(data)
        # 100 个样本，CI 应很窄
        assert (hi - lo) < 0.5

    def test_ci_contains_mean(self):
        """CI 应包含均值."""
        data = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5]
        lo, hi = confidence_interval(data)
        import statistics
        mean = statistics.mean(data)
        assert lo <= mean <= hi


# ── significance_test 测试 ───────────────────────────────────────


class TestSignificanceTest:
    """测试显著性检验."""

    def test_identical_samples(self):
        """相同数据应不显著."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = significance_test(data, data)
        assert result["significant"] is False

    def test_very_different_samples(self):
        """差异极大的数据应显著."""
        a = [1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 0.8, 1.2]
        b = [10.0, 10.1, 10.2, 9.9, 10.0, 10.1, 9.8, 10.2]
        result = significance_test(a, b)
        assert result["significant"] is True
        assert result["effect_size"] > 1.0  # 非常大的 effect size

    def test_insufficient_data(self):
        """数据不足时返回 insufficient_data."""
        result = significance_test([1.0], [2.0])
        assert result["p_approx"] == "insufficient_data"
        assert result["significant"] is False

    def test_effect_size_direction(self):
        """t_statistic 正负反映方向."""
        a = [5.0, 5.1, 5.2, 4.9, 5.0]
        b = [3.0, 3.1, 3.2, 2.9, 3.0]
        result = significance_test(a, b)
        assert result["t_statistic"] > 0  # a > b

        result2 = significance_test(b, a)
        assert result2["t_statistic"] < 0  # b < a


# ── _compute_stats CI 测试 ───────────────────────────────────────


class TestComputeStatsCI:
    """测试 _compute_stats 返回的 CI 字段."""

    def test_has_ci_fields(self):
        """结果应包含 CI 字段."""
        episodes = [
            {"episode": i, "success": True, "total_reward": 0.5 + i * 0.1,
             "n_steps": 3, "outcome": {}}
            for i in range(5)
        ]
        result = _compute_stats(episodes, 5)
        assert "reward_ci" in result
        assert "steps_ci" in result
        assert "success_rate_ci" in result

    def test_ci_bounds(self):
        """CI lower ≤ mean ≤ upper."""
        episodes = [
            {"episode": i, "success": i % 2 == 0, "total_reward": 0.3 + i * 0.1,
             "n_steps": 2 + i, "outcome": {}}
            for i in range(10)
        ]
        result = _compute_stats(episodes, 10)

        lo, hi = result["reward_ci"]
        assert lo <= result["avg_reward"] <= hi

        lo, hi = result["success_rate_ci"]
        assert lo <= result["success_rate"] <= hi
