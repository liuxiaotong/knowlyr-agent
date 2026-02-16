"""集成测试 — Wrappers + collect() 跨包组合."""

from typing import Any

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep
from knowlyrcore.wrappers import (
    EpisodeStatisticsWrapper,
    ObservationTruncateWrapper,
)

from trajectoryhub.collect import collect


# ── 共享环境 ─────────────────────────────────────────────────────


class VerboseEnv(AgentEnv):
    """产生长 observation 的测试环境."""

    domain = "test"

    def __init__(self):
        self._step_count = 0

    def reset(self, *, task=None, seed=None) -> TimeStep:
        self._step_count = 0
        return TimeStep(observation="START " + "x" * 500)

    def step(self, action: dict) -> TimeStep:
        self._step_count += 1
        tool = action.get("tool", "noop")
        # 产生很长的 observation
        long_output = f"{tool}-result: " + "y" * 800
        return TimeStep(
            observation=long_output,
            terminated=(tool == "submit"),
            info={"success": True} if tool == "submit" else {},
        )

    def close(self):
        pass

    @property
    def available_tools(self):
        return ["bash", "submit"]


def _make_agent(submit_after: int = 3):
    """N 步后 submit 的 agent."""
    call_count = 0

    def agent(obs: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count >= submit_after:
            call_count = 0
            return {"tool": "submit", "params": {}}
        return {"tool": "bash", "params": {"command": "ls"}}

    return agent


# ── EpisodeStatistics + collect ──────────────────────────────


class TestEpisodeStatisticsWithCollect:
    """EpisodeStatisticsWrapper 与 collect() 的集成."""

    def test_statistics_no_interference(self):
        """EpisodeStatistics 不干扰正常收集流程."""
        env = EpisodeStatisticsWrapper(VerboseEnv())
        trajs = collect(
            env,
            agent=_make_agent(submit_after=3),
            n_episodes=2,
            max_steps=10,
        )

        assert len(trajs) == 2
        for traj in trajs:
            assert len(traj["steps"]) == 3
            assert traj["outcome"]["success"] is True

    def test_statistics_multiple_episodes(self):
        """多 episode 收集时 EpisodeStatistics 正常工作."""
        env = EpisodeStatisticsWrapper(VerboseEnv())
        trajs = collect(
            env,
            agent=_make_agent(submit_after=2),
            n_episodes=3,
            max_steps=10,
        )

        assert len(trajs) == 3
        for traj in trajs:
            assert len(traj["steps"]) == 2
            assert traj["outcome"]["success"] is True


# ── ObservationTruncate + collect ────────────────────────────


class TestObservationTruncateWithCollect:
    """ObservationTruncateWrapper 与 collect() 的集成."""

    def test_long_observations_truncated(self):
        """过长 observation 应被截断."""
        env = ObservationTruncateWrapper(VerboseEnv(), max_chars=100)
        trajs = collect(
            env,
            agent=_make_agent(submit_after=2),
            n_episodes=1,
            max_steps=10,
        )

        assert len(trajs) == 1
        assert trajs[0]["outcome"]["success"] is True

    def test_short_observations_unchanged(self):
        """短 observation 不应被截断."""
        # 用大 max_chars 确保不截断
        env = ObservationTruncateWrapper(VerboseEnv(), max_chars=10000)
        trajs = collect(
            env,
            agent=_make_agent(submit_after=2),
            n_episodes=1,
            max_steps=10,
        )

        assert trajs[0]["outcome"]["success"] is True
        assert len(trajs[0]["steps"]) == 2


# ── 多 Wrapper 组合 + collect ────────────────────────────────


class TestCombinedWrappersWithCollect:
    """多 Wrapper 叠加 + collect() 的组合测试."""

    def test_statistics_plus_truncate(self):
        """EpisodeStatistics + ObservationTruncate + collect 三层组合."""
        env = VerboseEnv()
        env = ObservationTruncateWrapper(env, max_chars=200)
        env = EpisodeStatisticsWrapper(env)

        trajs = collect(
            env,
            agent=_make_agent(submit_after=3),
            n_episodes=2,
            max_steps=10,
        )

        assert len(trajs) == 2
        for traj in trajs:
            assert traj["outcome"]["success"] is True
            assert len(traj["steps"]) == 3

    def test_reward_fn_with_wrappers(self):
        """Wrapper + collect + reward_fn 全组合."""
        env = EpisodeStatisticsWrapper(VerboseEnv())

        def reward_fn(steps, action):
            return 0.8 if action.get("tool") == "submit" else 0.2

        trajs = collect(
            env,
            agent=_make_agent(submit_after=2),
            n_episodes=2,
            max_steps=10,
            reward_fn=reward_fn,
        )

        assert len(trajs) == 2
        for traj in trajs:
            rewards = [s["reward"] for s in traj["steps"]]
            assert rewards[-1] == 0.8  # submit 步
            assert rewards[0] == 0.2   # bash 步
