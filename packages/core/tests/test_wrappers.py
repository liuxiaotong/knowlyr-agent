"""测试 Wrapper 套件 — MaxSteps / Timeout / Reward / Recorder / Statistics / Truncate."""

import time

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep
from knowlyrcore.wrappers import (
    EpisodeStatisticsWrapper,
    MaxStepsWrapper,
    ObservationTruncateWrapper,
    RecorderWrapper,
    RewardWrapper,
    TimeoutWrapper,
)


# ── MockEnv ─────────────────────────────────────────────────────────


class MockEnv(AgentEnv):
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
            observation=f"{tool}-output",
            terminated=(tool == "submit"),
            info={"exit_code": 0, "step": self._step_count},
        )

    @property
    def available_tools(self):
        return ["bash", "edit", "submit"]


# ── MaxStepsWrapper ────────────────────────────────────────────────


class TestMaxStepsWrapper:
    """MaxStepsWrapper 测试."""

    def test_normal_steps(self):
        env = MaxStepsWrapper(MockEnv(), max_steps=5)
        env.reset()
        ts = env.step({"tool": "bash"})
        assert ts.truncated is False
        assert env.step_count == 1

    def test_truncated_at_limit(self):
        env = MaxStepsWrapper(MockEnv(), max_steps=3)
        env.reset()
        env.step({"tool": "bash"})
        env.step({"tool": "bash"})
        ts = env.step({"tool": "bash"})
        assert ts.truncated is True
        assert env.step_count == 3

    def test_reset_clears_count(self):
        env = MaxStepsWrapper(MockEnv(), max_steps=5)
        env.reset()
        env.step({"tool": "bash"})
        env.step({"tool": "bash"})
        env.reset()
        assert env.step_count == 0

    def test_terminated_before_limit(self):
        """submit 导致 terminated，不应被 truncated 覆盖."""
        env = MaxStepsWrapper(MockEnv(), max_steps=5)
        env.reset()
        ts = env.step({"tool": "submit"})
        assert ts.terminated is True
        assert ts.truncated is False

    def test_full_episode(self):
        env = MaxStepsWrapper(MockEnv(), max_steps=3)
        ts = env.reset()
        steps = 0
        while not ts.done:
            ts = env.step({"tool": "bash"})
            steps += 1
        assert steps == 3


# ── TimeoutWrapper ─────────────────────────────────────────────────


class TestTimeoutWrapper:
    """TimeoutWrapper 测试."""

    def test_no_timeout(self):
        env = TimeoutWrapper(MockEnv(), timeout_seconds=10.0)
        env.reset()
        ts = env.step({"tool": "bash"})
        assert ts.truncated is False

    def test_timeout_triggers(self):
        env = TimeoutWrapper(MockEnv(), timeout_seconds=0.0)
        env.reset()
        ts = env.step({"tool": "bash"})
        assert ts.truncated is True
        assert ts.info.get("timeout") is True

    def test_elapsed(self):
        env = TimeoutWrapper(MockEnv(), timeout_seconds=10.0)
        env.reset()
        time.sleep(0.01)
        assert env.elapsed >= 0.01

    def test_reset_resets_timer(self):
        env = TimeoutWrapper(MockEnv(), timeout_seconds=10.0)
        env.reset()
        time.sleep(0.02)
        env.reset()
        assert env.elapsed < 0.02


# ── RewardWrapper ──────────────────────────────────────────────────


class TestRewardWrapper:
    """RewardWrapper 测试."""

    def test_constant_reward(self):
        def constant_reward(steps, action):
            return 1.0

        env = RewardWrapper(MockEnv(), reward_fn=constant_reward)
        env.reset()
        ts = env.step({"tool": "bash"})
        assert ts.reward == 1.0

    def test_step_count_reward(self):
        """reward 递减: 1.0 / step_count."""
        def decreasing_reward(steps, action):
            return 1.0 / len(steps)

        env = RewardWrapper(MockEnv(), reward_fn=decreasing_reward)
        env.reset()
        ts1 = env.step({"tool": "bash"})
        ts2 = env.step({"tool": "bash"})
        assert ts1.reward == 1.0
        assert ts2.reward == 0.5

    def test_tool_based_reward(self):
        """根据工具类型给不同 reward."""
        def tool_reward(steps, action):
            return 0.0 if action.get("tool") == "bash" else 1.0

        env = RewardWrapper(MockEnv(), reward_fn=tool_reward)
        env.reset()
        ts1 = env.step({"tool": "bash"})
        ts2 = env.step({"tool": "edit"})
        assert ts1.reward == 0.0
        assert ts2.reward == 1.0

    def test_reset_clears_steps(self):
        def count_reward(steps, action):
            return float(len(steps))

        env = RewardWrapper(MockEnv(), reward_fn=count_reward)
        env.reset()
        env.step({"tool": "bash"})
        env.step({"tool": "bash"})
        env.reset()
        ts = env.step({"tool": "bash"})
        assert ts.reward == 1.0  # 重置后是第 1 步


# ── RecorderWrapper ────────────────────────────────────────────────


class TestRecorderWrapper:
    """RecorderWrapper 测试."""

    def test_records_steps(self):
        env = RecorderWrapper(MockEnv(), agent_name="test-agent")
        env.reset()
        env.step({"tool": "bash", "params": {"command": "ls"}})
        env.step({"tool": "edit", "params": {"path": "f.py"}})

        traj = env.get_trajectory()
        assert len(traj["steps"]) == 2
        assert traj["steps"][0]["tool"] == "bash"
        assert traj["steps"][1]["tool"] == "edit"

    def test_trajectory_metadata(self):
        env = RecorderWrapper(MockEnv(), agent_name="my-agent", model_name="gpt-4o")
        env.reset()
        env.step({"tool": "submit"})

        traj = env.get_trajectory()
        assert traj["agent"] == "my-agent"
        assert traj["model"] == "gpt-4o"
        assert traj["metadata"]["domain"] == "test"

    def test_trajectory_outcome(self):
        env = RecorderWrapper(MockEnv())
        env.reset()
        env.step({"tool": "bash"})
        env.step({"tool": "submit"})

        traj = env.get_trajectory()
        assert traj["outcome"]["success"] is True
        assert traj["outcome"]["total_steps"] == 2
        assert traj["outcome"]["terminated"] is True

    def test_trajectory_with_task(self):
        """传入 task 时应记录 task_id 和 description."""
        class FakeTask:
            task_id = "task-001"
            description = "Fix the bug"

        env = RecorderWrapper(MockEnv())
        env.reset(task=FakeTask())
        env.step({"tool": "submit"})

        traj = env.get_trajectory()
        assert traj["task"]["task_id"] == "task-001"
        assert traj["task"]["description"] == "Fix the bug"

    def test_reset_clears_recording(self):
        env = RecorderWrapper(MockEnv())
        env.reset()
        env.step({"tool": "bash"})
        env.reset()
        env.step({"tool": "edit"})

        traj = env.get_trajectory()
        assert len(traj["steps"]) == 1
        assert traj["steps"][0]["tool"] == "edit"

    def test_step_records_thought(self):
        env = RecorderWrapper(MockEnv())
        env.reset()
        env.step({"tool": "bash", "thought": "看看文件列表"})

        traj = env.get_trajectory()
        assert traj["steps"][0]["thought"] == "看看文件列表"

    def test_step_records_reward(self):
        """嵌套 RewardWrapper 时应记录 reward."""
        def constant_reward(steps, action):
            return 0.8

        inner = MockEnv()
        rewarded = RewardWrapper(inner, reward_fn=constant_reward)
        env = RecorderWrapper(rewarded)

        env.reset()
        env.step({"tool": "bash"})

        traj = env.get_trajectory()
        assert traj["steps"][0]["reward"] == 0.8


# ── 组合测试 ──────────────────────────────────────────────────────


class TestWrapperComposition:
    """Wrapper 组合测试."""

    def test_max_steps_plus_recorder(self):
        """MaxSteps + Recorder 组合."""
        inner = MockEnv()
        env = MaxStepsWrapper(inner, max_steps=3)
        env = RecorderWrapper(env)

        ts = env.reset()
        while not ts.done:
            ts = env.step({"tool": "bash"})

        traj = env.get_trajectory()
        assert len(traj["steps"]) == 3
        assert traj["outcome"]["truncated"] is True

    def test_full_stack(self):
        """MaxSteps + Reward + Recorder 三层组合."""
        def count_reward(steps, action):
            return 1.0 / len(steps)

        inner = MockEnv()
        env = MaxStepsWrapper(inner, max_steps=3)
        env = RewardWrapper(env, reward_fn=count_reward)
        env = RecorderWrapper(env)

        ts = env.reset()
        while not ts.done:
            ts = env.step({"tool": "bash"})

        traj = env.get_trajectory()
        assert len(traj["steps"]) == 3
        # Recorder 应捕获 RewardWrapper 注入的 reward
        assert traj["steps"][0]["reward"] == 1.0
        assert traj["steps"][1]["reward"] == 0.5

    def test_unwrapped_through_stack(self):
        """多层 Wrapper 的 unwrapped 应返回最内层."""
        inner = MockEnv()
        env = MaxStepsWrapper(inner, max_steps=10)
        env = RewardWrapper(env, reward_fn=lambda s, a: 0.0)
        env = RecorderWrapper(env)

        assert env.unwrapped is inner


# ── EpisodeStatisticsWrapper ──────────────────────────────────────


class TestEpisodeStatisticsWrapper:
    """EpisodeStatisticsWrapper 测试."""

    def test_tracks_reward(self):
        """应累计 episode reward."""
        def r_fn(steps, action):
            return 0.5

        inner = MockEnv()
        env = RewardWrapper(inner, reward_fn=r_fn)
        env = EpisodeStatisticsWrapper(env)
        env.reset()
        env.step({"tool": "bash"})
        ts = env.step({"tool": "submit"})

        assert "episode" in ts.info
        assert ts.info["episode"]["r"] == 1.0  # 0.5 * 2
        assert ts.info["episode"]["l"] == 2

    def test_tracks_length(self):
        """应记录 episode 步数."""
        env = EpisodeStatisticsWrapper(MaxStepsWrapper(MockEnv(), max_steps=3))
        ts = env.reset()
        while not ts.done:
            ts = env.step({"tool": "bash"})

        assert ts.info["episode"]["l"] == 3

    def test_tracks_time(self):
        """应记录 episode 耗时."""
        env = EpisodeStatisticsWrapper(MockEnv())
        env.reset()
        ts = env.step({"tool": "submit"})

        assert "t" in ts.info["episode"]
        assert ts.info["episode"]["t"] >= 0.0

    def test_no_stats_before_done(self):
        """episode 未结束时不应注入统计."""
        env = EpisodeStatisticsWrapper(MockEnv())
        env.reset()
        ts = env.step({"tool": "bash"})

        assert "episode" not in ts.info

    def test_reset_clears_stats(self):
        """reset 应清零统计."""
        def r_fn(steps, action):
            return 1.0

        inner = MockEnv()
        env = RewardWrapper(inner, reward_fn=r_fn)
        env = EpisodeStatisticsWrapper(env)
        env.reset()
        env.step({"tool": "bash"})
        env.step({"tool": "submit"})

        # 重置后统计应重新开始
        env.reset()
        assert env.episode_reward == 0.0
        assert env.episode_length == 0

    def test_properties(self):
        """episode_reward 和 episode_length 属性应实时更新."""
        def r_fn(steps, action):
            return 0.3

        inner = MockEnv()
        env = RewardWrapper(inner, reward_fn=r_fn)
        env = EpisodeStatisticsWrapper(env)
        env.reset()

        env.step({"tool": "bash"})
        assert env.episode_length == 1
        assert abs(env.episode_reward - 0.3) < 0.01

        env.step({"tool": "bash"})
        assert env.episode_length == 2
        assert abs(env.episode_reward - 0.6) < 0.01


# ── ObservationTruncateWrapper ────────────────────────────────────


class LongOutputEnv(AgentEnv):
    """产生长 observation 的测试环境."""

    domain = "test"

    def __init__(self, output_len: int = 10000):
        self._output_len = output_len

    def reset(self, *, task=None, seed=None) -> TimeStep:
        return TimeStep(observation="x" * self._output_len)

    def step(self, action: dict) -> TimeStep:
        return TimeStep(
            observation="y" * self._output_len,
            terminated=(action.get("tool") == "submit"),
        )


class TestObservationTruncateWrapper:
    """ObservationTruncateWrapper 测试."""

    def test_truncates_long_output(self):
        """超长 observation 应被截断."""
        env = ObservationTruncateWrapper(LongOutputEnv(10000), max_chars=100)
        env.reset()
        ts = env.step({"tool": "bash"})

        assert len(ts.observation) == 100 + len("...[truncated]")
        assert ts.observation.endswith("...[truncated]")
        assert ts.info.get("observation_truncated") is True

    def test_no_truncation_short_output(self):
        """短 observation 不应被截断."""
        env = ObservationTruncateWrapper(MockEnv(), max_chars=8000)
        env.reset()
        ts = env.step({"tool": "bash"})

        assert "observation_truncated" not in ts.info
        assert ts.observation == "bash-output"

    def test_truncates_reset_observation(self):
        """reset 的 observation 也应被截断."""
        env = ObservationTruncateWrapper(LongOutputEnv(5000), max_chars=200)
        ts = env.reset()

        assert len(ts.observation) == 200 + len("...[truncated]")

    def test_custom_suffix(self):
        """自定义 suffix 应生效."""
        env = ObservationTruncateWrapper(
            LongOutputEnv(1000), max_chars=50, suffix="[CUT]",
        )
        env.reset()
        ts = env.step({"tool": "bash"})

        assert ts.observation.endswith("[CUT]")

    def test_exact_boundary(self):
        """刚好等于 max_chars 时不应截断."""
        env = ObservationTruncateWrapper(LongOutputEnv(100), max_chars=100)
        env.reset()
        ts = env.step({"tool": "bash"})

        assert len(ts.observation) == 100
        assert "observation_truncated" not in ts.info

    def test_composition_with_recorder(self):
        """Truncate + Recorder 组合应生效."""
        inner = LongOutputEnv(5000)
        env = ObservationTruncateWrapper(inner, max_chars=200)
        env = RecorderWrapper(env)

        env.reset()
        env.step({"tool": "submit"})

        traj = env.get_trajectory()
        assert len(traj["steps"][0]["output"]) <= 200 + len("...[truncated]")
