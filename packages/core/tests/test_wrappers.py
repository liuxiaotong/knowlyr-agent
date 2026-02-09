"""测试 Wrapper 套件 — MaxSteps / Timeout / Reward / Recorder."""

import time

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep
from knowlyrcore.wrappers import (
    MaxStepsWrapper,
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
