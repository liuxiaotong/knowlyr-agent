"""测试 AgentEnv 协议 + EnvWrapper."""

from knowlyrcore.env import AgentEnv, EnvWrapper
from knowlyrcore.timestep import TimeStep


# ── MockEnv ─────────────────────────────────────────────────────────


class MockEnv(AgentEnv):
    """测试用环境."""

    domain = "test"

    def __init__(self, max_steps: int = 5):
        self._max_steps = max_steps
        self._step_count = 0
        self._closed = False

    def reset(self, *, task=None, seed=None) -> TimeStep:
        self._step_count = 0
        return TimeStep(
            observation="环境就绪",
            info={"task": task, "seed": seed},
        )

    def step(self, action: dict) -> TimeStep:
        self._step_count += 1
        tool = action.get("tool", "unknown")
        terminated = tool == "submit"
        truncated = self._step_count >= self._max_steps

        return TimeStep(
            observation=f"执行 {tool}: ok",
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info={"step": self._step_count, "tool": tool},
        )

    def close(self) -> None:
        self._closed = True

    @property
    def available_tools(self) -> list[str]:
        return ["bash", "read", "submit"]


# ── TimeStep 测试 ──────────────────────────────────────────────────


class TestTimeStep:
    """TimeStep 数据结构测试."""

    def test_defaults(self):
        ts = TimeStep()
        assert ts.observation == ""
        assert ts.reward == 0.0
        assert ts.terminated is False
        assert ts.truncated is False
        assert ts.info == {}
        assert ts.done is False

    def test_done_on_terminated(self):
        ts = TimeStep(terminated=True)
        assert ts.done is True

    def test_done_on_truncated(self):
        ts = TimeStep(truncated=True)
        assert ts.done is True

    def test_done_on_both(self):
        ts = TimeStep(terminated=True, truncated=True)
        assert ts.done is True

    def test_not_done(self):
        ts = TimeStep(observation="hello", reward=0.5)
        assert ts.done is False

    def test_info_dict(self):
        ts = TimeStep(info={"exit_code": 0, "tokens": 100})
        assert ts.info["exit_code"] == 0
        assert ts.info["tokens"] == 100


# ── AgentEnv 测试 ──────────────────────────────────────────────────


class TestAgentEnv:
    """AgentEnv 协议测试."""

    def test_reset(self):
        env = MockEnv()
        ts = env.reset()
        assert ts.observation == "环境就绪"
        assert ts.done is False

    def test_reset_with_task(self):
        env = MockEnv()
        ts = env.reset(task="my_task", seed=42)
        assert ts.info["task"] == "my_task"
        assert ts.info["seed"] == 42

    def test_step(self):
        env = MockEnv()
        env.reset()
        ts = env.step({"tool": "bash", "params": {"command": "ls"}})
        assert "bash" in ts.observation
        assert ts.terminated is False
        assert ts.info["step"] == 1

    def test_step_terminated(self):
        env = MockEnv()
        env.reset()
        ts = env.step({"tool": "submit"})
        assert ts.terminated is True

    def test_step_truncated(self):
        env = MockEnv(max_steps=2)
        env.reset()
        env.step({"tool": "bash"})
        ts = env.step({"tool": "bash"})
        assert ts.truncated is True

    def test_close(self):
        env = MockEnv()
        env.close()
        assert env._closed is True

    def test_context_manager(self):
        with MockEnv() as env:
            env.reset()
            env.step({"tool": "bash"})
        assert env._closed is True

    def test_available_tools(self):
        env = MockEnv()
        assert "bash" in env.available_tools
        assert "submit" in env.available_tools

    def test_unwrapped(self):
        env = MockEnv()
        assert env.unwrapped is env

    def test_domain(self):
        env = MockEnv()
        assert env.domain == "test"

    def test_repr(self):
        env = MockEnv()
        assert "MockEnv" in repr(env)
        assert "test" in repr(env)

    def test_full_episode(self):
        """完整 episode 循环."""
        env = MockEnv(max_steps=10)
        ts = env.reset()
        steps = 0
        while not ts.done:
            ts = env.step({"tool": "bash"})
            steps += 1
        assert steps == 10
        assert ts.truncated is True


# ── EnvWrapper 测试 ────────────────────────────────────────────────


class TestEnvWrapper:
    """EnvWrapper 透传测试."""

    def test_passthrough_reset(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        ts = wrapper.reset()
        assert ts.observation == "环境就绪"

    def test_passthrough_step(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        wrapper.reset()
        ts = wrapper.step({"tool": "bash"})
        assert "bash" in ts.observation

    def test_passthrough_close(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        wrapper.close()
        assert env._closed is True

    def test_passthrough_available_tools(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        assert wrapper.available_tools == ["bash", "read", "submit"]

    def test_unwrapped_returns_inner(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        assert wrapper.unwrapped is env

    def test_domain_passthrough(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        assert wrapper.domain == "test"

    def test_nested_wrappers(self):
        """多层 Wrapper 嵌套."""
        env = MockEnv()
        w1 = EnvWrapper(env)
        w2 = EnvWrapper(w1)
        w3 = EnvWrapper(w2)

        assert w3.unwrapped is env
        ts = w3.reset()
        assert ts.observation == "环境就绪"

    def test_repr(self):
        env = MockEnv()
        wrapper = EnvWrapper(env)
        assert "EnvWrapper" in repr(wrapper)
        assert "MockEnv" in repr(wrapper)

    def test_custom_wrapper(self):
        """自定义 Wrapper 覆盖 step."""
        class DoubleRewardWrapper(EnvWrapper):
            def step(self, action):
                ts = self.env.step(action)
                ts.reward = 2.0
                return ts

        env = MockEnv()
        wrapped = DoubleRewardWrapper(env)
        wrapped.reset()
        ts = wrapped.step({"tool": "bash"})
        assert ts.reward == 2.0
