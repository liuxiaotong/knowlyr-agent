"""测试 collect() — Gymnasium 风格轨迹收集."""

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from trajectoryhub.collect import collect


# ── MockEnv ─────────────────────────────────────────────────────────


class MockEnv(AgentEnv):
    """测试用环境."""

    domain = "test"

    def __init__(self):
        self._step_count = 0
        self._closed = False

    def reset(self, *, task=None, seed=None) -> TimeStep:
        self._step_count = 0
        return TimeStep(observation="ready", info={"task": task})

    def step(self, action: dict) -> TimeStep:
        self._step_count += 1
        tool = action.get("tool", "noop")
        return TimeStep(
            observation=f"{tool}-result",
            terminated=(tool == "submit"),
            info={"exit_code": 0, "step": self._step_count},
        )

    def close(self):
        self._closed = True

    @property
    def available_tools(self):
        return ["bash", "submit"]


# ── mock agent ──────────────────────────────────────────────────────


def simple_agent(observation: str) -> dict:
    """3 步 bash 后 submit 的简单 agent."""
    if "step-3" in observation or observation.count("bash") >= 3:
        return {"tool": "submit", "params": {}}
    return {"tool": "bash", "params": {"command": "ls"}}


class StatefulAgent:
    """有状态的 agent，第 n 步后 submit."""

    def __init__(self, submit_after: int = 3):
        self._step = 0
        self._submit_after = submit_after

    def __call__(self, observation: str) -> dict:
        self._step += 1
        if self._step >= self._submit_after:
            self._step = 0  # 重置，支持多 episode
            return {"tool": "submit", "params": {}}
        return {"tool": "bash", "params": {"command": f"step-{self._step}"}}


# ── collect 测试 ──────────────────────────────────────────────────


class TestCollect:
    """collect() 测试."""

    def test_single_episode(self):
        """单 episode 收集."""
        agent = StatefulAgent(submit_after=3)
        trajs = collect(MockEnv(), agent=agent, n_episodes=1, max_steps=10)

        assert len(trajs) == 1
        traj = trajs[0]
        assert len(traj["steps"]) == 3
        assert traj["outcome"]["success"] is True
        assert traj["outcome"]["terminated"] is True

    def test_multiple_episodes(self):
        """多 episode 收集."""
        agent = StatefulAgent(submit_after=2)
        trajs = collect(MockEnv(), agent=agent, n_episodes=3, max_steps=10)

        assert len(trajs) == 3
        for traj in trajs:
            assert len(traj["steps"]) == 2
            assert traj["outcome"]["success"] is True

    def test_max_steps_truncation(self):
        """超过 max_steps 应 truncate."""
        def never_submit(obs):
            return {"tool": "bash", "params": {}}

        trajs = collect(MockEnv(), agent=never_submit, n_episodes=1, max_steps=5)

        assert len(trajs) == 1
        traj = trajs[0]
        assert len(traj["steps"]) == 5
        assert traj["outcome"]["truncated"] is True

    def test_agent_name_model_name(self):
        """agent_name 和 model_name 应记录在轨迹中."""
        agent = StatefulAgent(submit_after=1)
        trajs = collect(
            MockEnv(),
            agent=agent,
            n_episodes=1,
            agent_name="test-agent",
            model_name="gpt-4o",
        )

        traj = trajs[0]
        assert traj["agent"] == "test-agent"
        assert traj["model"] == "gpt-4o"

    def test_task_passed_to_env(self):
        """task 应传给 env.reset()."""
        class FakeTask:
            task_id = "task-001"
            description = "Fix the bug"

        agent = StatefulAgent(submit_after=1)
        trajs = collect(
            MockEnv(),
            agent=agent,
            n_episodes=1,
            task=FakeTask(),
        )

        traj = trajs[0]
        assert traj["task"]["task_id"] == "task-001"

    def test_trajectory_has_steps_detail(self):
        """每步应有 tool/params/output 等详情."""
        agent = StatefulAgent(submit_after=2)
        trajs = collect(MockEnv(), agent=agent, n_episodes=1, max_steps=10)

        step = trajs[0]["steps"][0]
        assert "tool" in step
        assert "params" in step
        assert "output" in step
        assert "step_id" in step

    def test_domain_in_metadata(self):
        """metadata 应包含 domain."""
        agent = StatefulAgent(submit_after=1)
        trajs = collect(MockEnv(), agent=agent, n_episodes=1)

        assert trajs[0]["metadata"]["domain"] == "test"

    def test_env_string_id(self):
        """传入字符串 env_id 应尝试 make()."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock", MockEnv, domain="test")

        agent = StatefulAgent(submit_after=1)
        trajs = collect("test/mock", agent=agent, n_episodes=1)

        assert len(trajs) == 1
        _clear_registry()
