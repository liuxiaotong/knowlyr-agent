"""测试 collect() — Gymnasium 风格轨迹收集."""

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from trajectoryhub.collect import collect, collect_parallel


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


# ── reward_fn 测试 ─────────────────────────────────────────────


class TestCollectWithReward:
    """collect() 带 reward_fn 参数的测试."""

    def test_no_reward_fn_backward_compat(self):
        """不传 reward_fn 时行为不变，reward 为 0.0."""
        agent = StatefulAgent(submit_after=2)
        trajs = collect(MockEnv(), agent=agent, n_episodes=1, max_steps=10)

        for step in trajs[0]["steps"]:
            assert step["reward"] == 0.0

    def test_reward_fn_injected(self):
        """传入 reward_fn 时每步 reward 非零."""
        def constant_reward(steps, action):
            return 0.42

        agent = StatefulAgent(submit_after=3)
        trajs = collect(
            MockEnv(), agent=agent, n_episodes=1, max_steps=10,
            reward_fn=constant_reward,
        )

        traj = trajs[0]
        assert len(traj["steps"]) == 3
        for step in traj["steps"]:
            assert step["reward"] == 0.42

    def test_reward_fn_step_dependent(self):
        """reward_fn 可以依赖步骤历史."""
        def progressive_reward(steps, action):
            return len(steps) * 0.1

        agent = StatefulAgent(submit_after=3)
        trajs = collect(
            MockEnv(), agent=agent, n_episodes=1, max_steps=10,
            reward_fn=progressive_reward,
        )

        rewards = [s["reward"] for s in trajs[0]["steps"]]
        assert abs(rewards[0] - 0.1) < 1e-6
        assert abs(rewards[1] - 0.2) < 1e-6
        assert abs(rewards[2] - 0.3) < 1e-6

    def test_reward_fn_multiple_episodes(self):
        """多 episode 时 reward 应正确重置."""
        def step_count_reward(steps, action):
            return float(len(steps))

        agent = StatefulAgent(submit_after=2)
        trajs = collect(
            MockEnv(), agent=agent, n_episodes=2, max_steps=10,
            reward_fn=step_count_reward,
        )

        # 每个 episode 的 reward 应从 1 开始
        for traj in trajs:
            rewards = [s["reward"] for s in traj["steps"]]
            assert abs(rewards[0] - 1.0) < 1e-6
            assert abs(rewards[1] - 2.0) < 1e-6


# ── collect_parallel 测试 ─────────────────────────────────────────


class TestCollectParallel:
    """collect_parallel() 并行收集测试."""

    def test_basic_parallel(self):
        """基本并行收集: 多 worker 收集总数正确."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-parallel", MockEnv, domain="test")

        def agent_factory():
            return StatefulAgent(submit_after=2)

        trajs = collect_parallel(
            "test/mock-parallel",
            agent_factory=agent_factory,
            n_episodes=6,
            max_steps=10,
            n_workers=2,
        )

        # 6 episodes 应收集 6 条轨迹
        assert len(trajs) == 6
        for traj in trajs:
            assert len(traj["steps"]) == 2
            assert traj["outcome"]["success"] is True

        _clear_registry()

    def test_parallel_single_worker(self):
        """单 worker 退化为串行."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-single", MockEnv, domain="test")

        def agent_factory():
            return StatefulAgent(submit_after=1)

        trajs = collect_parallel(
            "test/mock-single",
            agent_factory=agent_factory,
            n_episodes=3,
            max_steps=10,
            n_workers=1,
        )

        assert len(trajs) == 3
        _clear_registry()

    def test_parallel_more_workers_than_episodes(self):
        """worker 数 > episode 数时不崩溃."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-excess", MockEnv, domain="test")

        def agent_factory():
            return StatefulAgent(submit_after=1)

        trajs = collect_parallel(
            "test/mock-excess",
            agent_factory=agent_factory,
            n_episodes=2,
            max_steps=10,
            n_workers=5,
        )

        assert len(trajs) == 2
        _clear_registry()

    def test_parallel_with_reward_fn(self):
        """并行收集支持 reward_fn."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-reward", MockEnv, domain="test")

        def agent_factory():
            return StatefulAgent(submit_after=2)

        def constant_reward(steps, action):
            return 0.5

        trajs = collect_parallel(
            "test/mock-reward",
            agent_factory=agent_factory,
            n_episodes=4,
            max_steps=10,
            n_workers=2,
            reward_fn=constant_reward,
        )

        assert len(trajs) == 4
        for traj in trajs:
            for step in traj["steps"]:
                assert step["reward"] == 0.5

        _clear_registry()

    def test_parallel_worker_error_graceful(self):
        """worker 异常时应跳过失败、返回其他 worker 的结果."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-error", MockEnv, domain="test")

        call_count = [0]

        def flaky_agent_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一个 worker 的 agent 抛异常
                def bad_agent(obs):
                    raise RuntimeError("模拟 worker 崩溃")
                return bad_agent
            return StatefulAgent(submit_after=1)

        trajs = collect_parallel(
            "test/mock-error",
            agent_factory=flaky_agent_factory,
            n_episodes=4,
            max_steps=10,
            n_workers=2,
        )

        # 至少有一部分成功的轨迹（第二个 worker 的）
        assert len(trajs) >= 1
        for traj in trajs:
            assert traj["outcome"]["success"] is True

        _clear_registry()

    def test_parallel_all_workers_fail(self):
        """所有 worker 都失败时应返回空列表."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-allfail", MockEnv, domain="test")

        def failing_agent_factory():
            def bad_agent(obs):
                raise RuntimeError("always fail")
            return bad_agent

        trajs = collect_parallel(
            "test/mock-allfail",
            agent_factory=failing_agent_factory,
            n_episodes=4,
            max_steps=10,
            n_workers=2,
        )

        assert len(trajs) == 0
        _clear_registry()

    def test_parallel_agent_name_has_worker_id(self):
        """每个 worker 的 agent_name 应包含 worker ID."""
        from knowlyrcore.registry import register, _clear_registry

        _clear_registry()
        register("test/mock-names", MockEnv, domain="test")

        def agent_factory():
            return StatefulAgent(submit_after=1)

        trajs = collect_parallel(
            "test/mock-names",
            agent_factory=agent_factory,
            n_episodes=4,
            max_steps=10,
            n_workers=2,
            agent_name="myagent",
        )

        assert len(trajs) == 4
        # agent 名应包含 worker 编号
        agents = set(traj["agent"] for traj in trajs)
        assert all("myagent-w" in a for a in agents)

        _clear_registry()
