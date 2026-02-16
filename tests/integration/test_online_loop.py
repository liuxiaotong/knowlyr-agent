"""集成测试 — online_training_loop + evaluate + collect 跨包闭环."""

from typing import Any

from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from trajectoryhub.collect import collect
from trajectoryhub.evaluate import evaluate_agent
from trajectoryhub.online import IterationStats, online_training_loop


# ── 共享 Mock 组件 ───────────────────────────────────────────────


class IntegrationEnv(AgentEnv):
    """集成测试用环境 — 支持 bash + submit."""

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
            observation=f"{tool}-result-{self._step_count}",
            terminated=(tool == "submit"),
            info={"success": True} if tool == "submit" else {},
        )

    def close(self):
        pass

    @property
    def available_tools(self):
        return ["bash", "submit"]


def _make_agent(submit_after: int = 2):
    """创建 N 步后 submit 的 agent."""
    call_count = 0

    def agent(obs: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count >= submit_after:
            call_count = 0
            return {"tool": "submit", "params": {}}
        return {"tool": "bash", "params": {"command": "ls"}}

    return agent


# ── collect + evaluate 闭环 ───────────────────────────────────


class TestCollectEvaluateIntegration:
    """collect() 和 evaluate_agent() 的跨包集成测试."""

    def test_collect_then_evaluate(self):
        """collect 收集轨迹后，evaluate 可用同一环境评估."""
        # Step 1: 收集
        agent = _make_agent(submit_after=2)
        trajs = collect(
            IntegrationEnv(),
            agent=agent,
            n_episodes=3,
            max_steps=10,
        )

        assert len(trajs) == 3
        for traj in trajs:
            assert traj["outcome"]["success"] is True

        # Step 2: 评估（用相同类型的 agent 和 env）
        result = evaluate_agent(
            agent_fn=_make_agent(submit_after=2),
            env=IntegrationEnv(),
            n_episodes=3,
            max_steps=10,
        )

        assert result["success_rate"] == 1.0
        assert result["n_episodes"] == 3
        assert result["avg_steps"] == 2.0

    def test_evaluate_with_reward_fn(self):
        """evaluate_agent 支持 reward_fn 注入."""
        def simple_reward(steps, action):
            return 0.5 if action.get("tool") == "submit" else 0.1

        result = evaluate_agent(
            agent_fn=_make_agent(submit_after=3),
            env=IntegrationEnv(),
            n_episodes=2,
            max_steps=10,
            reward_fn=simple_reward,
        )

        assert result["success_rate"] == 1.0
        assert result["avg_reward"] > 0


# ── online_training_loop 端到端 ──────────────────────────────


class TestOnlineLoopIntegration:
    """online_training_loop 跨 hub + core 集成测试."""

    def test_full_loop_with_eval(self, tmp_path):
        """完整闭环: collect → reward → (skip train) → evaluate."""
        def agent_factory(iteration: int, checkpoint: str):
            return _make_agent(submit_after=2)

        def reward_fn(steps, action):
            return 0.5 if action.get("tool") == "submit" else 0.1

        def eval_fn(iteration, agent_fn, checkpoint):
            """评估回调 — 直接用 evaluate_agent."""
            return evaluate_agent(
                agent_fn=agent_fn,
                env=IntegrationEnv(),
                n_episodes=2,
                max_steps=10,
                reward_fn=reward_fn,
            )

        results = online_training_loop(
            agent_factory=agent_factory,
            env_factory=IntegrationEnv,
            reward_fn=reward_fn,
            n_iterations=2,
            n_episodes=3,
            max_steps=10,
            output_dir=str(tmp_path / "full_loop"),
            skip_training=True,
            eval_fn=eval_fn,
        )

        # 验证结果完整性
        assert len(results) == 2
        for stats in results:
            assert isinstance(stats, IterationStats)
            # 收集指标
            assert stats.n_episodes == 3
            assert stats.success_rate == 1.0
            assert stats.avg_reward > 0
            # 评估指标
            assert stats.eval_success_rate == 1.0
            assert stats.eval_avg_reward > 0

        # 验证输出文件
        assert (tmp_path / "full_loop" / "summary.json").exists()
        assert (tmp_path / "full_loop" / "iter-0" / "trajectories.jsonl").exists()
        assert (tmp_path / "full_loop" / "iter-0" / "stats.json").exists()

    def test_loop_with_early_stopping(self, tmp_path):
        """在线训练 + 早停的集成场景."""
        def declining_agent_factory(iteration, checkpoint):
            steps_needed = 2 + iteration
            call_count = 0

            def agent(obs):
                nonlocal call_count
                call_count += 1
                if call_count >= steps_needed:
                    call_count = 0
                    return {"tool": "submit", "params": {}}
                return {"tool": "bash", "params": {}}

            return agent

        def declining_reward(steps, action):
            if action.get("tool") == "submit":
                return max(1.0 - len(steps) * 0.15, 0.0)
            return 0.0

        results = online_training_loop(
            agent_factory=declining_agent_factory,
            env_factory=IntegrationEnv,
            reward_fn=declining_reward,
            n_iterations=10,
            n_episodes=2,
            max_steps=20,
            output_dir=str(tmp_path / "early_stop"),
            patience=2,
            skip_training=True,
        )

        # 应因早停而少于 10 轮
        assert len(results) < 10
        assert len(results) >= 3  # 至少跑了 patience + 1 轮
