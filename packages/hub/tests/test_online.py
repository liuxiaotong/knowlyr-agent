"""测试 online_training_loop — 在线训练循环."""

import json
from dataclasses import asdict
from typing import Any, Callable

import pytest
from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from trajectoryhub.online import IterationStats, online_training_loop, _compute_stats

try:
    import agenttrainer  # noqa: F401
    _HAS_TRAINER = True
except ImportError:
    _HAS_TRAINER = False


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

    @property
    def available_tools(self):
        return ["bash", "submit"]


def _mock_agent_factory(iteration: int, checkpoint: str) -> Callable[[str], dict[str, Any]]:
    """Mock agent 工厂 — 2 步 bash 后 submit."""
    call_count = 0

    def agent(obs: str) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            call_count = 0
            return {"tool": "submit", "params": {}}
        return {"tool": "bash", "params": {"command": "ls"}}

    return agent


def _mock_reward_fn(steps: list[dict], action: dict) -> float:
    """Mock reward."""
    return 0.5 if action.get("tool") == "submit" else 0.1


# ── IterationStats 测试 ───────────────────────────────────────────


class TestIterationStats:
    """IterationStats dataclass 测试."""

    def test_defaults(self):
        """默认值应为零."""
        stats = IterationStats()
        assert stats.iteration == 0
        assert stats.avg_reward == 0.0
        assert stats.n_episodes == 0

    def test_asdict(self):
        """应能序列化为 dict."""
        stats = IterationStats(iteration=1, avg_reward=0.75, n_episodes=10)
        d = asdict(stats)
        assert d["iteration"] == 1
        assert d["avg_reward"] == 0.75

    def test_json_serializable(self):
        """应能序列化为 JSON."""
        stats = IterationStats(iteration=0, success_rate=0.5)
        result = json.dumps(asdict(stats))
        assert '"success_rate": 0.5' in result


# ── _compute_stats 测试 ──────────────────────────────────────────


class TestComputeStats:
    """_compute_stats() 内部函数测试."""

    def test_empty_trajectories(self):
        """空轨迹列表应返回零值."""
        stats = _compute_stats(0, [])
        assert stats.n_episodes == 0
        assert stats.avg_reward == 0.0
        assert stats.min_reward == 0.0

    def test_single_trajectory(self):
        """单条轨迹统计应正确."""
        trajs = [{
            "steps": [{"reward": 0.1}, {"reward": 0.5}],
            "outcome": {"success": True},
        }]
        stats = _compute_stats(0, trajs)
        assert stats.n_episodes == 1
        assert stats.success_rate == 1.0
        assert abs(stats.avg_reward - 0.6) < 1e-6
        assert stats.avg_steps == 2.0

    def test_mixed_success(self):
        """多条轨迹成功率计算."""
        trajs = [
            {"steps": [{"reward": 0.5}], "outcome": {"success": True}},
            {"steps": [{"reward": 0.1}], "outcome": {"success": False}},
            {"steps": [{"reward": 0.3}], "outcome": {"success": True}},
        ]
        stats = _compute_stats(2, trajs)
        assert stats.iteration == 2
        assert stats.n_episodes == 3
        assert abs(stats.success_rate - 2 / 3) < 1e-6
        assert abs(stats.min_reward - 0.1) < 1e-6
        assert abs(stats.max_reward - 0.5) < 1e-6


# ── online_training_loop 测试 ────────────────────────────────────


class TestOnlineTrainingLoop:
    """online_training_loop() 集成测试."""

    def test_basic_loop(self, tmp_path):
        """基本循环应跑通."""
        results = online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=2,
            n_episodes=3,
            max_steps=10,
            output_dir=str(tmp_path / "loop"),
            skip_training=True,
        )

        assert len(results) == 2
        for stats in results:
            assert isinstance(stats, IterationStats)
            assert stats.n_episodes == 3

    def test_output_files_created(self, tmp_path):
        """应生成 summary.json 和每轮 stats.json."""
        out = tmp_path / "output"
        online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=2,
            n_episodes=2,
            max_steps=10,
            output_dir=str(out),
            skip_training=True,
        )

        # 汇总文件
        assert (out / "summary.json").exists()
        summary = json.loads((out / "summary.json").read_text())
        assert len(summary) == 2

        # 每轮文件
        for i in range(2):
            iter_dir = out / f"iter-{i}"
            assert (iter_dir / "stats.json").exists()
            assert (iter_dir / "trajectories.jsonl").exists()

    def test_early_stopping(self, tmp_path):
        """patience 触发时应提前停止."""
        # 用一个 reward 递减的 agent factory
        def declining_agent_factory(iteration: int, checkpoint: str):
            # 每次迭代步骤变多 → reward 不会改善
            steps_before_submit = 2 + iteration
            call_count = 0

            def agent(obs: str) -> dict:
                nonlocal call_count
                call_count += 1
                if call_count >= steps_before_submit:
                    call_count = 0
                    return {"tool": "submit", "params": {}}
                return {"tool": "bash", "params": {}}

            return agent

        def declining_reward(steps, action):
            """步骤越多 reward 越低."""
            if action.get("tool") == "submit":
                return max(1.0 - len(steps) * 0.2, 0.0)
            return 0.0

        results = online_training_loop(
            agent_factory=declining_agent_factory,
            env_factory=_MockEnv,
            reward_fn=declining_reward,
            n_iterations=10,
            n_episodes=2,
            max_steps=20,
            output_dir=str(tmp_path / "early"),
            patience=2,
            skip_training=True,
        )

        # 应少于 10 轮
        assert len(results) < 10

    def test_skip_training_flag(self, tmp_path):
        """skip_training=True 应跳过训练但仍收集和导出."""
        results = online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=1,
            n_episodes=2,
            max_steps=10,
            output_dir=str(tmp_path / "skip"),
            skip_training=True,
        )

        assert len(results) == 1
        # 轨迹文件应存在
        traj_file = tmp_path / "skip" / "iter-0" / "trajectories.jsonl"
        assert traj_file.exists()

    def test_checkpoint_path_progression(self, tmp_path):
        """agent_factory 应收到正确的 checkpoint 路径."""
        received_checkpoints = []

        def tracking_factory(iteration: int, checkpoint: str):
            received_checkpoints.append(checkpoint)
            return _mock_agent_factory(iteration, checkpoint)

        online_training_loop(
            agent_factory=tracking_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            model_path="initial-model",
            n_iterations=3,
            n_episodes=1,
            max_steps=10,
            output_dir=str(tmp_path / "ckpt"),
            skip_training=True,
        )

        # iter 0 → initial-model, iter 1 → iter-0/final, iter 2 → iter-1/final
        assert received_checkpoints[0] == "initial-model"
        assert "iter-0" in received_checkpoints[1] and "final" in received_checkpoints[1]
        assert "iter-1" in received_checkpoints[2] and "final" in received_checkpoints[2]

    def test_single_iteration(self, tmp_path):
        """单次迭代应正常工作."""
        results = online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=1,
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path / "single"),
            skip_training=True,
        )

        assert len(results) == 1
        assert results[0].iteration == 0
        assert results[0].n_episodes == 1

    def test_eval_fn_callback(self, tmp_path):
        """eval_fn 回调应被调用且结果写入 stats."""
        eval_calls = []

        def my_eval(iteration: int, agent_fn, checkpoint: str) -> dict:
            eval_calls.append(iteration)
            return {"success_rate": 0.8, "avg_reward": 0.65, "n_episodes": 5}

        results = online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=2,
            n_episodes=2,
            max_steps=10,
            output_dir=str(tmp_path / "eval_fn"),
            skip_training=True,
            eval_fn=my_eval,
        )

        # eval_fn 被调用 2 次
        assert eval_calls == [0, 1]
        # 评估结果写入 stats
        for stats in results:
            assert stats.eval_success_rate == 0.8
            assert stats.eval_avg_reward == 0.65
            assert stats.eval_n_episodes == 5

    def test_eval_fn_receives_correct_agent(self, tmp_path):
        """eval_fn 应收到正确的 agent 和 checkpoint."""
        eval_checkpoints = []

        def tracking_eval(iteration: int, agent_fn, checkpoint: str) -> dict:
            eval_checkpoints.append(checkpoint)
            # 验证 agent_fn 可调用
            result = agent_fn("test")
            assert "tool" in result
            return {"success_rate": 0.5, "avg_reward": 0.3}

        online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            model_path="my-model",
            n_iterations=2,
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path / "eval_ckpt"),
            skip_training=True,
            eval_fn=tracking_eval,
        )

        assert eval_checkpoints[0] == "my-model"
        assert "iter-0" in eval_checkpoints[1]

    @pytest.mark.skipif(not _HAS_TRAINER, reason="knowlyr-trainer 未安装")
    def test_eval_episodes_builtin(self, tmp_path):
        """eval_episodes > 0 应使用内置 evaluate_agent."""
        results = online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=1,
            n_episodes=2,
            max_steps=10,
            output_dir=str(tmp_path / "eval_builtin"),
            skip_training=True,
            eval_episodes=3,
        )

        # 内置评估结果应写入 stats
        assert len(results) == 1
        assert results[0].eval_n_episodes == 3
        assert results[0].eval_success_rate is not None
        assert results[0].eval_avg_reward is not None

    def test_no_eval_by_default(self, tmp_path):
        """默认不评估 (eval_episodes=0, eval_fn=None)."""
        results = online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=1,
            n_episodes=1,
            max_steps=5,
            output_dir=str(tmp_path / "no_eval"),
            skip_training=True,
        )

        assert results[0].eval_success_rate is None
        assert results[0].eval_avg_reward is None

    def test_eval_results_in_summary_json(self, tmp_path):
        """评估结果应写入 summary.json."""
        out = tmp_path / "eval_summary"

        def simple_eval(iteration, agent_fn, checkpoint):
            return {"success_rate": 0.9, "avg_reward": 0.7}

        online_training_loop(
            agent_factory=_mock_agent_factory,
            env_factory=_MockEnv,
            reward_fn=_mock_reward_fn,
            n_iterations=1,
            n_episodes=1,
            max_steps=5,
            output_dir=str(out),
            skip_training=True,
            eval_fn=simple_eval,
        )

        summary = json.loads((out / "summary.json").read_text())
        assert summary[0]["eval_success_rate"] == 0.9
        assert summary[0]["eval_avg_reward"] == 0.7
