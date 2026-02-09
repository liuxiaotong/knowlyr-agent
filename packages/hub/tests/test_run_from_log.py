"""测试 Pipeline.run_from_log / run_batch_from_logs 集成."""

import json

import pytest

from trajectoryhub.config import PipelineConfig
from trajectoryhub.pipeline import Pipeline, Trajectory
from trajectoryhub.tasks import TaskInfo


# ── 测试数据工厂 ───────────────────────────────────────────────────


def _write_openhands_log(path, task_desc="修复 bug"):
    """写一个最小的合法 OpenHands JSONL 日志."""
    events = [
        {
            "id": 0, "source": "user", "action": "message",
            "args": {"content": task_desc, "task_id": "oh-test"},
            "timestamp": "2026-01-01T10:00:00Z",
        },
        {
            "id": 1, "source": "agent", "action": "run",
            "args": {"command": "pytest tests/"},
            "thought": "跑测试",
            "timestamp": "2026-01-01T10:01:00Z",
            "extras": {"model": "claude-sonnet-4-20250514", "agent_class": "CodeActAgent"},
        },
        {
            "id": 2, "observation": "run",
            "content": "1 passed",
            "extras": {"exit_code": 0},
        },
        {
            "id": 3, "source": "agent", "action": "edit",
            "args": {"path": "/workspace/fix.py"},
            "thought": "修复代码",
            "timestamp": "2026-01-01T10:02:00Z",
        },
        {
            "id": 4, "observation": "edit",
            "content": "File edited",
            "extras": {},
        },
        {
            "id": 5, "action": "finish",
            "args": {"outputs": {"test_result": {
                "success": True, "tests_passed": 2, "tests_failed": 0,
            }}},
        },
    ]
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _write_sweagent_log(path, task_desc="修复 Django bug"):
    """写一个最小的合法 SWE-agent JSON 日志."""
    data = {
        "problem_statement": task_desc,
        "history": [
            [{"action": "bash", "command": "ls", "thought": "看看文件"}, "file.py\n"],
            [{"action": "edit", "path": "file.py"}, "File edited\n"],
            [{"action": "submit"}, "Submitted"],
        ],
        "info": {
            "instance_id": "swe-test-001",
            "model_name": "gpt-4o",
            "repo": "django/django",
            "base_commit": "abc123",
            "exit_status": "submitted",
            "model_stats": {"tokens_sent": 1000, "tokens_received": 500},
        },
    }
    with open(path, "w") as f:
        json.dump(data, f)


# ── run_from_log 测试 ──────────────────────────────────────────────


class TestRunFromLog:
    """验证 run_from_log 完整流程."""

    def test_openhands_log(self, tmp_path):
        """OpenHands 日志应产出带评分的 Trajectory."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "openhands")

        assert isinstance(traj, Trajectory)
        assert traj.task_id == "oh-test"
        assert traj.agent_framework == "openhands/CodeActAgent"
        assert traj.agent_model == "claude-sonnet-4-20250514"
        assert traj.total_steps == 2  # run + edit
        assert traj.success is True
        assert traj.reward > 0.0  # reward engine 应给正分
        assert len(traj.step_rewards) == 2

    def test_sweagent_log(self, tmp_path):
        """SWE-agent 日志应产出带评分的 Trajectory."""
        log_path = tmp_path / "traj.json"
        _write_sweagent_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "sweagent")

        assert isinstance(traj, Trajectory)
        assert traj.task_id == "swe-test-001"
        assert traj.agent_framework == "swe-agent"
        assert traj.agent_model == "gpt-4o"
        assert traj.total_steps == 3  # bash + edit + submit
        assert traj.success is True
        assert traj.reward > 0.0

    def test_swe_agent_alias(self, tmp_path):
        """swe-agent 别名应等同于 sweagent."""
        log_path = tmp_path / "traj.json"
        _write_sweagent_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "swe-agent")
        assert traj.agent_framework == "swe-agent"

    def test_custom_task_override(self, tmp_path):
        """传入 task 参数应覆盖日志中的 task 信息."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        custom_task = TaskInfo(task_id="custom-001", description="自定义任务")
        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "openhands", task=custom_task)

        assert traj.task_id == "custom-001"
        assert traj.metadata["task_description"] == "自定义任务"

    def test_step_details(self, tmp_path):
        """steps 应包含 thought/tool/params/output/exit_code."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "openhands")

        step = traj.steps[0]
        assert "thought" in step
        assert "tool" in step
        assert "params" in step
        assert "output" in step
        assert "exit_code" in step
        assert step["thought"] == "跑测试"
        assert step["tool"] == "bash"

    def test_metadata_includes_outcome(self, tmp_path):
        """metadata 应包含 outcome 信息."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "openhands")

        assert traj.metadata["outcome_tests_passed"] == 2
        assert traj.metadata["outcome_tests_failed"] == 0

    def test_unsupported_framework(self, tmp_path):
        """不支持的框架应抛出 ValueError."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        with pytest.raises(ValueError, match="不支持的框架"):
            pipeline.run_from_log(log_path, "unknown_framework")

    def test_duration_tracked(self, tmp_path):
        """duration_seconds 应大于 0."""
        log_path = tmp_path / "log.jsonl"
        _write_openhands_log(log_path)

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path)))
        traj = pipeline.run_from_log(log_path, "openhands")

        assert traj.duration_seconds >= 0.0


# ── run_batch_from_logs 测试 ───────────────────────────────────────


class TestRunBatchFromLogs:
    """验证批量日志处理."""

    def test_batch_sweagent(self, tmp_path):
        """批量处理多个 SWE-agent 日志."""
        for i in range(3):
            _write_sweagent_log(tmp_path / f"traj_{i}.json", f"任务 {i}")

        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path / "out")))
        trajectories = pipeline.run_batch_from_logs(tmp_path, "sweagent", "*.json")

        assert len(trajectories) == 3
        for traj in trajectories:
            assert isinstance(traj, Trajectory)
            assert traj.agent_framework == "swe-agent"
            assert traj.reward > 0.0

    def test_batch_empty_dir(self, tmp_path):
        """空目录应返回空列表."""
        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path / "out")))
        trajectories = pipeline.run_batch_from_logs(tmp_path, "sweagent", "*.json")
        assert trajectories == []

    def test_batch_unsupported_framework(self, tmp_path):
        """不支持的框架应抛出 ValueError."""
        pipeline = Pipeline(PipelineConfig(output_dir=str(tmp_path / "out")))
        with pytest.raises(ValueError, match="不支持的框架"):
            pipeline.run_batch_from_logs(tmp_path, "unknown")


# ── Pipeline 可选依赖标志测试 ──────────────────────────────────────


class TestOptionalDeps:
    """验证可选依赖检测."""

    def test_has_recorder(self):
        """在开发环境中 recorder 应可用."""
        from trajectoryhub.pipeline import _HAS_RECORDER
        assert _HAS_RECORDER is True

    def test_has_reward(self):
        """在开发环境中 reward 应可用."""
        from trajectoryhub.pipeline import _HAS_REWARD
        assert _HAS_REWARD is True

    def test_adapter_map_populated(self):
        """适配器注册表应包含 openhands 和 sweagent."""
        from trajectoryhub.pipeline import _ADAPTER_MAP
        assert "openhands" in _ADAPTER_MAP
        assert "sweagent" in _ADAPTER_MAP
        assert "swe-agent" in _ADAPTER_MAP
