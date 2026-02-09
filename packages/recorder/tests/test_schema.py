"""测试 Schema 数据模型."""

import json
import tempfile
from pathlib import Path

import pytest

from agentrecorder.schema import (
    Trajectory,
    Step,
    ToolCall,
    ToolResult,
    TaskInfo,
    Outcome,
)


class TestToolCall:
    """ToolCall 模型测试."""

    def test_create(self):
        tc = ToolCall(name="bash", parameters={"command": "ls -la"})
        assert tc.name == "bash"
        assert tc.parameters == {"command": "ls -la"}

    def test_empty_parameters(self):
        tc = ToolCall(name="submit", parameters={})
        assert tc.name == "submit"
        assert tc.parameters == {}

    def test_default_parameters(self):
        tc = ToolCall(name="submit")
        assert tc.parameters == {}

    def test_serialization(self):
        tc = ToolCall(name="edit", parameters={"file": "main.py", "content": "print('hi')"})
        data = tc.model_dump()
        assert data["name"] == "edit"
        assert data["parameters"]["file"] == "main.py"


class TestToolResult:
    """ToolResult 模型测试."""

    def test_create_success(self):
        tr = ToolResult(output="file1.py\nfile2.py", exit_code=0)
        assert tr.output == "file1.py\nfile2.py"
        assert tr.exit_code == 0
        assert tr.error is None

    def test_create_error(self):
        tr = ToolResult(output="", exit_code=1, error="command not found")
        assert tr.exit_code == 1
        assert tr.error == "command not found"

    def test_default_exit_code(self):
        tr = ToolResult(output="ok")
        assert tr.exit_code == 0

    def test_serialization(self):
        tr = ToolResult(output="done", exit_code=0, error=None)
        data = tr.model_dump()
        assert data["output"] == "done"
        assert data["exit_code"] == 0
        assert data["error"] is None


class TestStep:
    """Step 模型测试."""

    def test_create(self):
        step = Step(
            step_id=1,
            thought="我需要查看文件列表",
            tool_call=ToolCall(name="bash", parameters={"command": "ls"}),
            tool_result=ToolResult(output="main.py\ntest.py", exit_code=0),
            timestamp="2026-01-15T10:30:00Z",
            token_count=150,
        )
        assert step.step_id == 1
        assert step.thought == "我需要查看文件列表"
        assert step.tool_call.name == "bash"
        assert step.tool_result.exit_code == 0
        assert step.token_count == 150

    def test_optional_token_count(self):
        step = Step(
            step_id=1,
            thought="思考",
            tool_call=ToolCall(name="bash", parameters={}),
            tool_result=ToolResult(output="ok"),
            timestamp="2026-01-15T10:30:00Z",
        )
        assert step.token_count is None

    def test_serialization(self):
        step = Step(
            step_id=1,
            thought="查看文件",
            tool_call=ToolCall(name="bash", parameters={"command": "ls"}),
            tool_result=ToolResult(output="file.py"),
            timestamp="2026-01-15T10:30:00Z",
        )
        data = step.model_dump()
        assert data["step_id"] == 1
        assert data["tool_call"]["name"] == "bash"


class TestTrajectory:
    """Trajectory 模型测试."""

    def _make_trajectory(self) -> Trajectory:
        """创建一个测试用的 Trajectory."""
        return Trajectory(
            task=TaskInfo(
                task_id="test-001",
                description="修复排序 bug",
                type="bug_fix",
                language="python",
                difficulty="easy",
                repo="owner/repo",
                base_commit="abc123",
                test_command="pytest tests/",
            ),
            agent="openhands",
            model="claude-sonnet-4-20250514",
            steps=[
                Step(
                    step_id=1,
                    thought="先看看代码",
                    tool_call=ToolCall(name="bash", parameters={"command": "cat sort.py"}),
                    tool_result=ToolResult(output="def sort(lst):\n    return lst"),
                    timestamp="2026-01-15T10:30:00Z",
                    token_count=100,
                ),
                Step(
                    step_id=2,
                    thought="修复排序实现",
                    tool_call=ToolCall(
                        name="edit",
                        parameters={"file": "sort.py", "content": "def sort(lst):\n    return sorted(lst)"},
                    ),
                    tool_result=ToolResult(output="File edited successfully"),
                    timestamp="2026-01-15T10:31:00Z",
                    token_count=120,
                ),
            ],
            outcome=Outcome(
                success=True,
                tests_passed=5,
                tests_failed=0,
                total_steps=2,
                total_tokens=220,
            ),
            metadata={"run_id": "run-abc", "duration": 42.5},
        )

    def test_create(self):
        traj = self._make_trajectory()
        assert traj.agent == "openhands"
        assert traj.model == "claude-sonnet-4-20250514"
        assert len(traj.steps) == 2
        assert traj.outcome.success is True
        assert traj.task.task_id == "test-001"

    def test_serialization(self):
        traj = self._make_trajectory()
        json_str = traj.model_dump_json()
        data = json.loads(json_str)
        assert data["agent"] == "openhands"
        assert len(data["steps"]) == 2
        assert data["outcome"]["success"] is True

    def test_to_jsonl_and_from_jsonl(self):
        traj = self._make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectories.jsonl"
            traj.to_jsonl(path)

            # 验证文件存在且非空
            assert path.exists()
            content = path.read_text()
            assert content.strip()

            # 从 JSONL 加载
            loaded = Trajectory.from_jsonl(path)
            assert len(loaded) == 1
            assert loaded[0].agent == "openhands"
            assert loaded[0].task.task_id == "test-001"
            assert len(loaded[0].steps) == 2

    def test_to_jsonl_append(self):
        traj = self._make_trajectory()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectories.jsonl"
            traj.to_jsonl(path)
            traj.to_jsonl(path)

            loaded = Trajectory.from_jsonl(path)
            assert len(loaded) == 2

    def test_schema_json_example(self):
        schema_str = Trajectory.schema_json_example()
        schema = json.loads(schema_str)
        assert "properties" in schema
        assert "agent" in schema["properties"]

    def test_empty_steps(self):
        traj = Trajectory(
            task=TaskInfo(task_id="empty-001"),
            agent="test-agent",
            steps=[],
            outcome=Outcome(success=False, total_steps=0),
        )
        assert len(traj.steps) == 0
        assert traj.outcome.success is False

    def test_metadata(self):
        traj = self._make_trajectory()
        assert traj.metadata["run_id"] == "run-abc"
        assert traj.metadata["duration"] == 42.5
