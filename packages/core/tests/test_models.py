"""Tests for core shared models."""

from knowlyrcore import TaskInfo, ToolResult


class TestToolResult:
    """ToolResult 模型测试."""

    def test_success(self):
        result = ToolResult(output="hello", exit_code=0)
        assert result.success is True
        assert result.output == "hello"
        assert result.exit_code == 0
        assert result.error is None

    def test_failure(self):
        result = ToolResult(output="", exit_code=1, error="not found")
        assert result.success is False
        assert result.error == "not found"

    def test_error_with_zero_exit_code(self):
        result = ToolResult(output="partial", exit_code=0, error="timeout")
        assert result.success is False

    def test_defaults(self):
        result = ToolResult()
        assert result.output == ""
        assert result.exit_code == 0
        assert result.error is None
        assert result.success is True

    def test_model_dump(self):
        result = ToolResult(output="ok", exit_code=0, error=None)
        data = result.model_dump()
        assert data == {"output": "ok", "exit_code": 0, "error": None}

    def test_model_validate(self):
        result = ToolResult.model_validate({"output": "x", "exit_code": 1, "error": "e"})
        assert result.output == "x"
        assert result.exit_code == 1
        assert result.success is False


class TestTaskInfo:
    """TaskInfo 模型测试."""

    def test_defaults(self):
        info = TaskInfo()
        assert info.task_id == ""
        assert info.language == ""
        assert info.metadata == {}

    def test_full(self):
        info = TaskInfo(
            task_id="test-001",
            description="修复排序 bug",
            type="bug_fix",
            language="python",
            difficulty="easy",
            repo="owner/repo",
            base_commit="abc123",
            test_command="pytest tests/",
            metadata={"source": "swebench"},
        )
        assert info.task_id == "test-001"
        assert info.repo == "owner/repo"
        assert info.metadata["source"] == "swebench"

    def test_model_dump(self):
        info = TaskInfo(task_id="t1", language="python")
        data = info.model_dump()
        assert data["task_id"] == "t1"
        assert data["language"] == "python"
        assert "metadata" in data

    def test_model_validate(self):
        info = TaskInfo.model_validate({
            "task_id": "t2",
            "repo": "o/r",
            "base_commit": "abc",
        })
        assert info.task_id == "t2"
        assert info.repo == "o/r"
        assert info.description == ""
