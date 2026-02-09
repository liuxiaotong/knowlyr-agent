"""集成测试 — 跨包数据模型互操作性."""

from knowlyrcore import TaskInfo, ToolResult

from agentrecorder.schema import (
    Outcome,
    Step,
    ToolCall,
    Trajectory as RecorderTrajectory,
)
from agentsandbox.tools import ToolResult as SandboxToolResult


class TestCoreModelInterop:
    """验证 core 模型在各包间一致性."""

    def test_tool_result_is_same_class(self):
        """sandbox 和 recorder 使用同一个 ToolResult."""
        assert SandboxToolResult is ToolResult

    def test_tool_result_in_recorder_step(self):
        """core ToolResult 可作为 recorder Step 的字段."""
        result = ToolResult(output="ok", exit_code=0)
        step = Step(
            step_id=0,
            thought="test",
            tool_call=ToolCall(name="bash", parameters={}),
            tool_result=result,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert step.tool_result.output == "ok"
        assert step.tool_result.success is True

    def test_tool_result_serialization_in_trajectory(self):
        """ToolResult 嵌套在 recorder Trajectory 中可正确序列化."""
        traj = RecorderTrajectory(
            task=TaskInfo(task_id="ser-test"),
            agent="test",
            steps=[
                Step(
                    step_id=0,
                    thought="t",
                    tool_call=ToolCall(name="ls", parameters={}),
                    tool_result=ToolResult(output="file.py\n", exit_code=0),
                    timestamp="2026-01-01T00:00:00Z",
                ),
            ],
            outcome=Outcome(success=True),
        )
        data = traj.model_dump()
        assert data["steps"][0]["tool_result"]["output"] == "file.py\n"
        assert data["steps"][0]["tool_result"]["exit_code"] == 0

    def test_task_info_in_recorder_trajectory(self):
        """core TaskInfo 可作为 recorder Trajectory.task 字段."""
        info = TaskInfo(
            task_id="ti-test",
            repo="user/repo",
            base_commit="abc",
            metadata={"key": "value"},
        )
        traj = RecorderTrajectory(
            task=info,
            agent="test",
            outcome=Outcome(success=False),
        )
        assert traj.task.task_id == "ti-test"
        assert traj.task.metadata["key"] == "value"

    def test_task_info_round_trip(self):
        """TaskInfo 序列化后再反序列化保持一致."""
        original = TaskInfo(
            task_id="rt-001",
            description="测试任务",
            type="bug_fix",
            language="python",
            repo="owner/repo",
            base_commit="deadbeef",
            metadata={"difficulty": "hard", "tags": ["sort", "algorithm"]},
        )
        data = original.model_dump()
        restored = TaskInfo.model_validate(data)
        assert restored.task_id == original.task_id
        assert restored.metadata == original.metadata
        assert restored.description == original.description
