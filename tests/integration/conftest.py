"""集成测试共享 fixtures."""

import pytest

from knowlyrcore import TaskInfo, ToolResult

from agentrecorder.schema import (
    Outcome,
    Step,
    ToolCall,
    Trajectory as RecorderTrajectory,
)


@pytest.fixture
def task_info():
    """标准任务信息."""
    return TaskInfo(
        task_id="integration-001",
        description="修复排序函数的 off-by-one 错误",
        type="bug_fix",
        language="python",
        difficulty="easy",
        repo="owner/repo",
        base_commit="abc123def",
        test_command="pytest tests/test_sort.py",
        metadata={"source": "integration_test"},
    )


@pytest.fixture
def recorder_trajectory(task_info):
    """完整的 recorder 格式轨迹 — 模拟一个成功的 bug fix 过程."""
    return RecorderTrajectory(
        task=task_info,
        agent="openhands",
        model="claude-sonnet-4-20250514",
        steps=[
            Step(
                step_id=0,
                thought="先看看排序函数的实现",
                tool_call=ToolCall(
                    name="read_file",
                    parameters={"file_path": "/workspace/sort.py"},
                ),
                tool_result=ToolResult(
                    output="def sort(lst):\n    for i in range(len(lst)):\n"
                    "        for j in range(len(lst) - 1):\n"
                    "            if lst[j] > lst[j+1]:\n"
                    "                lst[j], lst[j+1] = lst[j+1], lst[j]\n"
                    "    return lst\n",
                    exit_code=0,
                ),
                timestamp="2026-02-09T10:00:00Z",
                token_count=100,
            ),
            Step(
                step_id=1,
                thought="运行测试确认 bug 存在",
                tool_call=ToolCall(
                    name="Bash",
                    parameters={"command": "cd /workspace && pytest tests/test_sort.py"},
                ),
                tool_result=ToolResult(
                    output="FAILED tests/test_sort.py::test_sort - AssertionError",
                    exit_code=1,
                    error="1 failed",
                ),
                timestamp="2026-02-09T10:01:00Z",
                token_count=80,
            ),
            Step(
                step_id=2,
                thought="range 应该用 len(lst) - 1 - i 来优化并修复 off-by-one",
                tool_call=ToolCall(
                    name="Edit",
                    parameters={
                        "file_path": "/workspace/sort.py",
                        "old_string": "for j in range(len(lst) - 1):",
                        "new_string": "for j in range(len(lst) - 1 - i):",
                    },
                ),
                tool_result=ToolResult(output="File edited successfully", exit_code=0),
                timestamp="2026-02-09T10:02:00Z",
                token_count=120,
            ),
            Step(
                step_id=3,
                thought="运行测试验证修复",
                tool_call=ToolCall(
                    name="Bash",
                    parameters={"command": "cd /workspace && pytest tests/test_sort.py"},
                ),
                tool_result=ToolResult(
                    output="1 passed in 0.02s",
                    exit_code=0,
                ),
                timestamp="2026-02-09T10:03:00Z",
                token_count=60,
            ),
        ],
        outcome=Outcome(
            success=True,
            tests_passed=1,
            tests_failed=0,
            total_steps=4,
            total_tokens=360,
        ),
        metadata={"run_id": "integration-run-001"},
    )


def recorder_steps_to_reward_steps(recorder_traj: RecorderTrajectory) -> list[dict]:
    """将 recorder Trajectory 的 steps 转换为 reward engine 期望的格式.

    Recorder Step 格式:
        tool_call.name, tool_call.parameters, tool_result.output

    Reward 期望格式:
        tool, params, output
    """
    return [
        {
            "tool": step.tool_call.name,
            "params": step.tool_call.parameters,
            "output": step.tool_result.output if step.tool_result else "",
        }
        for step in recorder_traj.steps
    ]


def recorder_outcome_to_reward_outcome(recorder_traj: RecorderTrajectory) -> dict:
    """将 recorder Outcome 转换为 reward engine 期望的格式."""
    outcome = recorder_traj.outcome
    return {
        "success": outcome.success,
        "tests_passed": outcome.tests_passed,
        "tests_total": outcome.tests_passed + outcome.tests_failed,
    }
