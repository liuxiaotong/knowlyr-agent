"""集成测试 — Recorder → Reward 数据流."""

from agentrecorder.schema import (
    Outcome,
    Step,
    ToolCall,
    Trajectory as RecorderTrajectory,
)
from agentreward.reward import RewardEngine
from knowlyrcore import ToolResult

from .conftest import recorder_outcome_to_reward_outcome, recorder_steps_to_reward_steps


class TestRecorderToReward:
    """验证 recorder 输出可被 reward engine 正确评分."""

    def test_score_successful_trajectory(self, recorder_trajectory):
        """成功的 4 步轨迹应获得正向 reward."""
        steps = recorder_steps_to_reward_steps(recorder_trajectory)
        outcome = recorder_outcome_to_reward_outcome(recorder_trajectory)

        engine = RewardEngine()
        result = engine.score({
            "task": recorder_trajectory.task.description,
            "steps": steps,
            "outcome": outcome,
            "reference_steps": 4,
        })

        assert result.total_score > 0.0
        assert result.outcome_score == 1.0  # tests_passed=1, tests_total=1
        assert len(result.step_rewards) == 4
        # 每步应有 rubric 分数
        for sr in result.step_rewards:
            assert sr.total_score >= 0.0
            assert sr.total_score <= 1.0

    def test_score_failed_trajectory(self, task_info):
        """失败的轨迹应有较低 outcome_score."""
        traj = RecorderTrajectory(
            task=task_info,
            agent="test",
            model="test",
            steps=[
                Step(
                    step_id=0,
                    thought="随便改",
                    tool_call=ToolCall(name="Edit", parameters={
                        "file_path": "/workspace/sort.py",
                        "old_string": "x",
                        "new_string": "y",
                    }),
                    tool_result=ToolResult(output="edited", exit_code=0),
                    timestamp="2026-01-01T00:00:00Z",
                ),
            ],
            outcome=Outcome(success=False, tests_passed=0, tests_failed=1),
        )

        steps = recorder_steps_to_reward_steps(traj)
        outcome = recorder_outcome_to_reward_outcome(traj)

        engine = RewardEngine()
        result = engine.score({
            "task": traj.task.description,
            "steps": steps,
            "outcome": outcome,
        })

        assert result.outcome_score == 0.0
        assert result.total_score < 0.5

    def test_redundancy_detected(self, task_info):
        """重复读取同一文件应被 reward 检测到."""
        # 读同一个文件两次（无中间编辑）
        def read_step(sid):
            return Step(
                step_id=sid,
                thought="再看看",
                tool_call=ToolCall(
                    name="read_file",
                    parameters={"file_path": "/workspace/sort.py"},
                ),
                tool_result=ToolResult(output="code...", exit_code=0),
                timestamp="2026-01-01T00:00:00Z",
            )

        traj = RecorderTrajectory(
            task=task_info,
            agent="test",
            steps=[read_step(0), read_step(1)],
            outcome=Outcome(success=True),
        )

        steps = recorder_steps_to_reward_steps(traj)
        outcome = recorder_outcome_to_reward_outcome(traj)

        engine = RewardEngine()
        result = engine.score({
            "task": traj.task.description,
            "steps": steps,
            "outcome": outcome,
        })

        # 第二次读取应被标记为冗余，拉低 process_score
        assert len(result.step_rewards) == 2
        # step 0 正常，step 1 应被扣分
        assert result.step_rewards[0].total_score >= result.step_rewards[1].total_score

    def test_empty_trajectory(self, task_info):
        """空轨迹应返回零分."""
        engine = RewardEngine()
        result = engine.score({
            "task": task_info.description,
            "steps": [],
            "outcome": {"success": False},
        })

        assert result.total_score == 0.0
        assert result.process_score == 0.0
        assert len(result.step_rewards) == 0

    def test_batch_scoring(self, recorder_trajectory, task_info):
        """批量评分应正确处理多条轨迹."""
        good_steps = recorder_steps_to_reward_steps(recorder_trajectory)
        good_outcome = recorder_outcome_to_reward_outcome(recorder_trajectory)

        engine = RewardEngine()
        results = engine.score_batch([
            {
                "task": task_info.description,
                "steps": good_steps,
                "outcome": good_outcome,
            },
            {
                "task": task_info.description,
                "steps": [],
                "outcome": {"success": False},
            },
        ])

        assert len(results) == 2
        assert results[0].total_score > results[1].total_score
