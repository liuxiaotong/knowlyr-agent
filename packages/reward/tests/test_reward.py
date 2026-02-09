"""测试 RewardEngine 核心评分引擎."""

import pytest

from agentreward.config import RewardConfig
from agentreward.reward import RewardEngine, StepReward, TrajectoryReward
from agentreward.rubrics import Rubric, RubricSet


# ── 辅助工厂 ──────────────────────────────────────────────────────


def _simple_trajectory(steps=None, outcome=None, task="修复 bug"):
    """构造一个简单的测试轨迹."""
    if steps is None:
        steps = [
            {"tool": "read_file", "params": {"path": "/a.py"}, "output": "def foo(): pass"},
            {"tool": "edit_file", "params": {"path": "/a.py"}, "output": "File edited"},
            {"tool": "bash", "params": {"command": "pytest"}, "output": "1 passed"},
        ]
    if outcome is None:
        outcome = {"success": True, "tests_passed": 1, "tests_total": 1}
    return {"task": task, "steps": steps, "outcome": outcome}


def _failed_trajectory():
    """构造一个失败的轨迹."""
    return {
        "task": "修复 bug",
        "steps": [
            {"tool": "bash", "params": {"command": "pytest"}, "output": "1 failed"},
        ],
        "outcome": {"success": False, "tests_passed": 0, "tests_total": 1},
    }


def _redundant_trajectory():
    """构造一个有冗余操作的轨迹."""
    return {
        "task": "修复 bug",
        "steps": [
            {"tool": "read_file", "params": {"path": "/a.py"}, "output": "code"},
            {"tool": "read_file", "params": {"path": "/a.py"}, "output": "code"},  # 冗余
            {"tool": "read_file", "params": {"path": "/a.py"}, "output": "code"},  # 冗余
            {"tool": "edit_file", "params": {"path": "/a.py"}, "output": "ok"},
        ],
        "outcome": {"success": True, "tests_passed": 1, "tests_total": 1},
    }


# ── StepReward / TrajectoryReward 数据类 ─────────────────────────


class TestStepReward:
    """测试 StepReward 数据类."""

    def test_defaults(self):
        """默认值应正确."""
        sr = StepReward(step_id=0)
        assert sr.step_id == 0
        assert sr.total_score == 0.0
        assert sr.rubric_scores == {}
        assert sr.rationale is None

    def test_to_dict(self):
        """to_dict 应包含所有字段."""
        sr = StepReward(
            step_id=1,
            rubric_scores={"a": 0.8, "b": 0.6},
            total_score=0.7,
            rationale="不错",
        )
        d = sr.to_dict()
        assert d["step_id"] == 1
        assert d["total_score"] == 0.7
        assert d["rubric_scores"]["a"] == 0.8
        assert d["rationale"] == "不错"

    def test_to_dict_rounds(self):
        """total_score 应四舍五入到 4 位."""
        sr = StepReward(step_id=0, total_score=0.33333333)
        d = sr.to_dict()
        assert d["total_score"] == 0.3333


class TestTrajectoryReward:
    """测试 TrajectoryReward 数据类."""

    def test_defaults(self):
        """默认值应正确."""
        tr = TrajectoryReward()
        assert tr.total_score == 0.0
        assert tr.outcome_score == 0.0
        assert tr.process_score == 0.0
        assert tr.step_rewards == []

    def test_to_dict(self):
        """to_dict 应包含所有字段."""
        tr = TrajectoryReward(
            step_rewards=[StepReward(step_id=0, total_score=0.8)],
            total_score=0.7,
            outcome_score=0.5,
            process_score=0.8,
        )
        d = tr.to_dict()
        assert d["step_count"] == 1
        assert d["total_score"] == 0.7
        assert d["outcome_score"] == 0.5
        assert len(d["step_rewards"]) == 1


# ── RewardEngine 测试 ─────────────────────────────────────────────


class TestRewardEngine:
    """测试 RewardEngine 核心评分逻辑."""

    def test_default_config(self):
        """默认配置创建引擎."""
        engine = RewardEngine()
        assert engine.config.rule_weight == 0.6
        assert engine.config.model_weight == 0.4
        assert len(engine.rubric_set.rubrics) == 5

    def test_custom_config(self):
        """自定义配置应生效."""
        config = RewardConfig(rule_weight=0.8, model_weight=0.2)
        engine = RewardEngine(config=config)
        assert engine.config.rule_weight == 0.8

    def test_invalid_weights(self):
        """权重之和不为 1 应抛出 ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RewardConfig(rule_weight=0.5, model_weight=0.3)

    def test_score_simple_trajectory(self):
        """成功轨迹应得到正分."""
        engine = RewardEngine()
        result = engine.score(_simple_trajectory())

        assert isinstance(result, TrajectoryReward)
        assert result.total_score > 0.0
        assert result.outcome_score == 1.0  # 1/1 tests passed
        assert result.process_score > 0.0
        assert len(result.step_rewards) == 3

    def test_score_failed_trajectory(self):
        """失败轨迹的 outcome_score 应为 0."""
        engine = RewardEngine()
        result = engine.score(_failed_trajectory())

        assert result.outcome_score == 0.0
        assert result.total_score < 0.5  # outcome 拉低总分

    def test_score_empty_trajectory(self):
        """空轨迹应返回 0 分."""
        engine = RewardEngine()
        result = engine.score({"task": "空", "steps": [], "outcome": {}})

        assert result.total_score == 0.0
        assert result.process_score == 0.0
        assert len(result.step_rewards) == 0

    def test_redundancy_penalty(self):
        """冗余操作应拉低 process_score."""
        engine = RewardEngine()
        clean_result = engine.score(_simple_trajectory())
        redundant_result = engine.score(_redundant_trajectory())

        assert redundant_result.process_score < clean_result.process_score

    def test_step_rewards_count(self):
        """step_rewards 数量应等于步数."""
        engine = RewardEngine()
        traj = _simple_trajectory()
        result = engine.score(traj)

        assert len(result.step_rewards) == len(traj["steps"])
        for sr in result.step_rewards:
            assert isinstance(sr, StepReward)
            assert 0.0 <= sr.total_score <= 1.0

    def test_step_rubric_scores(self):
        """每步应有所有 rubric 的分数."""
        engine = RewardEngine()
        result = engine.score(_simple_trajectory())

        for sr in result.step_rewards:
            # 默认 rubric set 有 5 个 rubric
            assert len(sr.rubric_scores) >= 2  # 至少 rule-based rubrics

    def test_efficiency_bonus(self):
        """步数少于 reference 应有效率加分."""
        engine = RewardEngine()
        traj = _simple_trajectory()
        traj["reference_steps"] = 10  # 实际 3 步远少于 10 步

        result = engine.score(traj)
        assert result.total_score > 0.0

    def test_outcome_with_partial_tests(self):
        """部分测试通过应得到部分 outcome_score."""
        engine = RewardEngine()
        traj = _simple_trajectory(
            outcome={"success": False, "tests_passed": 3, "tests_total": 5},
        )
        result = engine.score(traj)

        assert abs(result.outcome_score - 0.6) < 0.01

    def test_score_batch(self):
        """score_batch 应返回多个结果."""
        engine = RewardEngine()
        results = engine.score_batch([
            _simple_trajectory(),
            _failed_trajectory(),
        ])

        assert len(results) == 2
        assert results[0].total_score > results[1].total_score

    def test_custom_rubric_set(self):
        """自定义 rubric set 应生效."""
        rubric_set = RubricSet(rubrics=[
            Rubric(
                id="custom", name="自定义",
                description="自定义维度",
                weight=1.0, evaluator="rule",
            ),
        ])
        engine = RewardEngine(rubric_set=rubric_set)
        assert len(engine.rubric_set.rubrics) == 1

    def test_total_score_range(self):
        """total_score 应在 [0, 1] 范围内."""
        engine = RewardEngine()
        for traj in [_simple_trajectory(), _failed_trajectory(), _redundant_trajectory()]:
            result = engine.score(traj)
            assert 0.0 <= result.total_score <= 1.0

    def test_to_dict(self):
        """结果的 to_dict 应可序列化."""
        import json

        engine = RewardEngine()
        result = engine.score(_simple_trajectory())
        d = result.to_dict()

        # 应能 JSON 序列化
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["step_count"] == 3
