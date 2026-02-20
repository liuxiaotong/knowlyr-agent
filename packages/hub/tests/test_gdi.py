"""GDI 评分模块测试."""

import time

import pytest

from trajectoryhub.gdi import DEFAULT_WEIGHTS, GDIScore, GDIScorer


class TestGDIScorer:
    """GDIScorer 基础功能."""

    def test_default_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.001

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="权重之和"):
            GDIScorer(weights={"intrinsic": 0.5, "utility": 0.5, "feedback": 0.5, "freshness": 0.5})

    def test_score_returns_gdi_score(self):
        scorer = GDIScorer()
        result = scorer.score(reward=0.8)
        assert isinstance(result, GDIScore)
        assert 0 <= result.total <= 1

    def test_high_reward_high_intrinsic(self):
        scorer = GDIScorer()
        high = scorer.score(reward=0.9)
        low = scorer.score(reward=0.1)
        assert high.intrinsic > low.intrinsic
        assert high.total > low.total

    def test_reward_clamped(self):
        scorer = GDIScorer()
        over = scorer.score(reward=1.5)
        under = scorer.score(reward=-0.3)
        assert over.intrinsic == 1.0
        assert under.intrinsic == 0.0

    def test_export_count_increases_utility(self):
        scorer = GDIScorer()
        zero = scorer.score(reward=0.5, export_count=0)
        some = scorer.score(reward=0.5, export_count=10)
        many = scorer.score(reward=0.5, export_count=50)
        assert zero.utility == 0.0
        assert some.utility > zero.utility
        assert many.utility > some.utility

    def test_freshness_decays_over_time(self):
        scorer = GDIScorer()
        now = time.time()
        fresh = scorer.score(reward=0.5, created_at=now)
        old_30d = scorer.score(reward=0.5, created_at=now - 30 * 86400)
        old_90d = scorer.score(reward=0.5, created_at=now - 90 * 86400)
        assert fresh.freshness > old_30d.freshness > old_90d.freshness

    def test_freshness_none_is_max(self):
        scorer = GDIScorer()
        result = scorer.score(reward=0.5, created_at=None)
        assert result.freshness == 1.0

    def test_feedback_default(self):
        scorer = GDIScorer()
        result = scorer.score(reward=0.5)
        assert result.feedback == 0.5

    def test_feedback_clamped(self):
        scorer = GDIScorer()
        over = scorer.score(reward=0.5, feedback_score=1.5)
        assert over.feedback == 1.0


class TestGDIRank:
    """GDIScorer.rank() 排名功能."""

    def test_rank_by_total(self):
        scorer = GDIScorer()
        items = [
            {"reward": 0.3, "export_count": 0},
            {"reward": 0.9, "export_count": 10},
            {"reward": 0.6, "export_count": 5},
        ]
        ranked = scorer.rank(items)
        assert len(ranked) == 3
        assert ranked[0]["reward"] == 0.9
        assert ranked[-1]["reward"] == 0.3
        # 每项有 gdi 字段
        assert all("gdi" in item for item in ranked)

    def test_rank_preserves_original_fields(self):
        scorer = GDIScorer()
        items = [{"reward": 0.5, "my_field": "hello"}]
        ranked = scorer.rank(items)
        assert ranked[0]["my_field"] == "hello"

    def test_rank_empty(self):
        scorer = GDIScorer()
        assert scorer.rank([]) == []
