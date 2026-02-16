"""测试统计工具模块 — bootstrap, Mann-Whitney U, paired t-test, Bonferroni."""

import statistics

import pytest

from agenttrainer.eval.stats import (
    StatTestResult,
    bootstrap_ci,
    bonferroni_correct,
    confidence_interval,
    mann_whitney_u,
    paired_t_test,
    welch_t_test,
)


# ── confidence_interval ────────────────────────────────────────


class TestConfidenceInterval:
    """测试 t 分布近似置信区间."""

    def test_empty(self):
        """空数据返回 (0, 0)."""
        assert confidence_interval([]) == (0.0, 0.0)

    def test_single_value(self):
        """单值返回 (value, value)."""
        assert confidence_interval([5.0]) == (5.0, 5.0)

    def test_symmetric_around_mean(self):
        """CI 应围绕均值对称."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = confidence_interval(data)
        mean = 3.0
        assert lo < mean < hi
        assert abs((mean - lo) - (hi - mean)) < 0.01


# ── bootstrap_ci ───────────────────────────────────────────────


class TestBootstrapCI:
    """测试 Bootstrap 置信区间."""

    def test_basic(self):
        """基本 bootstrap CI 包含均值."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = bootstrap_ci(data, n_resamples=5000, seed=42)
        mean = statistics.mean(data)
        assert lo < mean < hi

    def test_empty(self):
        """空数据返回 (0, 0)."""
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_single_value(self):
        """单值返回 (value, value)."""
        assert bootstrap_ci([7.0]) == (7.0, 7.0)

    def test_custom_statistic(self):
        """自定义统计量 (median)."""
        data = [1, 2, 3, 100]  # 有离群值
        lo, hi = bootstrap_ci(data, statistic_fn=statistics.median, seed=42)
        assert lo <= statistics.median(data) <= hi

    def test_reproducible_with_seed(self):
        """相同 seed 结果一致."""
        data = [0.1, 0.5, 0.9, 0.3]
        r1 = bootstrap_ci(data, n_resamples=1000, seed=123)
        r2 = bootstrap_ci(data, n_resamples=1000, seed=123)
        assert r1 == r2

    def test_narrow_with_large_sample(self):
        """大样本 CI 较窄."""
        data = [1.0] * 50 + [2.0] * 50
        lo, hi = bootstrap_ci(data, n_resamples=3000, seed=42)
        assert (hi - lo) < 0.3


# ── welch_t_test ───────────────────────────────────────────────


class TestWelchTTest:
    """测试 Welch's t 检验."""

    def test_returns_dataclass(self):
        """返回 StatTestResult."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = welch_t_test(a, b)
        assert isinstance(result, StatTestResult)
        assert result.test_name == "welch_t"

    def test_identical_samples(self):
        """相同数据不显著."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = welch_t_test(data, data)
        assert not result.significant

    def test_very_different(self):
        """差异极大应显著."""
        a = [1.0, 1.1, 1.2, 0.9, 1.0, 1.1, 0.8, 1.2]
        b = [10.0, 10.1, 10.2, 9.9, 10.0, 10.1, 9.8, 10.2]
        result = welch_t_test(a, b)
        assert result.significant
        assert result.effect_size > 1.0

    def test_insufficient_data(self):
        """数据不足."""
        result = welch_t_test([1.0], [2.0])
        assert result.p_value_approx == "insufficient_data"


# ── paired_t_test ──────────────────────────────────────────────


class TestPairedTTest:
    """测试配对 t 检验."""

    def test_basic_paired(self):
        """配对检验正常工作 (差值有方差)."""
        a = [0.5, 0.7, 0.6, 0.9, 0.8]
        b = [0.3, 0.5, 0.5, 0.6, 0.5]
        result = paired_t_test(a, b)
        assert result.test_name == "paired_t"
        assert result.statistic > 0  # a > b

    def test_length_mismatch(self):
        """长度不一致报错."""
        with pytest.raises(ValueError, match="样本长度不一致"):
            paired_t_test([1, 2], [1, 2, 3])

    def test_identical_pairs(self):
        """完全相同 → identical."""
        a = [1, 2, 3, 4]
        result = paired_t_test(a, a)
        assert result.p_value_approx == "identical"
        assert not result.significant

    def test_constant_difference(self):
        """常数差值 → std=0 → identical."""
        a = [5.0, 5.0, 5.0, 5.0]
        b = [3.0, 3.0, 3.0, 3.0]
        result = paired_t_test(a, b)
        assert result.p_value_approx == "identical"

    def test_insufficient_data(self):
        """数据不足."""
        result = paired_t_test([1.0], [2.0])
        assert result.p_value_approx == "insufficient_data"


# ── mann_whitney_u ─────────────────────────────────────────────


class TestMannWhitneyU:
    """测试 Mann-Whitney U 检验."""

    def test_identical_samples(self):
        """相同数据不显著."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = mann_whitney_u(data, data)
        assert not result.significant

    def test_very_different(self):
        """完全分离的数据应显著."""
        a = [1, 1, 2, 2, 1, 2]
        b = [10, 11, 10, 9, 11, 10]
        result = mann_whitney_u(a, b)
        assert result.significant
        assert result.effect_size > 0.8

    def test_ties(self):
        """并列值处理不崩溃."""
        a = [1, 1, 1, 2]
        b = [1, 2, 2, 2]
        result = mann_whitney_u(a, b)
        assert result.test_name == "mann_whitney_u"
        assert "z_score" in result.details

    def test_insufficient_data(self):
        """数据不足."""
        result = mann_whitney_u([1.0], [2.0])
        assert result.p_value_approx == "insufficient_data"

    def test_effect_size_range(self):
        """effect_size 在 [0, 1] 范围内."""
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        result = mann_whitney_u(a, b)
        assert 0.0 <= result.effect_size <= 1.0


# ── bonferroni_correct ─────────────────────────────────────────


class TestBonferroniCorrect:
    """测试 Bonferroni 多重比较校正."""

    def test_single_comparison(self):
        """单次检验: α=0.05, p<0.01 → 显著."""
        assert bonferroni_correct(["p<0.01"]) == [True]

    def test_multiple_comparisons(self):
        """3 次检验: α_corrected ≈ 0.0167."""
        p_values = ["p<0.001", "p<0.05", "p<0.15"]
        result = bonferroni_correct(p_values, alpha=0.05)
        # 0.001 < 0.0167 → True
        # 0.05 > 0.0167 → False
        # 0.15 > 0.0167 → False
        assert result == [True, False, False]

    def test_ten_comparisons(self):
        """10 次检验: α_corrected = 0.005, p<0.01 全部 False."""
        p_values = ["p<0.01"] * 10
        result = bonferroni_correct(p_values)
        assert all(not x for x in result)

    def test_empty(self):
        """空列表."""
        assert bonferroni_correct([]) == []
