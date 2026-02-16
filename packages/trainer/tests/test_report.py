"""测试评估报告生成."""

import json

import pytest

from agenttrainer.eval.report import (
    format_comparison_report,
    format_evaluation_report,
    save_report,
)


# ── 测试数据 ───────────────────────────────────────────────────


def _make_eval_results():
    """构造 evaluate_agent() 的模拟返回值."""
    return {
        "success_rate": 0.75,
        "success_rate_ci": (0.68, 0.82),
        "avg_reward": 0.62,
        "std_reward": 0.15,
        "reward_ci": (0.55, 0.69),
        "min_reward": 0.1,
        "max_reward": 0.95,
        "avg_steps": 4.2,
        "std_steps": 1.8,
        "steps_ci": (3.5, 4.9),
        "reward_distribution": {
            "<0.25": 2,
            "0.25-0.5": 3,
            "0.5-0.75": 5,
            ">=0.75": 10,
        },
        "n_episodes": 20,
        "episodes": [
            {"episode": i, "success": True, "total_reward": 0.5, "n_steps": 3}
            for i in range(20)
        ],
    }


def _make_comparison_results():
    """构造 compare_agents() 的模拟返回值."""
    return {
        "baseline": {
            "success_rate": 0.6,
            "success_rate_ci": (0.5, 0.7),
            "avg_reward": 0.45,
            "reward_ci": (0.38, 0.52),
            "avg_steps": 5.2,
            "episodes": [],
        },
        "finetuned": {
            "success_rate": 0.85,
            "success_rate_ci": (0.78, 0.92),
            "avg_reward": 0.72,
            "reward_ci": (0.65, 0.79),
            "avg_steps": 3.1,
            "episodes": [],
        },
        "_leaderboard": [
            {
                "rank": 1,
                "agent": "finetuned",
                "success_rate": 0.85,
                "avg_reward": 0.72,
                "avg_steps": 3.1,
            },
            {
                "rank": 2,
                "agent": "baseline",
                "success_rate": 0.6,
                "avg_reward": 0.45,
                "avg_steps": 5.2,
            },
        ],
        "_comparisons": {
            "baseline_vs_finetuned": {
                "t_statistic": 3.24,
                "p_approx": "p<0.01",
                "significant": True,
                "effect_size": 1.15,
            },
        },
    }


# ── format_evaluation_report ───────────────────────────────────


class TestFormatEvaluationReport:
    """测试单 agent 报告."""

    def test_contains_title(self):
        """报告包含标题."""
        report = format_evaluation_report(_make_eval_results())
        assert "# Agent 评估报告" in report

    def test_contains_metrics(self):
        """报告包含关键指标."""
        report = format_evaluation_report(_make_eval_results())
        assert "75.0%" in report  # success_rate
        assert "0.620" in report  # avg_reward
        assert "95% CI" in report

    def test_contains_distribution(self):
        """报告包含分布表格."""
        report = format_evaluation_report(_make_eval_results())
        assert "Reward 分布" in report
        assert ">=0.75" in report


# ── format_comparison_report ───────────────────────────────────


class TestFormatComparisonReport:
    """测试多 agent 对比报告."""

    def test_leaderboard(self):
        """包含排行榜."""
        report = format_comparison_report(_make_comparison_results())
        assert "排行榜" in report
        assert "finetuned" in report
        assert "baseline" in report

    def test_significance_table(self):
        """包含显著性检验表."""
        report = format_comparison_report(_make_comparison_results())
        assert "Welch's t-test" in report
        assert "baseline_vs_finetuned" in report

    def test_bonferroni_section(self):
        """有 _corrected 时包含校正表."""
        results = _make_comparison_results()
        results["_corrected"] = {"baseline_vs_finetuned": True}
        report = format_comparison_report(results)
        assert "Bonferroni" in report


# ── save_report ────────────────────────────────────────────────


class TestSaveReport:
    """测试报告保存."""

    def test_save_json(self, tmp_path):
        """保存为 JSON."""
        output = tmp_path / "report.json"
        save_report(_make_eval_results(), str(output))
        data = json.loads(output.read_text())
        assert data["success_rate"] == 0.75
        # episodes 应被移除
        assert "episodes" not in data

    def test_save_markdown(self, tmp_path):
        """保存为 Markdown."""
        output = tmp_path / "report.md"
        save_report(_make_eval_results(), str(output))
        content = output.read_text()
        assert "# Agent 评估报告" in content

    def test_save_comparison_markdown(self, tmp_path):
        """保存对比报告为 Markdown."""
        output = tmp_path / "compare.md"
        save_report(_make_comparison_results(), str(output), is_comparison=True)
        content = output.read_text()
        assert "# Agent 对比报告" in content

    def test_invalid_format(self, tmp_path):
        """不支持的格式报错."""
        with pytest.raises(ValueError, match="不支持的输出格式"):
            save_report({}, str(tmp_path / "report.txt"))
