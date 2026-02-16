"""评估工具.

- evaluator.py: 训练指标（perplexity、token accuracy）
- agent_eval.py: 部署指标（成功率、reward 分布、步数统计）
- stats.py: 统计检验 (bootstrap, Mann-Whitney U, paired t-test, Bonferroni)
- report.py: Markdown / JSON 报告生成
"""

from agenttrainer.eval.agent_eval import (
    evaluate_agent,
    compare_agents,
    significance_test,
)
from agenttrainer.eval.stats import (
    StatTestResult,
    confidence_interval,
    bootstrap_ci,
    welch_t_test,
    paired_t_test,
    mann_whitney_u,
    bonferroni_correct,
)
from agenttrainer.eval.report import (
    format_evaluation_report,
    format_comparison_report,
    save_report,
)

__all__ = [
    # 主评估函数
    "evaluate_agent",
    "compare_agents",
    "significance_test",
    # 统计工具
    "StatTestResult",
    "confidence_interval",
    "bootstrap_ci",
    "welch_t_test",
    "paired_t_test",
    "mann_whitney_u",
    "bonferroni_correct",
    # 报告生成
    "format_evaluation_report",
    "format_comparison_report",
    "save_report",
]
