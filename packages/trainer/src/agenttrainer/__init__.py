"""AgentTrainer - Agent 轨迹训练工具

支持 SFT / DPO / GRPO 三种训练方法，无缝对接 knowlyr-hub 导出的数据集。
"""

import logging

__version__ = "0.1.0"

from agenttrainer.config import TrainConfig, SFTConfig, DPOConfig, GRPOConfig
from agenttrainer.inference import AgentInference, parse_action
from agenttrainer.eval.agent_eval import evaluate_agent, compare_agents
from agenttrainer.eval.report import format_evaluation_report, format_comparison_report, save_report
from agenttrainer.eval.stats import (
    StatTestResult,
    bootstrap_ci,
    welch_t_test,
    paired_t_test,
    mann_whitney_u,
    bonferroni_correct,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # 配置
    "TrainConfig",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    # 推理
    "AgentInference",
    "parse_action",
    # 评估
    "evaluate_agent",
    "compare_agents",
    # 统计检验
    "StatTestResult",
    "bootstrap_ci",
    "welch_t_test",
    "paired_t_test",
    "mann_whitney_u",
    "bonferroni_correct",
    # 报告
    "format_evaluation_report",
    "format_comparison_report",
    "save_report",
    "__version__",
]
