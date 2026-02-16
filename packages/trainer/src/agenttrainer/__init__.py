"""AgentTrainer - Agent 轨迹训练工具

支持 SFT / DPO / GRPO 三种训练方法，无缝对接 knowlyr-hub 导出的数据集。
"""

import logging

__version__ = "0.1.0"

from agenttrainer.config import TrainConfig, SFTConfig, DPOConfig, GRPOConfig
from agenttrainer.inference import AgentInference, parse_action
from agenttrainer.eval.agent_eval import evaluate_agent, compare_agents

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "TrainConfig",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    "AgentInference",
    "parse_action",
    "evaluate_agent",
    "compare_agents",
    "__version__",
]
