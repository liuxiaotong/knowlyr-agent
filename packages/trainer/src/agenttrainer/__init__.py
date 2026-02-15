"""AgentTrainer - Agent 轨迹训练工具

支持 SFT / DPO / GRPO 三种训练方法，无缝对接 knowlyr-hub 导出的数据集。
"""

import logging

__version__ = "0.1.0"

from agenttrainer.config import TrainConfig, SFTConfig, DPOConfig, GRPOConfig

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "TrainConfig",
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    "__version__",
]
