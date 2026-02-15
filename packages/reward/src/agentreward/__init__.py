"""AgentReward - 过程级 Reward 计算引擎

对 Agent 轨迹的每一步计算多维 Rubric Reward，支持规则层 + 模型层 + 人工校准。
"""

import logging

__version__ = "0.1.0"

from agentreward.reward import RewardEngine, StepReward, TrajectoryReward
from agentreward.rubrics import Rubric, RubricSet, get_rubric_set_for_domain
from agentreward.preferences import PreferencePair

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "RewardEngine",
    "StepReward",
    "TrajectoryReward",
    "Rubric",
    "RubricSet",
    "get_rubric_set_for_domain",
    "PreferencePair",
    "__version__",
]
