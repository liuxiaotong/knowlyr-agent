"""AgentTrajectoryHub - Agent 轨迹数据 Pipeline 编排层

串联 Sandbox -> Recorder -> Reward 全流程，产出可训练的数据集。
"""

import logging

__version__ = "0.1.0"

from trajectoryhub.collect import collect, collect_parallel, make_reward_fn
from trajectoryhub.pipeline import Pipeline, PipelineConfig, PipelineResult, Trajectory
from trajectoryhub.exporter import DatasetExporter
from trajectoryhub.inference import create_model_agent
from trajectoryhub.online import IterationStats, online_training_loop
from trajectoryhub.evaluate import evaluate_agent, compare_agents

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "Trajectory",
    "DatasetExporter",
    "collect",
    "collect_parallel",
    "make_reward_fn",
    "create_model_agent",
    "online_training_loop",
    "IterationStats",
    "evaluate_agent",
    "compare_agents",
    "__version__",
]
