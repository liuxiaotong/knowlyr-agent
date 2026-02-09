"""AgentTrajectoryHub - Agent 轨迹数据 Pipeline 编排层

串联 Sandbox -> Recorder -> Reward 全流程，产出可训练的数据集。
"""

import logging

__version__ = "0.1.0"

from trajectoryhub.collect import collect
from trajectoryhub.pipeline import Pipeline, PipelineConfig, PipelineResult, Trajectory
from trajectoryhub.exporter import DatasetExporter

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "Trajectory",
    "DatasetExporter",
    "collect",
    "__version__",
]
