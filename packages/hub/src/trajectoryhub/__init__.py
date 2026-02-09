"""AgentTrajectoryHub - Agent 轨迹数据 Pipeline 编排层

串联 Sandbox -> Recorder -> Reward 全流程，产出可训练的数据集。
"""

__version__ = "0.1.0"

from trajectoryhub.pipeline import Pipeline, PipelineConfig
from trajectoryhub.exporter import DatasetExporter

__all__ = ["Pipeline", "PipelineConfig", "DatasetExporter", "__version__"]
