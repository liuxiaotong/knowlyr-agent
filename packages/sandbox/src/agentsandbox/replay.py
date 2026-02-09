"""Trajectory replay - 轨迹重放功能."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentsandbox.sandbox import Sandbox
from agentsandbox.tools import ToolResult


@dataclass
class TrajectoryStep:
    """轨迹中的单个操作步骤.

    Attributes:
        tool_name: 工具名称 (如 'file_read', 'shell')
        params: 工具参数
        expected_output: 期望输出 (可选，用于验证)
    """

    tool_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None


@dataclass
class Trajectory:
    """Agent 执行轨迹.

    Attributes:
        steps: 操作步骤列表
        metadata: 轨迹元数据 (如 agent 名称、模型、时间戳)
    """

    steps: List[TrajectoryStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """从字典创建轨迹对象.

        Args:
            data: 轨迹数据字典

        Returns:
            Trajectory 实例
        """
        steps = [
            TrajectoryStep(
                tool_name=s.get("tool_name", ""),
                params=s.get("params", {}),
                expected_output=s.get("expected_output"),
            )
            for s in data.get("steps", [])
        ]
        return cls(steps=steps, metadata=data.get("metadata", {}))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典.

        Returns:
            轨迹数据字典
        """
        return {
            "steps": [
                {
                    "tool_name": s.tool_name,
                    "params": s.params,
                    "expected_output": s.expected_output,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }


@dataclass
class ReplayResult:
    """轨迹重放结果.

    Attributes:
        success: 重放是否全部成功
        divergence_step: 首次偏离的步骤索引 (-1 表示未偏离)
        details: 每步的详细执行结果
        total_steps: 总步骤数
        completed_steps: 已完成的步骤数
    """

    success: bool = False
    divergence_step: int = -1
    details: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    completed_steps: int = 0


def replay_trajectory(sandbox: Sandbox, trajectory: Trajectory) -> ReplayResult:
    """在沙箱中重放 Agent 执行轨迹.

    逐步执行轨迹中的每个操作，验证输出是否与期望一致。
    如果某步输出偏离预期，记录偏离位置并继续执行。

    Args:
        sandbox: 目标沙箱实例
        trajectory: 要重放的执行轨迹

    Returns:
        ReplayResult 重放结果

    Raises:
        NotImplementedError: 当前为 stub 实现
    """
    raise NotImplementedError(
        "replay_trajectory() 尚未实现。"
        "计划功能: 逐步执行轨迹 → 比较输出 → 记录偏离 → 生成报告"
    )
