"""Trajectory replay - 轨迹重放功能."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentsandbox.sandbox import Sandbox

logger = logging.getLogger(__name__)


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
    """
    total = len(trajectory.steps)
    logger.info("开始重放轨迹: %d 步", total)

    details = []
    divergence_step = -1
    completed = 0

    for i, step in enumerate(trajectory.steps):
        result = sandbox.execute_tool(step.tool_name, step.params)
        completed += 1

        diverged = False
        if step.expected_output is not None and result.output.strip() != step.expected_output.strip():
            diverged = True
            if divergence_step == -1:
                divergence_step = i
                logger.warning("轨迹偏离 (step %d): tool=%s", i, step.tool_name)

        details.append({
            "step": i,
            "tool_name": step.tool_name,
            "exit_code": result.exit_code,
            "diverged": diverged,
            "output_preview": result.output[:200] if result.output else "",
        })

        if not result.success:
            logger.warning("步骤执行失败 (step %d): %s -> exit_code=%d",
                          i, step.tool_name, result.exit_code)

    success = divergence_step == -1 and all(d["exit_code"] == 0 for d in details)
    logger.info("重放完成: %d/%d 步, success=%s", completed, total, success)

    return ReplayResult(
        success=success,
        divergence_step=divergence_step,
        details=details,
        total_steps=total,
        completed_steps=completed,
    )
