"""核心录制器 - 将 Agent 日志转换为标准化轨迹."""

from pathlib import Path

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.schema import Trajectory


class Recorder:
    """Agent 轨迹录制器.

    使用适配器模式，将不同 Agent 框架的日志转换为标准化轨迹格式。

    Args:
        adapter: Agent 框架适配器实例。

    Example:
        >>> from agentrecorder import Recorder
        >>> from agentrecorder.adapters import OpenHandsAdapter
        >>> recorder = Recorder(OpenHandsAdapter())
        >>> trajectory = recorder.convert("path/to/log.jsonl")
    """

    def __init__(self, adapter: BaseAdapter) -> None:
        self.adapter = adapter

    def convert(self, log_path: str | Path) -> Trajectory:
        """将单个日志文件转换为标准轨迹.

        Args:
            log_path: Agent 日志文件路径。

        Returns:
            标准化轨迹对象。

        Raises:
            FileNotFoundError: 日志文件不存在。
            ValueError: 日志格式不匹配当前适配器。
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"日志文件不存在: {log_path}")

        if not self.adapter.validate(str(log_path)):
            raise ValueError(
                f"日志格式不匹配适配器 {self.adapter.__class__.__name__}: {log_path}"
            )

        return self.adapter.parse(str(log_path))

    def convert_batch(self, log_dir: str | Path, pattern: str = "*") -> list[Trajectory]:
        """批量转换目录下的日志文件.

        Args:
            log_dir: 包含日志文件的目录路径。
            pattern: 文件匹配模式 (默认匹配所有文件)。

        Returns:
            标准化轨迹列表。
        """
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            raise NotADirectoryError(f"目录不存在: {log_dir}")

        trajectories = []
        for log_path in sorted(log_dir.glob(pattern)):
            if log_path.is_file() and self.adapter.validate(str(log_path)):
                trajectory = self.adapter.parse(str(log_path))
                trajectories.append(trajectory)

        return trajectories

    def record(self, sandbox: object, agent: object) -> Trajectory:
        """实时录制 Agent 执行过程 (暂未实现).

        此功能计划支持直接接入 Agent 的执行过程，实时录制轨迹，
        而不是事后从日志解析。

        Args:
            sandbox: 沙箱环境实例。
            agent: Agent 实例。

        Returns:
            录制的标准化轨迹。

        Raises:
            NotImplementedError: 此功能尚未实现。
        """
        raise NotImplementedError(
            "实时录制功能尚未实现。请使用 convert() 从日志文件转换。"
        )
