"""SWE-agent 适配器.

将 SWE-agent 框架的执行日志转换为标准轨迹格式。

SWE-agent 日志格式说明:
- 日志通常为 JSON 文件，包含完整的 trajectory 数据
- 包含 history 数组，每个元素是一个 (action, observation) 对
- 元数据包含 model_name, instance_id, model_stats 等
- trajectory 目录结构: <instance_id>/<run_id>/trajectory.json
"""

import json
from pathlib import Path

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.schema import Trajectory


class SWEAgentAdapter(BaseAdapter):
    """SWE-agent 框架适配器.

    支持解析 SWE-agent 的 JSON 格式轨迹文件。

    Example:
        >>> adapter = SWEAgentAdapter()
        >>> if adapter.validate("trajectories/instance/trajectory.json"):
        ...     trajectory = adapter.parse("trajectories/instance/trajectory.json")
    """

    def parse(self, log_path: str) -> Trajectory:
        """将 SWE-agent 日志解析为标准轨迹格式.

        SWE-agent 日志结构:
        - 单个 JSON 文件包含完整轨迹
        - "history" 字段: [[action_dict, observation_str], ...]
        - "info" 字段: 包含 model_stats, exit_status 等
        - "trajectory" 字段: 详细的中间状态

        Args:
            log_path: SWE-agent 轨迹文件路径 (JSON 格式)。

        Returns:
            标准化轨迹对象。

        Raises:
            NotImplementedError: 解析逻辑尚未实现。
        """
        raise NotImplementedError(
            "SWE-agent 适配器解析功能尚未实现。"
            "请参考 SWE-agent 轨迹格式文档实现 history 到 Step 的映射。"
        )

    def validate(self, log_path: str) -> bool:
        """验证是否为 SWE-agent 日志格式.

        检查规则:
        1. 文件必须是 .json 格式
        2. JSON 应包含 SWE-agent 特征字段 (history, info)

        Args:
            log_path: 待验证的文件路径。

        Returns:
            如果是 SWE-agent 日志格式则返回 True。
        """
        path = Path(log_path)
        if not path.exists() or path.suffix != ".json":
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # SWE-agent 轨迹通常包含 history 和 info 字段
                return "history" in data and "info" in data
        except (json.JSONDecodeError, OSError):
            return False
