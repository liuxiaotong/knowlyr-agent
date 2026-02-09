"""OpenHands 适配器.

将 OpenHands (原 OpenDevin) 框架的执行日志转换为标准轨迹格式。

OpenHands 日志格式说明:
- 日志文件通常为 JSONL 格式，每行一个 JSON 对象
- 每个事件包含 action/observation 对
- action 类型包括: CmdRunAction, FileWriteAction, BrowseURLAction 等
- observation 类型包括: CmdOutputObservation, FileReadObservation 等
- 元数据包含 model, agent_class, max_iterations 等
"""

import json
from pathlib import Path

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.schema import Trajectory


class OpenHandsAdapter(BaseAdapter):
    """OpenHands 框架适配器.

    支持解析 OpenHands 的 JSONL 格式日志文件。

    Example:
        >>> adapter = OpenHandsAdapter()
        >>> if adapter.validate("output/log.jsonl"):
        ...     trajectory = adapter.parse("output/log.jsonl")
    """

    def parse(self, log_path: str) -> Trajectory:
        """将 OpenHands 日志解析为标准轨迹格式.

        OpenHands 日志结构:
        - 每行是一个事件 JSON
        - 事件包含 "action" 或 "observation" 字段
        - action 事件: {"id": N, "action": "run", "args": {"command": "..."}, ...}
        - observation 事件: {"id": N, "observation": "run", "content": "...", ...}

        Args:
            log_path: OpenHands 日志文件路径 (JSONL 格式)。

        Returns:
            标准化轨迹对象。

        Raises:
            NotImplementedError: 解析逻辑尚未实现。
        """
        raise NotImplementedError(
            "OpenHands 适配器解析功能尚未实现。"
            "请参考 OpenHands 日志格式文档实现 action/observation 事件到 Step 的映射。"
        )

    def validate(self, log_path: str) -> bool:
        """验证是否为 OpenHands 日志格式.

        检查规则:
        1. 文件必须是 .jsonl 格式
        2. 首行 JSON 应包含 OpenHands 特征字段

        Args:
            log_path: 待验证的文件路径。

        Returns:
            如果是 OpenHands 日志格式则返回 True。
        """
        path = Path(log_path)
        if not path.exists() or path.suffix not in (".jsonl", ".json"):
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                data = json.loads(first_line)
                # OpenHands 日志通常包含 action 或 observation 字段
                return "action" in data or "observation" in data
        except (json.JSONDecodeError, OSError):
            return False
