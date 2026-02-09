"""SWE-agent 适配器.

将 SWE-agent 框架的执行日志转换为标准轨迹格式。

SWE-agent 日志格式说明:
- 日志通常为 JSON 文件，包含完整的 trajectory 数据
- 包含 history 数组，每个元素是一个 (action, observation) 对
- 元数据包含 model_name, instance_id, model_stats 等
- trajectory 目录结构: <instance_id>/<run_id>/trajectory.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.schema import Outcome, Step, ToolCall, Trajectory
from knowlyrcore import TaskInfo, ToolResult

logger = logging.getLogger(__name__)

# SWE-agent action 到标准工具名的映射
_ACTION_TOOL_MAP = {
    "bash": "bash",
    "edit": "edit_file",
    "open": "read_file",
    "scroll_up": "read_file",
    "scroll_down": "read_file",
    "search_dir": "search",
    "search_file": "search",
    "find_file": "search",
    "create": "write_file",
    "submit": "submit",
    "exit": "finish",
    "think": "think",
}


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
        """
        path = Path(log_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_info = self._extract_task(data)
        model = self._extract_model(data)
        steps = self._parse_history(data.get("history", []))
        outcome = self._determine_outcome(data, steps)
        metadata = self._extract_metadata(data)

        return Trajectory(
            task=task_info,
            agent="swe-agent",
            model=model,
            steps=steps,
            outcome=outcome,
            metadata=metadata,
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

    def _extract_task(self, data: dict) -> TaskInfo:
        """从数据中提取任务信息."""
        info = data.get("info", {})
        instance_id = info.get("instance_id", data.get("instance_id", ""))
        problem_statement = data.get("problem_statement", "")

        return TaskInfo(
            task_id=instance_id,
            description=problem_statement,
            repo=info.get("repo", ""),
            base_commit=info.get("base_commit", ""),
            metadata={"source": "swe-agent"},
        )

    def _extract_model(self, data: dict) -> str:
        """提取模型名称."""
        info = data.get("info", {})
        return info.get("model_name", data.get("model_name_or_path", ""))

    def _parse_history(self, history: list) -> list[Step]:
        """将 history 数组转换为 Step 列表.

        SWE-agent history 格式:
        - 每个元素是 [action_dict, observation_str]
        - action_dict 包含 action 名称和参数
        - observation_str 是工具输出的字符串
        """
        steps = []
        now = datetime.now(timezone.utc).isoformat()

        for step_id, entry in enumerate(history):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue

            action_data, observation = entry[0], entry[1]

            # action_data 可以是 dict 或 str
            if isinstance(action_data, dict):
                action_name = action_data.get("action", "unknown")
                args = {k: v for k, v in action_data.items() if k != "action"}
                thought = action_data.get("thought", "")
            else:
                # 纯字符串命令
                action_name = "bash"
                args = {"command": str(action_data)}
                thought = ""

            # 跳过 think 步骤（仅思考，无工具调用）
            if action_name == "think":
                continue

            tool_name = _ACTION_TOOL_MAP.get(action_name, action_name)
            obs_str = str(observation) if observation else ""

            # 判断 exit_code
            exit_code = 0
            if action_name in ("bash", "run") and obs_str.startswith("Error"):
                exit_code = 1

            step = Step(
                step_id=len(steps),
                thought=thought,
                tool_call=ToolCall(name=tool_name, parameters=args),
                tool_result=ToolResult(output=obs_str, exit_code=exit_code),
                timestamp=now,
            )
            steps.append(step)

        return steps

    def _determine_outcome(self, data: dict, steps: list[Step]) -> Outcome:
        """判断执行结果."""
        info = data.get("info", {})
        exit_status = info.get("exit_status", "")

        success = exit_status == "submitted"

        # 模型统计
        model_stats = info.get("model_stats", {})
        total_tokens = (
            model_stats.get("tokens_sent", 0)
            + model_stats.get("tokens_received", 0)
        )

        return Outcome(
            success=success,
            total_steps=len(steps),
            total_tokens=total_tokens,
        )

    def _extract_metadata(self, data: dict) -> dict:
        """提取额外元数据."""
        info = data.get("info", {})
        metadata = {}

        if info.get("instance_id"):
            metadata["instance_id"] = info["instance_id"]
        if info.get("model_stats"):
            metadata["model_stats"] = info["model_stats"]
        if info.get("exit_status"):
            metadata["exit_status"] = info["exit_status"]
        if data.get("environment"):
            metadata["environment"] = data["environment"]

        return metadata
