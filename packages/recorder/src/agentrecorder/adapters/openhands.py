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
import logging
from datetime import datetime, timezone
from pathlib import Path

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.schema import Outcome, Step, ToolCall, Trajectory
from knowlyrcore import TaskInfo, ToolResult

logger = logging.getLogger(__name__)

# OpenHands action 类型到标准工具名的映射
_ACTION_TOOL_MAP = {
    "run": "bash",
    "run_ipython": "ipython",
    "read": "read_file",
    "write": "write_file",
    "browse": "browse_url",
    "edit": "edit_file",
    "think": "think",
}


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
        """
        path = Path(log_path)
        events = self._read_events(path)

        task_info = self._extract_task(events)
        model, agent = self._extract_metadata(events)
        steps = self._pair_events_to_steps(events)
        outcome = self._determine_outcome(events, steps)
        metadata = self._extract_extra_metadata(events)

        return Trajectory(
            task=task_info,
            agent=agent,
            model=model,
            steps=steps,
            outcome=outcome,
            metadata=metadata,
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

    def _read_events(self, path: Path) -> list[dict]:
        """读取 JSONL 事件列表."""
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def _extract_task(self, events: list[dict]) -> TaskInfo:
        """从事件中提取任务信息.

        OpenHands 通常将任务描述放在第一个 message_action 事件中。
        """
        for event in events:
            if event.get("action") == "message" and event.get("source") == "user":
                args = event.get("args", {})
                content = args.get("content", "")
                return TaskInfo(
                    task_id=event.get("args", {}).get("task_id", ""),
                    description=content,
                    metadata={"source": "openhands"},
                )
        return TaskInfo(metadata={"source": "openhands"})

    def _extract_metadata(self, events: list[dict]) -> tuple[str, str]:
        """提取 model 和 agent 信息."""
        model = ""
        agent = "openhands"
        for event in events:
            extras = event.get("extras", {})
            if extras.get("model"):
                model = extras["model"]
            if extras.get("agent_class"):
                agent = f"openhands/{extras['agent_class']}"
            # 有些日志在顶层放 metadata
            if event.get("model"):
                model = event["model"]
        return model, agent

    def _pair_events_to_steps(self, events: list[dict]) -> list[Step]:
        """将 action/observation 事件配对为 Step.

        跳过 message_action（用户输入）和 agent_finish 等非工具事件。
        """
        steps = []
        step_id = 0
        pending_action = None

        for event in events:
            if "action" in event:
                action_type = event["action"]
                source = event.get("source", "")

                # 跳过用户消息和 finish 事件
                if source == "user" or action_type in ("message", "finish"):
                    # 如果前一个 action 未配对，丢弃它
                    pending_action = None
                    continue

                # 记录 agent 的 action，等待后续 observation 配对
                pending_action = event

            elif "observation" in event and pending_action is not None:
                # 配对当前 observation 与上一个 action
                step = self._create_step(step_id, pending_action, event)
                steps.append(step)
                step_id += 1
                pending_action = None

        return steps

    def _create_step(self, step_id: int, action: dict, observation: dict) -> Step:
        """从 action + observation 创建 Step."""
        action_type = action.get("action", "unknown")
        tool_name = _ACTION_TOOL_MAP.get(action_type, action_type)
        args = action.get("args", {})
        thought = action.get("thought", "")

        # 提取 observation 内容
        content = observation.get("content", "")
        extras = observation.get("extras", {})
        exit_code = extras.get("exit_code", 0)
        error = extras.get("error", None)

        # 时间戳
        timestamp = action.get("timestamp", "")
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        return Step(
            step_id=step_id,
            thought=thought,
            tool_call=ToolCall(name=tool_name, parameters=args),
            tool_result=ToolResult(
                output=content,
                exit_code=exit_code if isinstance(exit_code, int) else 0,
                error=error,
            ),
            timestamp=timestamp,
            token_count=action.get("extras", {}).get("token_count"),
        )

    def _determine_outcome(self, events: list[dict], steps: list[Step]) -> Outcome:
        """从事件中判断执行结果."""
        success = False
        tests_passed = 0
        tests_failed = 0

        for event in events:
            # finish 事件通常包含结果
            if event.get("action") == "finish":
                args = event.get("args", {})
                outputs = args.get("outputs", {})
                if "test_result" in outputs:
                    test_result = outputs["test_result"]
                    success = test_result.get("success", False)
                    tests_passed = test_result.get("tests_passed", 0)
                    tests_failed = test_result.get("tests_failed", 0)
                elif outputs.get("success") is not None:
                    success = bool(outputs["success"])
                break

            # 检查最后的测试 observation
            if event.get("observation") == "run":
                content = event.get("content", "")
                exit_code = event.get("extras", {}).get("exit_code", -1)
                if exit_code == 0 and ("passed" in content or "PASSED" in content):
                    success = True

        total_tokens = sum(
            s.token_count for s in steps if s.token_count is not None
        )

        return Outcome(
            success=success,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            total_steps=len(steps),
            total_tokens=total_tokens,
        )

    def _extract_extra_metadata(self, events: list[dict]) -> dict:
        """提取额外元数据."""
        metadata = {}
        for event in events:
            extras = event.get("extras", {})
            if extras.get("instance_id"):
                metadata["instance_id"] = extras["instance_id"]
            if extras.get("max_iterations"):
                metadata["max_iterations"] = extras["max_iterations"]
            if extras.get("agent_class"):
                metadata["agent_class"] = extras["agent_class"]
        return metadata
