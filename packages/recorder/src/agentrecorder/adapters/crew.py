"""Crew 适配器.

将 knowlyr-crew 的 session JSONL 日志转换为标准轨迹格式。

Crew session JSONL 格式:
- 每行一个 JSON 事件
- event=start: 会话开始，含 session_type, subject, metadata
- event=project_info: 项目检测信息（可选）
- event=message, role=prompt: 发给 LLM 的完整 prompt
- event=message, role=assistant: LLM 的回复（仅 --execute 模式）
- event=message, role=step: 流水线单步 prompt（pipeline 模式）
- event=end: 会话结束，含 status
"""

import json
import logging
from pathlib import Path

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.schema import Outcome, Step, ToolCall, Trajectory
from knowlyrcore import TaskInfo, ToolResult

logger = logging.getLogger(__name__)


class CrewAdapter(BaseAdapter):
    """knowlyr-crew 框架适配器.

    支持解析 .crew/sessions/ 下的 JSONL 格式会话日志。
    只有包含 assistant 回复（即 --execute 模式产生）的 session 才能转为有效轨迹。

    Example:
        >>> adapter = CrewAdapter()
        >>> if adapter.validate(".crew/sessions/20260215-130000-abcd1234.jsonl"):
        ...     trajectory = adapter.parse(".crew/sessions/20260215-130000-abcd1234.jsonl")
    """

    domain: str = "crew"

    def validate(self, log_path: str) -> bool:
        """验证是否为 crew session 日志格式.

        检查规则:
        1. 文件为 .jsonl 格式
        2. 首行 JSON 包含 event=start 和 session_type 字段
        3. 至少包含一条 role=assistant 的消息（有实际执行）
        """
        path = Path(log_path)
        if not path.exists() or path.suffix != ".jsonl":
            return False

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return False

            first = json.loads(lines[0])
            if first.get("event") != "start" or "session_type" not in first:
                return False

            # 必须有 assistant 回复才是有效的执行轨迹
            for line in lines:
                data = json.loads(line)
                if data.get("event") == "message" and data.get("role") == "assistant":
                    return True

            return False
        except (json.JSONDecodeError, OSError):
            return False

    def parse(self, log_path: str) -> Trajectory:
        """将 crew session 日志解析为标准轨迹格式.

        Args:
            log_path: session JSONL 文件路径。

        Returns:
            标准化轨迹对象。
        """
        path = Path(log_path)
        events = self._read_events(path)

        start_event = events[0] if events else {}
        session_type = start_event.get("session_type", "employee")

        if session_type == "pipeline":
            return self._parse_pipeline(events, path.stem)
        return self._parse_employee(events, path.stem)

    def _read_events(self, path: Path) -> list[dict]:
        """读取 JSONL 事件列表."""
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def _parse_employee(self, events: list[dict], session_id: str) -> Trajectory:
        """解析单员工 session."""
        start = events[0] if events else {}
        meta = start.get("metadata", {})
        employee_name = start.get("subject", "unknown")
        args = meta.get("args", {})

        # 提取 prompt 和 assistant 回复
        prompt_content = ""
        assistant_content = ""
        model = ""
        input_tokens = 0
        output_tokens = 0
        timestamp = start.get("timestamp", "")

        for event in events:
            if event.get("event") == "message":
                role = event.get("role", "")
                if role == "prompt":
                    prompt_content = event.get("content", "")
                elif role == "assistant":
                    assistant_content = event.get("content", "")
                    emeta = event.get("metadata", {})
                    model = emeta.get("model", "")
                    input_tokens = emeta.get("input_tokens", 0)
                    output_tokens = emeta.get("output_tokens", 0)

        # 判断结果
        end_event = next((e for e in events if e.get("event") == "end"), {})
        success = end_event.get("status") == "completed"

        # 构建任务描述
        task_desc = args.get("target", "") or args.get("goal", "") or employee_name

        steps = []
        if assistant_content:
            steps.append(Step(
                step_id=0,
                thought=assistant_content,
                tool_call=ToolCall(name="respond", parameters=args),
                tool_result=ToolResult(output=assistant_content, exit_code=0),
                timestamp=timestamp,
                token_count=input_tokens + output_tokens,
            ))

        return Trajectory(
            task=TaskInfo(
                task_id=session_id,
                description=task_desc,
                domain="crew",
            ),
            agent=f"crew/{employee_name}",
            model=model,
            steps=steps,
            outcome=Outcome(
                success=success,
                total_steps=len(steps),
                total_tokens=input_tokens + output_tokens,
            ),
            metadata={
                "employee": employee_name,
                "session_type": "employee",
                "args": args,
            },
        )

    def _parse_pipeline(self, events: list[dict], session_id: str) -> Trajectory:
        """解析流水线 session（多员工串联）."""
        start = events[0] if events else {}
        meta = start.get("metadata", {})
        pipeline_name = start.get("subject", "unknown")
        timestamp = start.get("timestamp", "")

        steps = []
        model = ""
        total_input = 0
        total_output = 0

        # 流水线中每个 step message 是一个员工的 prompt
        # assistant 回复紧跟在 prompt 后面
        step_prompts: list[dict] = []
        assistant_msgs: list[dict] = []

        for event in events:
            if event.get("event") == "message":
                role = event.get("role", "")
                if role == "step":
                    step_prompts.append(event)
                elif role == "assistant":
                    assistant_msgs.append(event)

        # 配对：按顺序匹配 step prompt 和 assistant 回复
        for i, step_event in enumerate(step_prompts):
            smeta = step_event.get("metadata", {})
            emp_name = smeta.get("employee", f"step-{i}")
            step_args = smeta.get("args", {})

            # 找到对应的 assistant 回复（如果有）
            assistant = assistant_msgs[i] if i < len(assistant_msgs) else {}
            content = assistant.get("content", "")
            ameta = assistant.get("metadata", {})

            if ameta.get("model"):
                model = ameta["model"]
            step_input = ameta.get("input_tokens", 0)
            step_output = ameta.get("output_tokens", 0)
            total_input += step_input
            total_output += step_output

            if content:
                steps.append(Step(
                    step_id=i,
                    thought=content,
                    tool_call=ToolCall(
                        name=f"employee/{emp_name}",
                        parameters=step_args,
                    ),
                    tool_result=ToolResult(output=content, exit_code=0),
                    timestamp=step_event.get("timestamp", timestamp),
                    token_count=step_input + step_output,
                ))

        end_event = next((e for e in events if e.get("event") == "end"), {})
        success = end_event.get("status") == "completed"

        return Trajectory(
            task=TaskInfo(
                task_id=session_id,
                description=pipeline_name,
                domain="crew",
            ),
            agent=f"crew/pipeline/{pipeline_name}",
            model=model,
            steps=steps,
            outcome=Outcome(
                success=success,
                total_steps=len(steps),
                total_tokens=total_input + total_output,
            ),
            metadata={
                "pipeline": pipeline_name,
                "session_type": "pipeline",
                "initial_args": meta.get("initial_args", {}),
            },
        )
