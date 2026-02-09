"""标准化轨迹 Schema 定义.

定义 Agent 执行轨迹的数据模型，所有适配器将日志转换为此格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from knowlyrcore import TaskInfo, ToolResult


class ToolCall(BaseModel):
    """工具调用记录."""

    name: str = Field(description="工具名称")
    parameters: dict[str, Any] = Field(default_factory=dict, description="工具调用参数")


class Step(BaseModel):
    """单步执行记录."""

    step_id: int = Field(description="步骤编号")
    thought: str = Field(description="Agent 思考过程")
    tool_call: ToolCall = Field(description="工具调用")
    tool_result: ToolResult = Field(description="工具执行结果")
    timestamp: str = Field(description="时间戳 (ISO 8601)")
    token_count: int | None = Field(default=None, description="该步骤消耗的 Token 数")


class Outcome(BaseModel):
    """执行结果."""

    success: bool = Field(description="是否成功")
    tests_passed: int = Field(default=0, description="通过的测试数")
    tests_failed: int = Field(default=0, description="失败的测试数")
    total_steps: int = Field(default=0, description="总步骤数")
    total_tokens: int = Field(default=0, description="总 Token 消耗")


class Trajectory(BaseModel):
    """标准化 Agent 执行轨迹.

    这是 AgentRecorder 的核心数据模型，所有适配器都将日志转换为此格式。
    """

    task: TaskInfo = Field(description="任务信息")
    agent: str = Field(description="Agent 框架名称 (如 openhands, swe-agent)")
    model: str = Field(default="", description="使用的 LLM 模型")
    steps: list[Step] = Field(default_factory=list, description="执行步骤列表")
    outcome: Outcome = Field(description="执行结果")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    def to_jsonl(self, path: str | Path) -> None:
        """将轨迹保存为 JSONL 格式.

        Args:
            path: 输出文件路径。如果文件已存在，将追加写入。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(self.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> list[Trajectory]:
        """从 JSONL 文件加载轨迹列表.

        Args:
            path: JSONL 文件路径，每行一个 JSON 对象。

        Returns:
            轨迹对象列表。
        """
        path = Path(path)
        trajectories = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    trajectories.append(cls.model_validate(data))
        return trajectories

    @classmethod
    def schema_json_example(cls) -> str:
        """返回 JSON Schema 的格式化字符串."""
        return json.dumps(cls.model_json_schema(), indent=2, ensure_ascii=False)
