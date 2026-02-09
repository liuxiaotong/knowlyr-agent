"""共享数据模型 — ToolResult / TaskInfo."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """工具执行结果.

    Attributes:
        output: 标准输出内容
        exit_code: 退出码 (0 表示成功)
        error: 错误信息 (如果有)
    """

    output: str = ""
    exit_code: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        """执行是否成功."""
        return self.exit_code == 0 and self.error is None


class TaskInfo(BaseModel):
    """任务信息.

    描述一个待执行的代码任务，包含仓库、commit、测试命令等。

    Attributes:
        task_id: 任务唯一标识
        description: 任务描述
        type: 任务类型 (如 bug_fix, feature, refactor)
        language: 编程语言
        difficulty: 难度等级 (easy / medium / hard)
        repo: 目标仓库 (如 "owner/repo")
        base_commit: 基础 commit hash
        test_command: 测试命令
        metadata: 额外元数据
    """

    task_id: str = ""
    description: str = ""
    type: str = ""
    language: str = ""
    difficulty: str = ""
    repo: str = ""
    base_commit: str = ""
    test_command: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
