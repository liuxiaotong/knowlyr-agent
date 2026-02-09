"""领域配置 — DomainProfile / ToolCategory / ToolSpec / OutcomeSpec.

通过 DomainProfile 声明工具分类和结果判定规则，让 pipeline 支持任意 tool-use agent 领域。
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """工具功能分类."""

    READ = "read"           # 读取/观察状态
    WRITE = "write"         # 修改状态
    SEARCH = "search"       # 搜索/发现
    EXECUTE = "execute"     # 执行命令
    NAVIGATE = "navigate"   # 切换上下文/位置
    SUBMIT = "submit"       # 提交/完成
    THINK = "think"         # 内部推理，无副作用


class ToolSpec(BaseModel):
    """工具规格定义.

    Attributes:
        name: 工具主名称
        category: 功能分类
        stateful_key: 标识操作目标的参数名 (如 file_path / url / element_id)
        aliases: 同一工具的别名列表
    """

    name: str
    category: ToolCategory
    stateful_key: str = ""
    aliases: list[str] = Field(default_factory=list)


class OutcomeSpec(BaseModel):
    """结果判定规格.

    定义如何从 outcome dict 中提取成功/分数信息。

    Attributes:
        success_field: 布尔成功字段名
        score_field: 分数字段名 (如 tests_passed)
        total_field: 总量字段名 (如 tests_total)，与 score_field 配合计算比例
        partial_credit_field: 手动部分得分字段名
    """

    success_field: str = "success"
    score_field: str = ""
    total_field: str = ""
    partial_credit_field: str = "partial_credit"


class DomainProfile(BaseModel):
    """领域配置.

    声明式地描述一个 agent 领域的工具集、结果判定、任务字段等。

    Attributes:
        domain: 领域标识 (如 coding / browser / data_analysis)
        display_name: 可读名称
        tools: 该领域的工具列表
        outcome_spec: 结果判定规则
        task_fields: 领域特有的任务字段 {字段名: 说明}
        default_rubric_weights: 默认 rubric 权重覆盖
    """

    domain: str
    display_name: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    outcome_spec: OutcomeSpec = Field(default_factory=OutcomeSpec)
    task_fields: dict[str, str] = Field(default_factory=dict)
    default_rubric_weights: dict[str, float] = Field(default_factory=dict)


# ============================================================
# 内置领域 Profiles
# ============================================================

CODING_PROFILE = DomainProfile(
    domain="coding",
    display_name="Code Agent",
    tools=[
        ToolSpec(
            name="read_file", category=ToolCategory.READ,
            stateful_key="file_path",
            aliases=["Read", "cat"],
        ),
        ToolSpec(
            name="edit_file", category=ToolCategory.WRITE,
            stateful_key="file_path",
            aliases=["Edit", "sed", "write_file", "Write"],
        ),
        ToolSpec(
            name="bash", category=ToolCategory.EXECUTE,
            aliases=["Bash", "shell", "run"],
        ),
        ToolSpec(
            name="grep", category=ToolCategory.SEARCH,
            aliases=["Grep", "Glob", "find", "ls", "search"],
        ),
        ToolSpec(name="git", category=ToolCategory.EXECUTE),
        ToolSpec(name="ipython", category=ToolCategory.EXECUTE),
        ToolSpec(name="submit", category=ToolCategory.SUBMIT),
        ToolSpec(name="finish", category=ToolCategory.SUBMIT),
        ToolSpec(name="think", category=ToolCategory.THINK),
    ],
    outcome_spec=OutcomeSpec(
        success_field="success",
        score_field="tests_passed",
        total_field="tests_total",
    ),
    task_fields={
        "repo": "Git 仓库 (owner/repo)",
        "base_commit": "基础 commit hash",
        "test_command": "测试命令",
        "language": "编程语言",
    },
)

BROWSER_PROFILE = DomainProfile(
    domain="browser",
    display_name="Browser Agent",
    tools=[
        ToolSpec(
            name="click", category=ToolCategory.WRITE,
            stateful_key="element_id",
            aliases=["tap"],
        ),
        ToolSpec(
            name="type_text", category=ToolCategory.WRITE,
            stateful_key="element_id",
            aliases=["type", "fill", "input"],
        ),
        ToolSpec(
            name="navigate", category=ToolCategory.NAVIGATE,
            stateful_key="url",
            aliases=["goto", "open_url"],
        ),
        ToolSpec(name="screenshot", category=ToolCategory.READ),
        ToolSpec(
            name="scroll", category=ToolCategory.NAVIGATE,
            aliases=["scroll_up", "scroll_down"],
        ),
        ToolSpec(name="wait", category=ToolCategory.READ),
        ToolSpec(name="extract_text", category=ToolCategory.READ),
        ToolSpec(name="select", category=ToolCategory.WRITE),
        ToolSpec(name="submit", category=ToolCategory.SUBMIT),
    ],
    outcome_spec=OutcomeSpec(success_field="success"),
    task_fields={
        "url": "目标 URL",
        "expected_result": "预期结果描述",
    },
)

GENERIC_PROFILE = DomainProfile(
    domain="generic",
    display_name="Generic Tool-Use Agent",
    tools=[],
    outcome_spec=OutcomeSpec(success_field="success"),
)

_BUILTIN_PROFILES: dict[str, DomainProfile] = {
    "coding": CODING_PROFILE,
    "browser": BROWSER_PROFILE,
    "generic": GENERIC_PROFILE,
}


def get_domain_profile(domain: str) -> DomainProfile:
    """获取内置领域 Profile.

    Args:
        domain: 领域标识 (coding / browser / generic)

    Returns:
        对应的 DomainProfile，未找到时返回 GENERIC_PROFILE
    """
    return _BUILTIN_PROFILES.get(domain, GENERIC_PROFILE)


def load_domain_profile(path: str) -> DomainProfile:
    """从 JSON 文件加载自定义 DomainProfile.

    Args:
        path: JSON 文件路径

    Returns:
        DomainProfile 实例
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DomainProfile.model_validate(data)


def list_domain_profiles() -> list[str]:
    """列出所有内置领域名称."""
    return list(_BUILTIN_PROFILES.keys())
