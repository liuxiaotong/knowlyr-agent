"""Standard tool interface for sandbox execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """工具执行结果.

    Attributes:
        output: 标准输出内容
        exit_code: 退出码 (0 表示成功)
        error: 错误信息 (如果有)
    """

    output: str = ""
    exit_code: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """执行是否成功."""
        return self.exit_code == 0 and self.error is None


# ---------------------------------------------------------------------------
# Tool definitions - 标准工具接口
# ---------------------------------------------------------------------------


def file_read(path: str, start_line: int = 0, end_line: int = 0) -> ToolResult:
    """读取文件内容.

    Args:
        path: 文件路径
        start_line: 起始行号 (0 表示从头开始)
        end_line: 结束行号 (0 表示读到末尾)

    Returns:
        ToolResult 包含文件内容

    Raises:
        NotImplementedError: 当前为 stub 实现
    """
    raise NotImplementedError(
        "file_read() 尚未实现。"
        "计划功能: 在 Docker 容器中读取指定路径的文件内容，支持行号范围"
    )


def file_write(path: str, content: str, create_dirs: bool = True) -> ToolResult:
    """写入文件内容.

    Args:
        path: 文件路径
        content: 文件内容
        create_dirs: 是否自动创建父目录

    Returns:
        ToolResult 包含写入结果

    Raises:
        NotImplementedError: 当前为 stub 实现
    """
    raise NotImplementedError(
        "file_write() 尚未实现。"
        "计划功能: 在 Docker 容器中写入文件，支持自动创建目录"
    )


def shell(command: str, timeout: int = 300, work_dir: str = "") -> ToolResult:
    """执行 Shell 命令.

    Args:
        command: Shell 命令
        timeout: 超时时间 (秒)
        work_dir: 工作目录 (空表示使用默认)

    Returns:
        ToolResult 包含命令输出

    Raises:
        NotImplementedError: 当前为 stub 实现
    """
    raise NotImplementedError(
        "shell() 尚未实现。"
        "计划功能: 在 Docker 容器中执行 Shell 命令，捕获 stdout/stderr"
    )


def search(pattern: str, path: str = ".", file_pattern: str = "") -> ToolResult:
    """搜索代码内容.

    Args:
        pattern: 搜索模式 (正则表达式)
        path: 搜索路径
        file_pattern: 文件名过滤 (如 '*.py')

    Returns:
        ToolResult 包含匹配结果

    Raises:
        NotImplementedError: 当前为 stub 实现
    """
    raise NotImplementedError(
        "search() 尚未实现。"
        "计划功能: 在 Docker 容器中使用 grep/ripgrep 搜索代码"
    )


def git(subcommand: str, args: str = "") -> ToolResult:
    """执行 Git 操作.

    Args:
        subcommand: Git 子命令 (如 'diff', 'log', 'status')
        args: 额外参数

    Returns:
        ToolResult 包含 Git 输出

    Raises:
        NotImplementedError: 当前为 stub 实现
    """
    raise NotImplementedError(
        "git() 尚未实现。"
        "计划功能: 在 Docker 容器中执行 Git 命令"
    )


# Tool registry - 工具注册表
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "file_read": {
        "function": file_read,
        "description": "读取文件内容",
        "parameters": {
            "path": {"type": "string", "description": "文件路径", "required": True},
            "start_line": {"type": "integer", "description": "起始行号", "default": 0},
            "end_line": {"type": "integer", "description": "结束行号", "default": 0},
        },
    },
    "file_write": {
        "function": file_write,
        "description": "写入文件内容",
        "parameters": {
            "path": {"type": "string", "description": "文件路径", "required": True},
            "content": {"type": "string", "description": "文件内容", "required": True},
            "create_dirs": {"type": "boolean", "description": "自动创建目录", "default": True},
        },
    },
    "shell": {
        "function": shell,
        "description": "执行 Shell 命令",
        "parameters": {
            "command": {"type": "string", "description": "Shell 命令", "required": True},
            "timeout": {"type": "integer", "description": "超时 (秒)", "default": 300},
            "work_dir": {"type": "string", "description": "工作目录", "default": ""},
        },
    },
    "search": {
        "function": search,
        "description": "搜索代码内容",
        "parameters": {
            "pattern": {"type": "string", "description": "搜索模式", "required": True},
            "path": {"type": "string", "description": "搜索路径", "default": "."},
            "file_pattern": {"type": "string", "description": "文件过滤", "default": ""},
        },
    },
    "git": {
        "function": git,
        "description": "执行 Git 操作",
        "parameters": {
            "subcommand": {"type": "string", "description": "Git 子命令", "required": True},
            "args": {"type": "string", "description": "额外参数", "default": ""},
        },
    },
}
