"""Standard tool interface for sandbox execution."""

import io
import logging
import tarfile
from typing import Any, Dict

from knowlyrcore import ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _exec_in_container(container, cmd: str, timeout: int = 300, work_dir: str = "") -> ToolResult:
    """在容器中执行命令，返回 ToolResult."""
    try:
        result = container.exec_run(
            ["bash", "-c", cmd],
            demux=True,
            workdir=work_dir or None,
        )
        stdout, stderr = result.output or (None, None)
        output = (stdout or b"").decode("utf-8", errors="replace")
        err = (stderr or b"").decode("utf-8", errors="replace")
        return ToolResult(
            output=output,
            exit_code=result.exit_code,
            error=err if err else None,
        )
    except Exception as e:
        logger.exception("容器命令执行失败: %s", cmd[:100])
        return ToolResult(output="", exit_code=1, error=str(e))


# ---------------------------------------------------------------------------
# Tool definitions - 标准工具接口
# ---------------------------------------------------------------------------


def file_read(container, path: str, start_line: int = 0, end_line: int = 0) -> ToolResult:
    """读取文件内容.

    Args:
        container: Docker 容器实例
        path: 文件路径
        start_line: 起始行号 (0 表示从头开始)
        end_line: 结束行号 (0 表示读到末尾)

    Returns:
        ToolResult 包含文件内容
    """
    if start_line > 0 and end_line > 0:
        cmd = f"sed -n '{start_line},{end_line}p' {path}"
    elif start_line > 0:
        cmd = f"sed -n '{start_line},$p' {path}"
    else:
        cmd = f"cat {path}"
    return _exec_in_container(container, cmd)


def file_write(container, path: str, content: str, create_dirs: bool = True) -> ToolResult:
    """写入文件内容.

    使用 tar archive 写入文件到容器，避免 shell 转义问题。

    Args:
        container: Docker 容器实例
        path: 文件路径
        content: 文件内容
        create_dirs: 是否自动创建父目录

    Returns:
        ToolResult 包含写入结果
    """
    try:
        if create_dirs:
            # 获取父目录路径
            parent = "/".join(path.rsplit("/", 1)[:-1]) if "/" in path else ""
            if parent:
                _exec_in_container(container, f"mkdir -p {parent}")

        # 用 tar 写入文件，避免 content 中特殊字符的转义问题
        data = content.encode("utf-8")
        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tarstream.seek(0)
        container.put_archive("/", tarstream)

        return ToolResult(output=f"已写入 {len(data)} 字节到 {path}", exit_code=0)
    except Exception as e:
        logger.exception("文件写入失败: %s", path)
        return ToolResult(output="", exit_code=1, error=str(e))


def shell(container, command: str, timeout: int = 300, work_dir: str = "") -> ToolResult:
    """执行 Shell 命令.

    Args:
        container: Docker 容器实例
        command: Shell 命令
        timeout: 超时时间 (秒)
        work_dir: 工作目录 (空表示使用默认)

    Returns:
        ToolResult 包含命令输出
    """
    return _exec_in_container(container, command, timeout=timeout, work_dir=work_dir)


def search(container, pattern: str, path: str = ".", file_pattern: str = "") -> ToolResult:
    """搜索代码内容.

    Args:
        container: Docker 容器实例
        pattern: 搜索模式 (正则表达式)
        path: 搜索路径
        file_pattern: 文件名过滤 (如 '*.py')

    Returns:
        ToolResult 包含匹配结果
    """
    cmd = f"grep -rn '{pattern}' {path}"
    if file_pattern:
        cmd += f" --include='{file_pattern}'"
    # grep 返回 1 表示无匹配（非错误）
    result = _exec_in_container(container, cmd)
    if result.exit_code == 1 and not result.error:
        return ToolResult(output="", exit_code=0, error=None)
    return result


def git(container, subcommand: str, args: str = "") -> ToolResult:
    """执行 Git 操作.

    Args:
        container: Docker 容器实例
        subcommand: Git 子命令 (如 'diff', 'log', 'status')
        args: 额外参数

    Returns:
        ToolResult 包含 Git 输出
    """
    cmd = f"git {subcommand}"
    if args:
        cmd += f" {args}"
    return _exec_in_container(container, cmd)


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
