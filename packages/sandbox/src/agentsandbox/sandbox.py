"""Core sandbox - Docker 容器化执行环境."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.tools import ToolResult


class Sandbox:
    """Docker 沙箱执行环境.

    提供隔离的 Docker 容器，用于安全执行 Code Agent 的代码操作。
    支持文件读写、Shell 执行、搜索、Git 操作等标准工具接口。

    Usage::

        config = SandboxConfig(image="python:3.11-slim", timeout=300)
        task = TaskConfig(repo_url="https://github.com/...", base_commit="abc123")

        sandbox = Sandbox.create(config, task)
        result = sandbox.execute_tool("shell", {"command": "python -m pytest"})
        print(result.output)
        sandbox.close()
    """

    def __init__(self, config: SandboxConfig, task: Optional[TaskConfig] = None):
        """初始化沙箱.

        Args:
            config: 沙箱环境配置
            task: 代码任务配置 (可选)
        """
        self.config = config
        self.task = task
        self._container_id: Optional[str] = None
        self._snapshots: List[str] = []

    @classmethod
    def create(cls, config: SandboxConfig, task: TaskConfig) -> "Sandbox":
        """创建并启动沙箱环境.

        根据 TaskConfig 拉取代码仓库、切换到指定 commit，
        执行初始化命令，返回就绪的沙箱实例。

        Args:
            config: 沙箱环境配置
            task: 代码任务配置

        Returns:
            初始化完成的 Sandbox 实例

        Raises:
            NotImplementedError: 当前为 stub 实现
        """
        raise NotImplementedError(
            "Sandbox.create() 尚未实现。"
            "计划功能: 创建 Docker 容器 → 克隆代码仓库 → checkout 指定 commit → 执行 setup 命令"
        )

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """在沙箱中执行工具调用.

        支持的工具:
        - file_read: 读取文件内容
        - file_write: 写入文件内容
        - shell: 执行 Shell 命令
        - search: 搜索代码
        - git: 执行 Git 操作

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            ToolResult 执行结果

        Raises:
            NotImplementedError: 当前为 stub 实现
        """
        raise NotImplementedError(
            f"Sandbox.execute_tool('{tool_name}') 尚未实现。"
            "计划功能: 将工具调用映射到 Docker exec 命令并捕获输出"
        )

    def reset(self) -> None:
        """重置沙箱到初始状态.

        恢复文件系统、清除所有修改，回到 base_commit 状态。

        Raises:
            NotImplementedError: 当前为 stub 实现
        """
        raise NotImplementedError(
            "Sandbox.reset() 尚未实现。"
            "计划功能: git checkout + git clean 恢复初始状态"
        )

    def snapshot(self) -> str:
        """保存当前沙箱状态快照.

        创建当前容器的 Docker commit，返回快照 ID。
        可用于后续恢复到此状态。

        Returns:
            快照 ID

        Raises:
            NotImplementedError: 当前为 stub 实现
        """
        raise NotImplementedError(
            "Sandbox.snapshot() 尚未实现。"
            "计划功能: docker commit 保存容器状态，返回镜像 ID"
        )

    def close(self) -> None:
        """清理沙箱资源.

        停止并删除 Docker 容器，清理临时文件和快照。

        Raises:
            NotImplementedError: 当前为 stub 实现
        """
        raise NotImplementedError(
            "Sandbox.close() 尚未实现。"
            "计划功能: 停止容器 → 删除容器 → 清理快照镜像"
        )

    @property
    def container_id(self) -> Optional[str]:
        """当前容器 ID."""
        return self._container_id

    @property
    def is_running(self) -> bool:
        """沙箱是否正在运行."""
        return self._container_id is not None
