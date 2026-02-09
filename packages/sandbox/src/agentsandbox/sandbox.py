"""Core sandbox - Docker 容器化执行环境."""

import logging
from typing import Any, Dict, List, Optional

import docker

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.tools import TOOL_REGISTRY, ToolResult, _exec_in_container

logger = logging.getLogger(__name__)

# 容器标签，用于 list 命令查找
_LABEL = "knowlyr-sandbox"


class Sandbox:
    """Docker 沙箱执行环境.

    提供隔离的 Docker 容器，用于安全执行 Code Agent 的代码操作。
    支持文件读写、Shell 执行、搜索、Git 操作等标准工具接口。

    Usage::

        config = SandboxConfig(image="python:3.11-slim", timeout=300)
        task = TaskConfig(repo_url="https://github.com/...", base_commit="abc123")

        with Sandbox.create(config, task) as sandbox:
            result = sandbox.execute_tool("shell", {"command": "python -m pytest"})
            print(result.output)
    """

    def __init__(self, config: SandboxConfig, task: Optional[TaskConfig] = None):
        """初始化沙箱.

        Args:
            config: 沙箱环境配置
            task: 代码任务配置 (可选)
        """
        self.config = config
        self.task = task
        self._client: Optional[docker.DockerClient] = None
        self._container = None
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
            docker.errors.DockerException: Docker 操作失败
        """
        logger.info("创建沙箱: image=%s, repo=%s, commit=%s",
                    config.image, task.repo_url, task.base_commit)

        sandbox = cls(config, task)
        sandbox._client = docker.from_env()

        # 创建容器
        sandbox._container = sandbox._client.containers.create(
            config.image,
            command="sleep infinity",
            working_dir=config.work_dir,
            mem_limit=config.memory_limit,
            nano_cpus=int(config.cpu_limit * 1e9),
            environment=config.env_vars,
            network_disabled=not config.network_enabled,
            labels={_LABEL: "true"},
            stdin_open=True,
            tty=True,
        )
        sandbox._container.start()
        sandbox._container_id = sandbox._container.short_id
        logger.info("容器已启动: %s", sandbox._container_id)

        # 克隆代码仓库
        if task.repo_url:
            result = _exec_in_container(
                sandbox._container,
                f"git clone {task.repo_url} {config.work_dir}",
            )
            if result.exit_code != 0:
                sandbox.close()
                raise RuntimeError(f"git clone 失败: {result.error or result.output}")

            if task.base_commit:
                result = _exec_in_container(
                    sandbox._container,
                    f"git checkout {task.base_commit}",
                    work_dir=config.work_dir,
                )
                if result.exit_code != 0:
                    sandbox.close()
                    raise RuntimeError(f"git checkout 失败: {result.error or result.output}")

        # 执行 setup 命令
        for cmd in task.setup_commands:
            logger.debug("执行 setup: %s", cmd)
            result = _exec_in_container(sandbox._container, cmd, work_dir=config.work_dir)
            if result.exit_code != 0:
                logger.warning("setup 命令失败 (继续执行): %s -> %s", cmd, result.error)

        logger.info("沙箱就绪: %s", sandbox._container_id)
        return sandbox

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
        """
        if not self._container:
            return ToolResult(output="", exit_code=1, error="沙箱未启动")

        if tool_name not in TOOL_REGISTRY:
            return ToolResult(output="", exit_code=1, error=f"未知工具: {tool_name}")

        logger.debug("执行工具: %s, params=%s", tool_name, params)
        func = TOOL_REGISTRY[tool_name]["function"]
        return func(self._container, **params)

    def reset(self) -> None:
        """重置沙箱到初始状态.

        恢复文件系统、清除所有修改，回到 base_commit 状态。
        """
        if not self._container:
            raise RuntimeError("沙箱未启动")

        work_dir = self.config.work_dir
        base_commit = self.task.base_commit if self.task else ""

        if base_commit:
            _exec_in_container(self._container, f"git checkout {base_commit}", work_dir=work_dir)
            _exec_in_container(self._container, "git clean -fdx", work_dir=work_dir)
        else:
            _exec_in_container(self._container, "git checkout .", work_dir=work_dir)
            _exec_in_container(self._container, "git clean -fdx", work_dir=work_dir)

        logger.info("沙箱已重置: %s", self._container_id)

    def snapshot(self) -> str:
        """保存当前沙箱状态快照.

        创建当前容器的 Docker commit，返回快照 ID。

        Returns:
            快照 ID (image short_id)
        """
        if not self._container:
            raise RuntimeError("沙箱未启动")

        image = self._container.commit(
            repository="knowlyr-sandbox-snapshot",
            tag=f"{self._container_id}-{len(self._snapshots)}",
        )
        self._snapshots.append(image.short_id)
        logger.info("快照已保存: %s", image.short_id)
        return image.short_id

    def close(self) -> None:
        """清理沙箱资源.

        停止并删除 Docker 容器，清理快照镜像。
        """
        if self._container:
            try:
                self._container.stop(timeout=5)
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            logger.info("容器已清理: %s", self._container_id)
            self._container = None
            self._container_id = None

        # 清理快照镜像
        if self._client and self._snapshots:
            for snapshot_id in self._snapshots:
                try:
                    self._client.images.remove(snapshot_id, force=True)
                except Exception:
                    pass
            self._snapshots.clear()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def container_id(self) -> Optional[str]:
        """当前容器 ID."""
        return self._container_id

    @property
    def is_running(self) -> bool:
        """沙箱是否正在运行."""
        return self._container_id is not None
