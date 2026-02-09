"""Sandbox configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SandboxConfig:
    """Docker 沙箱环境配置.

    Attributes:
        image: Docker 镜像名称 (默认: python:3.11-slim)
        timeout: 单次命令执行超时 (秒)
        memory_limit: 容器内存限制 (如 '512m', '2g')
        cpu_limit: CPU 核心数限制 (如 1.0, 2.0)
        work_dir: 容器内工作目录
        network_enabled: 是否启用网络访问
        env_vars: 环境变量
    """

    image: str = "python:3.11-slim"
    timeout: int = 300
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    work_dir: str = "/workspace"
    network_enabled: bool = False
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """代码任务配置.

    Attributes:
        repo_url: Git 仓库 URL
        base_commit: 起始 commit SHA
        test_command: 测试命令 (如 'pytest tests/')
        language: 编程语言 (如 'python', 'javascript')
        setup_commands: 环境初始化命令列表
        description: 任务描述
    """

    repo_url: str = ""
    base_commit: str = ""
    test_command: str = ""
    language: str = "python"
    setup_commands: List[str] = field(default_factory=list)
    description: str = ""

    def validate(self) -> List[str]:
        """验证配置，返回错误信息列表.

        Returns:
            错误信息列表，为空表示配置有效
        """
        errors = []
        if not self.repo_url:
            errors.append("repo_url 不能为空")
        if not self.base_commit:
            errors.append("base_commit 不能为空")
        return errors
