"""AgentSandbox CLI - 命令行界面."""

import json
import sys

import click
import docker

from agentsandbox import __version__
from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.sandbox import Sandbox, _LABEL


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-sandbox")
def main():
    """AgentSandbox - Code Agent 执行沙箱

    提供可复现的 Docker 执行环境，支持代码任务的隔离执行与轨迹重放。
    """
    pass


@main.command()
@click.option("--repo", required=True, help="Git 仓库 URL")
@click.option("--commit", required=True, help="起始 commit SHA")
@click.option("--language", default="python", help="编程语言 (默认: python)")
@click.option("--image", default="python:3.11-slim", help="Docker 镜像")
@click.option("--timeout", type=int, default=300, help="超时秒数 (默认: 300)")
@click.option("--memory", default="512m", help="内存限制 (默认: 512m)")
@click.option("--cpu", type=float, default=1.0, help="CPU 限制 (默认: 1.0)")
def create(
    repo: str,
    commit: str,
    language: str,
    image: str,
    timeout: int,
    memory: str,
    cpu: float,
):
    """创建沙箱环境

    根据 Git 仓库和 commit 创建隔离的 Docker 执行环境。
    """
    sandbox_config = SandboxConfig(
        image=image,
        timeout=timeout,
        memory_limit=memory,
        cpu_limit=cpu,
    )
    task_config = TaskConfig(
        repo_url=repo,
        base_commit=commit,
        language=language,
    )

    errors = task_config.validate()
    if errors:
        for err in errors:
            click.echo(f"  配置错误: {err}", err=True)
        sys.exit(1)

    click.echo("正在创建沙箱...")
    try:
        sandbox = Sandbox.create(sandbox_config, task_config)
        click.echo(f"沙箱已创建: {sandbox.container_id}")
        click.echo(f"  镜像: {image}")
        click.echo(f"  仓库: {repo}")
        click.echo(f"  Commit: {commit}")
    except Exception as e:
        click.echo(f"创建失败: {e}", err=True)
        sys.exit(1)


@main.command("exec")
@click.argument("sandbox_id")
@click.option("--tool", required=True, type=click.Choice(
    ["file_read", "file_write", "shell", "search", "git"]
), help="工具名称")
@click.option("--params", required=True, help="工具参数 (JSON 格式)")
def exec_tool(sandbox_id: str, tool: str, params: str):
    """在沙箱中执行工具

    SANDBOX_ID: 目标沙箱 ID
    """
    try:
        tool_params = json.loads(params)
    except json.JSONDecodeError as e:
        click.echo(f"  参数解析失败: {e}", err=True)
        sys.exit(1)

    try:
        client = docker.from_env()
        containers = client.containers.list(filters={"label": _LABEL})
        container = next(
            (c for c in containers if c.short_id == sandbox_id or c.id.startswith(sandbox_id)),
            None,
        )
        if not container:
            click.echo(f"沙箱不存在: {sandbox_id}", err=True)
            sys.exit(1)

        # 构建临时 Sandbox 对象来执行工具
        sandbox = Sandbox(SandboxConfig())
        sandbox._container = container
        sandbox._container_id = container.short_id

        result = sandbox.execute_tool(tool, tool_params)
        if result.output:
            click.echo(result.output)
        if result.error:
            click.echo(f"错误: {result.error}", err=True)
        sys.exit(result.exit_code)
    except Exception as e:
        click.echo(f"执行失败: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("sandbox_id")
def reset(sandbox_id: str):
    """重置沙箱到初始状态

    SANDBOX_ID: 目标沙箱 ID
    """
    try:
        client = docker.from_env()
        containers = client.containers.list(filters={"label": _LABEL})
        container = next(
            (c for c in containers if c.short_id == sandbox_id or c.id.startswith(sandbox_id)),
            None,
        )
        if not container:
            click.echo(f"沙箱不存在: {sandbox_id}", err=True)
            sys.exit(1)

        sandbox = Sandbox(SandboxConfig())
        sandbox._container = container
        sandbox._container_id = container.short_id
        sandbox.reset()
        click.echo(f"沙箱已重置: {sandbox_id}")
    except Exception as e:
        click.echo(f"重置失败: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("sandbox_id")
@click.argument("trajectory_file", type=click.Path(exists=True))
def replay(sandbox_id: str, trajectory_file: str):
    """重放 Agent 执行轨迹

    SANDBOX_ID: 目标沙箱 ID
    TRAJECTORY_FILE: 轨迹文件路径 (JSON)
    """
    with open(trajectory_file, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    from agentsandbox.replay import Trajectory, replay_trajectory

    trajectory = Trajectory.from_dict(trajectory_data)

    try:
        client = docker.from_env()
        containers = client.containers.list(filters={"label": _LABEL})
        container = next(
            (c for c in containers if c.short_id == sandbox_id or c.id.startswith(sandbox_id)),
            None,
        )
        if not container:
            click.echo(f"沙箱不存在: {sandbox_id}", err=True)
            sys.exit(1)

        sandbox = Sandbox(SandboxConfig())
        sandbox._container = container
        sandbox._container_id = container.short_id

        click.echo(f"重放轨迹: {len(trajectory.steps)} 步")
        result = replay_trajectory(sandbox, trajectory)

        click.echo(f"完成: {result.completed_steps}/{result.total_steps} 步")
        click.echo(f"成功: {result.success}")
        if result.divergence_step >= 0:
            click.echo(f"首次偏离: step {result.divergence_step}")
    except Exception as e:
        click.echo(f"重放失败: {e}", err=True)
        sys.exit(1)


@main.command("list")
def list_sandboxes():
    """列出活跃的沙箱"""
    try:
        client = docker.from_env()
        containers = client.containers.list(filters={"label": _LABEL})

        if not containers:
            click.echo("  (无活跃沙箱)")
            return

        click.echo(f"活跃沙箱 ({len(containers)} 个):")
        for c in containers:
            click.echo(f"  {c.short_id}  {c.image.tags[0] if c.image.tags else c.image.short_id}"
                       f"  {c.status}")
    except docker.errors.DockerException as e:
        click.echo(f"Docker 连接失败: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
