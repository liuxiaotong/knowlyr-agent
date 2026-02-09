"""AgentSandbox CLI - 命令行界面."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from agentsandbox import __version__
from agentsandbox.config import SandboxConfig, TaskConfig


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
    click.echo(f"  仓库: {repo}")
    click.echo(f"  Commit: {commit}")
    click.echo(f"  镜像: {image}")
    click.echo(f"  超时: {timeout}s")
    click.echo(f"  内存: {memory}")
    click.echo(f"  CPU: {cpu}")

    # TODO: 实际创建沙箱
    click.echo("\n[未实现] 沙箱创建功能尚在开发中", err=True)
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

    click.echo(f"在沙箱 {sandbox_id} 中执行工具...")
    click.echo(f"  工具: {tool}")
    click.echo(f"  参数: {json.dumps(tool_params, ensure_ascii=False, indent=2)}")

    # TODO: 实际执行工具
    click.echo("\n[未实现] 工具执行功能尚在开发中", err=True)
    sys.exit(1)


@main.command()
@click.argument("sandbox_id")
def reset(sandbox_id: str):
    """重置沙箱到初始状态

    SANDBOX_ID: 目标沙箱 ID
    """
    click.echo(f"正在重置沙箱 {sandbox_id}...")

    # TODO: 实际重置沙箱
    click.echo("\n[未实现] 沙箱重置功能尚在开发中", err=True)
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

    from agentsandbox.replay import Trajectory

    trajectory = Trajectory.from_dict(trajectory_data)

    click.echo(f"在沙箱 {sandbox_id} 中重放轨迹...")
    click.echo(f"  步骤数: {len(trajectory.steps)}")
    click.echo(f"  元数据: {json.dumps(trajectory.metadata, ensure_ascii=False)}")

    # TODO: 实际重放轨迹
    click.echo("\n[未实现] 轨迹重放功能尚在开发中", err=True)
    sys.exit(1)


@main.command("list")
def list_sandboxes():
    """列出活跃的沙箱"""
    click.echo("活跃沙箱列表:")

    # TODO: 查询 Docker 容器
    click.echo("  (无活跃沙箱)")


if __name__ == "__main__":
    main()
