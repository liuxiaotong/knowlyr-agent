"""AgentRecorder CLI - 命令行界面."""

import sys
from pathlib import Path
from typing import Optional

import click

from agentrecorder import __version__
from agentrecorder.schema import Trajectory


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-recorder")
def main():
    """AgentRecorder - Agent 轨迹录制工具

    将 Agent 框架的执行日志转换为标准化轨迹格式。
    """
    pass


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option(
    "-f",
    "--framework",
    type=click.Choice(["openhands", "swe-agent"]),
    required=True,
    help="Agent 框架名称",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="输出 JSONL 文件路径",
)
def convert(log_path: str, framework: str, output: Optional[str]):
    """将 Agent 日志转换为标准化轨迹格式

    LOG_PATH: Agent 日志文件路径
    """
    adapter = _get_adapter(framework)
    if adapter is None:
        click.echo(f"不支持的框架: {framework}", err=True)
        sys.exit(1)

    if not adapter.validate(log_path):
        click.echo(f"日志格式不匹配 {framework}: {log_path}", err=True)
        sys.exit(1)

    click.echo("正在转换日志...")
    click.echo(f"  文件: {log_path}")
    click.echo(f"  框架: {framework}")

    try:
        from agentrecorder.recorder import Recorder

        recorder = Recorder(adapter)
        trajectory = recorder.convert(log_path)

        if output:
            trajectory.to_jsonl(output)
            click.echo(f"转换成功: {output}")
            click.echo(f"  步骤数: {len(trajectory.steps)}")
            click.echo(f"  结果: {'成功' if trajectory.outcome.success else '失败'}")
        else:
            click.echo(trajectory.model_dump_json(indent=2))
    except NotImplementedError as e:
        click.echo(f"适配器尚未实现: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"转换失败: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option(
    "-f",
    "--framework",
    type=click.Choice(["openhands", "swe-agent"]),
    help="指定框架进行验证（不指定则自动检测）",
)
def validate(log_path: str, framework: Optional[str]):
    """验证日志文件格式

    LOG_PATH: 待验证的日志文件路径
    """
    if framework:
        adapter = _get_adapter(framework)
        if adapter is None:
            click.echo(f"不支持的框架: {framework}", err=True)
            sys.exit(1)

        is_valid = adapter.validate(log_path)
        if is_valid:
            click.echo(f"有效: {log_path} 是 {framework} 格式")
        else:
            click.echo(f"无效: {log_path} 不是 {framework} 格式")
            sys.exit(1)
    else:
        click.echo(f"自动检测: {log_path}")
        matched = []
        for fw_name in ["openhands", "swe-agent"]:
            adapter = _get_adapter(fw_name)
            if adapter and adapter.validate(log_path):
                matched.append(fw_name)

        if matched:
            click.echo(f"  匹配框架: {', '.join(matched)}")
        else:
            click.echo("  未匹配任何已知框架格式")
            sys.exit(1)


@main.command()
def schema():
    """输出标准化轨迹的 JSON Schema"""
    click.echo(Trajectory.schema_json_example())


@main.command()
@click.argument("log_dir", type=click.Path(exists=True))
@click.option(
    "-f",
    "--framework",
    type=click.Choice(["openhands", "swe-agent"]),
    required=True,
    help="Agent 框架名称",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="输出 JSONL 文件路径",
)
@click.option(
    "-p",
    "--pattern",
    type=str,
    default="*",
    help="文件匹配模式 (默认: *)",
)
def batch(log_dir: str, framework: str, output: str, pattern: str):
    """批量转换目录下的日志文件

    LOG_DIR: 包含日志文件的目录路径
    """
    adapter = _get_adapter(framework)
    if adapter is None:
        click.echo(f"不支持的框架: {framework}", err=True)
        sys.exit(1)

    click.echo("正在批量转换...")
    click.echo(f"  目录: {log_dir}")
    click.echo(f"  框架: {framework}")
    click.echo(f"  模式: {pattern}")

    try:
        from agentrecorder.recorder import Recorder

        recorder = Recorder(adapter)
        trajectories = recorder.convert_batch(log_dir, pattern=pattern)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for trajectory in trajectories:
            trajectory.to_jsonl(output_path)

        click.echo(f"批量转换完成: {output}")
        click.echo(f"  转换文件数: {len(trajectories)}")
    except NotImplementedError as e:
        click.echo(f"适配器尚未实现: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"转换失败: {e}", err=True)
        sys.exit(1)


def _get_adapter(framework: str):
    """根据框架名称获取适配器实例."""
    if framework == "openhands":
        from agentrecorder.adapters.openhands import OpenHandsAdapter

        return OpenHandsAdapter()
    elif framework == "swe-agent":
        from agentrecorder.adapters.sweagent import SWEAgentAdapter

        return SWEAgentAdapter()
    return None


if __name__ == "__main__":
    main()
