"""CLI - 命令行界面."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from trajectoryhub import __version__
from trajectoryhub.config import AgentConfig, PipelineConfig, TaskSource
from trajectoryhub.exporter import DatasetExporter
from trajectoryhub.pipeline import Pipeline
from trajectoryhub.tasks import TaskLoader


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-hub")
def main():
    """AgentTrajectoryHub - Agent 轨迹数据 Pipeline 编排层

    串联 Sandbox -> Recorder -> Reward 全流程，产出可训练的数据集。
    """
    pass


@main.command()
@click.argument("task_source", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default="./output", help="输出目录 (默认: ./output)")
@click.option(
    "-f",
    "--framework",
    type=click.Choice(["openhands", "sweagent", "custom"]),
    default="openhands",
    help="Agent 框架",
)
@click.option("-m", "--model", type=str, default="claude-sonnet-4-20250514", help="LLM 模型")
@click.option("--max-steps", type=int, default=30, help="Agent 最大执行步数")
@click.option("-w", "--workers", type=int, default=1, help="并行工作进程数")
@click.option("--resume", type=click.Path(), default=None, help="从 checkpoint 恢复执行")
def run(
    task_source: str,
    output: str,
    framework: str,
    model: str,
    max_steps: int,
    workers: int,
    resume: Optional[str],
):
    """运行完整 Pipeline

    TASK_SOURCE: 任务定义文件路径 (JSONL 格式)
    """
    config = PipelineConfig(
        task_source=TaskSource(path=task_source),
        agents=[
            AgentConfig(
                framework=framework,
                model=model,
                max_steps=max_steps,
            )
        ],
        output_dir=output,
        parallel_workers=workers,
    )

    click.echo("启动 Agent 轨迹 Pipeline...")
    click.echo(f"  任务来源: {task_source}")
    click.echo(f"  Agent: {framework} ({model})")
    click.echo(f"  输出目录: {output}")

    pipeline = Pipeline(config)

    if resume:
        click.echo(f"  从 checkpoint 恢复: {resume}")
        result = pipeline.resume(resume)
    else:
        result = pipeline.run()

    if result.completed > 0 or result.failed > 0:
        click.echo("\nPipeline 执行完成:")
        click.echo(f"  总任务数: {result.total_tasks}")
        click.echo(f"  完成: {result.completed}")
        click.echo(f"  失败: {result.failed}")
        click.echo(f"  轨迹输出: {result.trajectories_path}")
        click.echo(f"  偏好对输出: {result.preferences_path}")
        click.echo(f"  质检报告: {result.quality_report_path}")
        click.echo(f"  耗时: {result.duration_seconds:.1f}s")
    else:
        click.echo("\n没有找到可执行的任务。请检查任务文件格式。")


@main.command()
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["sft", "dpo", "grpo", "benchmark", "huggingface"]),
    required=True,
    help="导出格式",
)
@click.option("-t", "--trajectories", type=click.Path(exists=True), required=True, help="轨迹文件路径")
@click.option("-p", "--preferences", type=click.Path(exists=True), default=None, help="偏好对文件路径")
@click.option("-o", "--output", type=click.Path(), required=True, help="输出文件路径")
@click.option("--repo-id", type=str, default=None, help="HuggingFace 仓库 ID (仅 huggingface 格式)")
def export(
    export_format: str,
    trajectories: str,
    preferences: Optional[str],
    output: str,
    repo_id: Optional[str],
):
    """导出数据集

    将轨迹数据导出为训练格式 (SFT / DPO / Benchmark)。
    """
    exporter = DatasetExporter(
        trajectories_dir=trajectories,
        preferences_dir=preferences,
    )

    click.echo(f"导出数据集 ({export_format})...")

    if export_format == "sft":
        result = exporter.export_sft(output)
    elif export_format == "dpo":
        result = exporter.export_dpo(output)
    elif export_format == "grpo":
        result = exporter.export_grpo(output)
    elif export_format == "benchmark":
        result = exporter.export_benchmark(output)
    elif export_format == "huggingface":
        if not repo_id:
            click.echo("HuggingFace 导出需要指定 --repo-id", err=True)
            sys.exit(1)
        result = exporter.export_huggingface(repo_id)
    else:
        click.echo(f"不支持的格式: {export_format}", err=True)
        sys.exit(1)

    if result.success:
        click.echo("导出成功:")
        click.echo(f"  格式: {result.format}")
        click.echo(f"  输出: {result.output_path}")
        click.echo(f"  记录数: {result.total_records}")
    else:
        click.echo(f"导出失败: {result.error}", err=True)
        sys.exit(1)


@main.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option(
    "-f",
    "--framework",
    type=click.Choice(["openhands", "sweagent", "swe-agent"]),
    required=True,
    help="Agent 框架",
)
@click.option("-o", "--output", type=click.Path(), default="./output", help="输出目录 (默认: ./output)")
@click.option("--save", is_flag=True, help="保存轨迹到输出目录的 trajectories.jsonl")
def process(log_path: str, framework: str, output: str, save: bool):
    """处理单个 Agent 日志文件

    LOG_PATH: Agent 日志文件路径 (OpenHands JSONL / SWE-agent JSON)
    """
    config = PipelineConfig(output_dir=output)
    pipeline = Pipeline(config)

    click.echo(f"处理日志: {log_path} ({framework})...")

    try:
        traj = pipeline.run_from_log(log_path, framework)
    except (RuntimeError, ValueError) as e:
        click.echo(f"处理失败: {e}", err=True)
        sys.exit(1)

    click.echo("处理完成:")
    click.echo(f"  任务 ID: {traj.task_id}")
    click.echo(f"  框架: {traj.agent_framework}")
    click.echo(f"  模型: {traj.agent_model}")
    click.echo(f"  步数: {traj.total_steps}")
    click.echo(f"  成功: {traj.success}")
    click.echo(f"  Reward: {traj.reward:.3f}")
    click.echo(f"  耗时: {traj.duration_seconds:.1f}s")

    if save:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        traj_path = output_dir / "trajectories.jsonl"
        with open(traj_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task_id": traj.task_id,
                "agent_framework": traj.agent_framework,
                "agent_model": traj.agent_model,
                "steps": traj.steps,
                "total_steps": traj.total_steps,
                "success": traj.success,
                "reward": traj.reward,
                "step_rewards": traj.step_rewards,
                "duration_seconds": traj.duration_seconds,
                "metadata": traj.metadata,
            }, ensure_ascii=False) + "\n")
        click.echo(f"  已保存: {traj_path}")


@main.command("process-batch")
@click.argument("log_dir", type=click.Path(exists=True))
@click.option(
    "-f",
    "--framework",
    type=click.Choice(["openhands", "sweagent", "swe-agent"]),
    required=True,
    help="Agent 框架",
)
@click.option("-o", "--output", type=click.Path(), default="./output", help="输出目录 (默认: ./output)")
@click.option("-p", "--pattern", type=str, default="*", help="文件匹配模式 (默认: *)")
def process_batch(log_dir: str, framework: str, output: str, pattern: str):
    """批量处理 Agent 日志目录

    LOG_DIR: 包含日志文件的目录
    """
    config = PipelineConfig(output_dir=output)
    pipeline = Pipeline(config)

    click.echo(f"批量处理日志: {log_dir} ({framework}, pattern={pattern})...")

    try:
        trajectories = pipeline.run_batch_from_logs(log_dir, framework, pattern)
    except (RuntimeError, ValueError) as e:
        click.echo(f"处理失败: {e}", err=True)
        sys.exit(1)

    if not trajectories:
        click.echo("没有找到匹配的日志文件。")
        return

    # 保存轨迹
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / "trajectories.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps({
                "task_id": traj.task_id,
                "agent_framework": traj.agent_framework,
                "agent_model": traj.agent_model,
                "steps": traj.steps,
                "total_steps": traj.total_steps,
                "success": traj.success,
                "reward": traj.reward,
                "step_rewards": traj.step_rewards,
                "duration_seconds": traj.duration_seconds,
                "metadata": traj.metadata,
            }, ensure_ascii=False) + "\n")

    # 统计
    success_count = sum(1 for t in trajectories if t.success)
    avg_reward = sum(t.reward for t in trajectories) / len(trajectories) if trajectories else 0.0

    click.echo("批量处理完成:")
    click.echo(f"  轨迹数: {len(trajectories)}")
    click.echo(f"  成功率: {success_count}/{len(trajectories)}")
    click.echo(f"  平均 Reward: {avg_reward:.3f}")
    click.echo(f"  输出: {traj_path}")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True))
def status(output_dir: str):
    """查看 Pipeline 状态

    OUTPUT_DIR: Pipeline 输出目录
    """
    output = Path(output_dir)

    click.echo(f"Pipeline 状态 ({output_dir}):")

    # Checkpoint
    checkpoint_path = output / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        click.echo(
            f"  Checkpoint: 已完成 {checkpoint.get('completed', 0)} 个, "
            f"失败 {checkpoint.get('failed', 0)} 个"
        )
    else:
        click.echo("  Checkpoint: 无")

    # Trajectories
    trajectories_path = output / "trajectories.jsonl"
    if trajectories_path.exists():
        with open(trajectories_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        click.echo(f"  轨迹文件: {len(lines)} 条")

        # 统计成功率和平均 reward
        rewards = []
        success_count = 0
        for line in lines:
            data = json.loads(line)
            if data.get("success"):
                success_count += 1
            rewards.append(data.get("reward", 0.0))

        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            click.echo(f"  成功率: {success_count}/{len(lines)} ({success_count/len(lines)*100:.1f}%)")
            click.echo(f"  平均 Reward: {avg_reward:.3f}")
    else:
        click.echo("  轨迹文件: 未生成")

    # Preferences
    preferences_path = output / "preferences.jsonl"
    if preferences_path.exists():
        with open(preferences_path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        click.echo(f"  偏好对: {count} 对")
    else:
        click.echo("  偏好对: 未生成")

    # Quality report
    report_path = output / "quality_report.json"
    if report_path.exists():
        click.echo("  质检报告: 已生成")
    else:
        click.echo("  质检报告: 未生成")


@main.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--language", type=str, default=None, help="按语言过滤")
@click.option("--difficulty", type=click.Choice(["easy", "medium", "hard"]), default=None, help="按难度过滤")
@click.option("--type", "task_type", type=str, default=None, help="按类型过滤")
@click.option("--limit", type=int, default=None, help="限制显示数量")
def tasks(
    source: str,
    language: Optional[str],
    difficulty: Optional[str],
    task_type: Optional[str],
    limit: Optional[int],
):
    """列出和过滤任务

    SOURCE: 任务定义文件路径 (JSONL 格式)
    """
    loader = TaskLoader()
    all_tasks = loader.load_from_jsonl(source)

    # 过滤
    filtered = loader.filter_tasks(
        all_tasks,
        language=language,
        difficulty=difficulty,
        task_type=task_type,
    )

    if limit:
        filtered = filtered[:limit]

    click.echo(f"任务列表 (共 {len(filtered)} 个，总计 {len(all_tasks)} 个):")
    click.echo("")

    for task in filtered:
        click.echo(f"  [{task.task_id}]")
        click.echo(f"    类型: {task.type} | 语言: {task.language} | 难度: {task.difficulty}")
        if task.description:
            desc = task.description[:80] + "..." if len(task.description) > 80 else task.description
            click.echo(f"    描述: {desc}")
        if task.repo:
            click.echo(f"    仓库: {task.repo}")
        click.echo("")


@main.command()
@click.option("-t", "--trajectories", type=click.Path(exists=True), required=True, help="轨迹文件路径")
@click.option("-p", "--preferences", type=click.Path(exists=True), default=None, help="偏好对文件路径")
@click.option("--repo-id", type=str, required=True, help="HuggingFace 仓库 ID")
@click.option("--generate-card", is_flag=True, help="同时生成 Dataset Card")
def publish(
    trajectories: str,
    preferences: Optional[str],
    repo_id: str,
    generate_card: bool,
):
    """发布数据集到 HuggingFace

    将轨迹和偏好对数据推送到 HuggingFace Hub。
    """
    exporter = DatasetExporter(
        trajectories_dir=trajectories,
        preferences_dir=preferences,
    )

    if generate_card:
        card = exporter.generate_datacard()
        click.echo("Dataset Card:")
        click.echo(card)
        click.echo("---")

    click.echo(f"发布到 HuggingFace: {repo_id}...")
    result = exporter.export_huggingface(repo_id)

    if result.success:
        click.echo(f"发布成功: https://huggingface.co/datasets/{repo_id}")
    else:
        click.echo(f"发布失败: {result.error}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
