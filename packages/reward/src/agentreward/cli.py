"""AgentReward CLI - 命令行界面."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from agentreward import __version__
from agentreward.config import RewardConfig
from agentreward.reward import RewardEngine
from agentreward.rubrics import get_default_rubric_set
from agentreward.preferences import build_preferences, preferences_summary, preferences_to_dicts
from agentreward.calibration import calibrate


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-reward")
def main():
    """AgentReward - 过程级 Reward 计算引擎

    对 Agent 轨迹的每一步计算多维 Rubric Reward，支持规则层 + 模型层 + 人工校准。
    """
    pass


@main.command()
@click.argument("trajectory_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="输出文件路径 (默认: stdout)",
)
@click.option(
    "--rule-weight",
    type=float,
    default=0.6,
    help="规则层权重 (默认: 0.6)",
)
@click.option(
    "--model-weight",
    type=float,
    default=0.4,
    help="模型层权重 (默认: 0.4)",
)
@click.option("-m", "--model", type=str, default="claude-sonnet-4-20250514", help="LLM 模型")
@click.option(
    "-p",
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default="anthropic",
    help="LLM 提供商",
)
def score(
    trajectory_file: str,
    output: Optional[str],
    rule_weight: float,
    model_weight: float,
    model: str,
    provider: str,
):
    """对单条轨迹计算 Reward

    TRAJECTORY_FILE: 轨迹 JSON 文件路径
    """
    with open(trajectory_file, "r", encoding="utf-8") as f:
        trajectory = json.load(f)

    config = RewardConfig(
        rule_weight=rule_weight,
        model_weight=model_weight,
        model_name=model,
        provider=provider,
    )
    engine = RewardEngine(config)
    result = engine.score(trajectory)

    result_dict = result.to_dict()

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        click.echo(f"评分结果已保存: {output}")
    else:
        click.echo(json.dumps(result_dict, ensure_ascii=False, indent=2))

    click.echo(f"\n总分: {result.total_score:.4f}")
    click.echo(f"结果分: {result.outcome_score:.4f}")
    click.echo(f"过程分: {result.process_score:.4f}")
    click.echo(f"步骤数: {len(result.step_rewards)}")


@main.command()
@click.argument("trajectory_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="输出文件路径 (默认: stdout)",
)
def compare(
    trajectory_files: tuple[str, ...],
    output: Optional[str],
):
    """比较同一任务的多条轨迹

    TRAJECTORY_FILES: 多个轨迹 JSON 文件
    """
    if len(trajectory_files) < 2:
        click.echo("至少需要 2 条轨迹进行比较", err=True)
        sys.exit(1)

    config = RewardConfig()
    engine = RewardEngine(config)

    results = []
    for filepath in trajectory_files:
        with open(filepath, "r", encoding="utf-8") as f:
            trajectory = json.load(f)

        reward = engine.score(trajectory)
        results.append(
            {
                "file": filepath,
                "total_score": reward.total_score,
                "outcome_score": reward.outcome_score,
                "process_score": reward.process_score,
                "step_count": len(reward.step_rewards),
            }
        )

    # Sort by total score descending
    results.sort(key=lambda r: r["total_score"], reverse=True)

    click.echo("轨迹比较结果 (按总分降序):\n")
    for i, r in enumerate(results, 1):
        click.echo(
            f"  #{i} {Path(r['file']).name}: "
            f"总分={r['total_score']:.4f} "
            f"结果={r['outcome_score']:.4f} "
            f"过程={r['process_score']:.4f} "
            f"步骤={r['step_count']}"
        )

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        click.echo(f"\n比较结果已保存: {output}")


@main.command()
@click.argument("trajectories_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="输出文件路径 (默认: stdout)",
)
@click.option(
    "--min-margin",
    type=float,
    default=0.05,
    help="最小分数差阈值 (默认: 0.05)",
)
def preferences(
    trajectories_file: str,
    output: Optional[str],
    min_margin: float,
):
    """从轨迹构建偏好对

    TRAJECTORIES_FILE: JSON 文件，格式为 {task_id: [trajectory, ...]}
    """
    with open(trajectories_file, "r", encoding="utf-8") as f:
        trajectories_by_task = json.load(f)

    pairs = build_preferences(trajectories_by_task, min_margin=min_margin)
    summary = preferences_summary(pairs)

    click.echo("偏好对构建完成:")
    click.echo(f"  总对数: {summary['total_pairs']}")
    click.echo(f"  涉及任务: {summary['unique_tasks']}")
    click.echo(f"  平均 margin: {summary['avg_margin']:.4f}")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pairs_data = preferences_to_dicts(pairs)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs_data, f, ensure_ascii=False, indent=2)
        click.echo(f"偏好对已保存: {output}")
    else:
        for p in pairs:
            click.echo(
                f"\n  [{p.task_id}] {p.chosen_trajectory_id} > {p.rejected_trajectory_id} "
                f"(margin={p.margin():.4f})"
            )


@main.command(name="calibrate")
@click.argument("scores_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="输出文件路径 (默认: stdout)",
)
def calibrate_cmd(
    scores_file: str,
    output: Optional[str],
):
    """对 Reward 进行人工校准

    SCORES_FILE: JSON 文件，格式为 {reward_scores: [...], human_scores: [...]}
    """
    with open(scores_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    reward_scores = data.get("reward_scores", [])
    human_scores = data.get("human_scores", [])

    if not reward_scores or not human_scores:
        click.echo("输入文件需要包含 reward_scores 和 human_scores 数组", err=True)
        sys.exit(1)

    try:
        result = calibrate(reward_scores, human_scores)
    except ValueError as e:
        click.echo(f"校准失败: {e}", err=True)
        sys.exit(1)

    click.echo("校准结果:")
    click.echo(f"  Pearson r: {result.pearson_r:.4f}")
    click.echo(f"  Spearman rho: {result.spearman_rho:.4f}")
    click.echo(f"  一致率: {result.agreement_rate:.4f}")

    details = result.details
    if "mean_absolute_error" in details:
        click.echo(f"  MAE: {details['mean_absolute_error']:.4f}")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        click.echo(f"校准结果已保存: {output}")


@main.command()
def rubrics():
    """列出可用的评估 Rubric"""
    rubric_set = get_default_rubric_set()

    click.echo("评估 Rubric 维度:\n")
    for r in rubric_set.rubrics:
        click.echo(f"  {r.id} ({r.name})")
        click.echo(f"    {r.description}")
        click.echo(f"    权重: {r.weight}, 评估方式: {r.evaluator}")
        click.echo()

    click.echo(f"总权重: {rubric_set.total_weight():.2f}")


if __name__ == "__main__":
    main()
