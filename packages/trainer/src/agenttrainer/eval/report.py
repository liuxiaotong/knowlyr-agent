"""评估报告格式化 — Markdown 表格 + JSON 持久化.

Usage::

    from agenttrainer.eval.report import format_evaluation_report, save_report

    report = format_evaluation_report(results)
    print(report)

    save_report(results, "report.md")
    save_report(results, "report.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_evaluation_report(results: dict[str, Any]) -> str:
    """生成单 agent 评估的 Markdown 报告.

    Args:
        results: evaluate_agent() 返回的结果字典

    Returns:
        Markdown 格式字符串
    """
    lines = ["# Agent 评估报告", ""]

    # 总体指标
    lines.append("## 总体指标")
    lines.append("")

    sr = results.get("success_rate", 0.0)
    sr_ci = results.get("success_rate_ci", (sr, sr))
    lines.append(
        f"- **成功率**: {sr:.1%}"
        f" (95% CI: {sr_ci[0]:.1%} - {sr_ci[1]:.1%})"
    )

    rw = results.get("avg_reward", 0.0)
    rw_ci = results.get("reward_ci", (rw, rw))
    rw_std = results.get("std_reward", 0.0)
    lines.append(
        f"- **平均 Reward**: {rw:.3f} \u00b1 {rw_std:.3f}"
        f" (95% CI: {rw_ci[0]:.3f} - {rw_ci[1]:.3f})"
    )

    lines.append(
        f"- **Reward 范围**: "
        f"[{results.get('min_reward', 0.0):.3f}, {results.get('max_reward', 0.0):.3f}]"
    )

    st = results.get("avg_steps", 0.0)
    st_ci = results.get("steps_ci", (st, st))
    st_std = results.get("std_steps", 0.0)
    lines.append(
        f"- **平均步数**: {st:.1f} \u00b1 {st_std:.1f}"
        f" (95% CI: {st_ci[0]:.1f} - {st_ci[1]:.1f})"
    )

    lines.append(f"- **评估轮数**: {results.get('n_episodes', 0)}")
    lines.append("")

    # Reward 分布
    dist = results.get("reward_distribution", {})
    if dist:
        lines.append("## Reward 分布")
        lines.append("")
        lines.append("| 区间 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        total = results.get("n_episodes", 1) or 1
        for bracket, count in dist.items():
            pct = count / total
            lines.append(f"| {bracket} | {count} | {pct:.1%} |")
        lines.append("")

    return "\n".join(lines)


def format_comparison_report(results: dict[str, dict[str, Any]]) -> str:
    """生成多 agent 对比的 Markdown 报告.

    Args:
        results: compare_agents() 返回的结果字典

    Returns:
        Markdown 格式字符串，含排行榜、详细指标、显著性检验表
    """
    lines = ["# Agent 对比报告", ""]

    # Leaderboard
    leaderboard = results.get("_leaderboard")
    if leaderboard:
        lines.append("## 排行榜")
        lines.append("")
        lines.append("| 排名 | Agent | 成功率 | Avg Reward | Avg Steps |")
        lines.append("|------|-------|--------|------------|-----------|")
        for entry in leaderboard:
            lines.append(
                f"| {entry['rank']} | {entry['agent']} | "
                f"{entry['success_rate']:.1%} | "
                f"{entry['avg_reward']:.3f} | "
                f"{entry['avg_steps']:.1f} |"
            )
        lines.append("")

    # 详细指标
    agent_names = [k for k in results if not k.startswith("_")]
    if agent_names:
        lines.append("## 详细指标")
        lines.append("")
        lines.append("| Agent | 成功率 (CI) | Avg Reward (CI) | Avg Steps |")
        lines.append("|-------|-------------|-----------------|-----------|")
        for name in agent_names:
            r = results[name]
            sr = r.get("success_rate", 0.0)
            sr_ci = r.get("success_rate_ci", (sr, sr))
            rw = r.get("avg_reward", 0.0)
            rw_ci = r.get("reward_ci", (rw, rw))
            st = r.get("avg_steps", 0.0)
            lines.append(
                f"| {name} | {sr:.1%} ({sr_ci[0]:.1%}-{sr_ci[1]:.1%}) | "
                f"{rw:.3f} ({rw_ci[0]:.3f}-{rw_ci[1]:.3f}) | {st:.1f} |"
            )
        lines.append("")

    # 显著性检验
    comparisons = results.get("_comparisons")
    if comparisons:
        lines.append("## 两两对比 (Welch's t-test)")
        lines.append("")
        lines.append("| 对比 | t 统计量 | p 值 | 显著 | 效应量 |")
        lines.append("|------|----------|------|------|--------|")
        for key, comp in comparisons.items():
            sig = "\u2713" if comp.get("significant") else "\u2717"
            lines.append(
                f"| {key} | {comp.get('t_statistic', 0):.3f} | "
                f"{comp.get('p_approx', '')} | {sig} | "
                f"{comp.get('effect_size', 0):.3f} |"
            )
        lines.append("")

    # Bonferroni 校正
    corrected = results.get("_corrected")
    if corrected and comparisons:
        lines.append("## Bonferroni 校正")
        lines.append("")
        lines.append("| 对比 | 原始显著 | 校正后显著 |")
        lines.append("|------|----------|------------|")
        for key, corr_sig in corrected.items():
            orig_sig = comparisons.get(key, {}).get("significant", False)
            orig = "\u2713" if orig_sig else "\u2717"
            corr = "\u2713" if corr_sig else "\u2717"
            lines.append(f"| {key} | {orig} | {corr} |")
        lines.append("")

    return "\n".join(lines)


def save_report(
    results: dict[str, Any],
    output_path: str,
    *,
    is_comparison: bool = False,
) -> None:
    """保存评估报告到文件.

    Args:
        results: evaluate_agent 或 compare_agents 结果
        output_path: 输出路径 (.md 或 .json)
        is_comparison: 是否为多 agent 对比

    Raises:
        ValueError: 不支持的文件格式
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        # 移除 episodes 详情以减小体积
        clean: dict[str, Any] = {}
        for k, v in results.items():
            if isinstance(v, dict) and not k.startswith("_") and "episodes" in v:
                clean[k] = {kk: vv for kk, vv in v.items() if kk != "episodes"}
            elif k == "episodes":
                continue  # 单 agent 模式跳过 episodes
            else:
                clean[k] = v
        path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
    elif path.suffix == ".md":
        if is_comparison:
            report = format_comparison_report(results)
        else:
            report = format_evaluation_report(results)
        path.write_text(report, encoding="utf-8")
    else:
        raise ValueError(f"不支持的输出格式: {path.suffix} (仅支持 .md 或 .json)")
