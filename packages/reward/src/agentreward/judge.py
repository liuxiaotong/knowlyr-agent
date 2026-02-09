"""LLM-as-Judge scoring for agent trajectory steps.

Uses an LLM to evaluate each step against model-based rubrics,
providing fine-grained scores and rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentreward.rubrics import Rubric


# --- Prompt Template ---

STEP_JUDGE_PROMPT = """你是一个 Agent 轨迹评估专家。请根据以下评估维度，对 Agent 的这一步操作进行打分。

## 任务描述
{task_description}

## 当前步骤 (Step {step_index}/{total_steps})
- 工具: {tool_name}
- 参数: {tool_params}
- 输出: {tool_output}

## 上下文 (前序步骤摘要)
{context_summary}

## 评估维度
{rubric_descriptions}

## 评分要求
请对每个评估维度给出 0.0-1.0 的分数，并给出简短理由。

输出格式 (JSON):
{{
  "scores": {{
    "<rubric_id>": <score>,
    ...
  }},
  "rationale": "简要说明这一步的整体评价",
  "overall_score": <加权总分>
}}

注意:
- 0.0 = 完全不满足
- 0.5 = 部分满足
- 1.0 = 完全满足
- overall_score 应该是各维度分数的加权平均
"""


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""

    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.1
    max_retries: int = 3


@dataclass
class StepJudgment:
    """Result of judging a single step."""

    score: float  # Overall weighted score for this step
    rationale: str  # Explanation of the judgment
    rubric_scores: dict[str, float] = field(default_factory=dict)  # Per-rubric scores


def build_judge_prompt(
    step: dict[str, Any],
    step_index: int,
    total_steps: int,
    context_summary: str,
    rubrics: list[Rubric],
    task_description: str = "",
) -> str:
    """Build the prompt for judging a single step.

    Args:
        step: Step dict with tool/params/output
        step_index: 1-based index of this step
        total_steps: Total number of steps
        context_summary: Summary of previous steps
        rubrics: List of rubrics to evaluate against
        task_description: Description of the overall task

    Returns:
        Formatted prompt string
    """
    rubric_lines = []
    for r in rubrics:
        rubric_lines.append(f"- {r.id} ({r.name}, weight={r.weight}): {r.description}")
    rubric_descriptions = "\n".join(rubric_lines)

    tool_params = step.get("params", {})
    tool_output = step.get("output", "")
    # Truncate long outputs to stay within context limits
    if len(str(tool_output)) > 2000:
        tool_output = str(tool_output)[:2000] + "\n... (truncated)"

    return STEP_JUDGE_PROMPT.format(
        task_description=task_description or "(未提供任务描述)",
        step_index=step_index,
        total_steps=total_steps,
        tool_name=step.get("tool", "unknown"),
        tool_params=str(tool_params),
        tool_output=tool_output,
        context_summary=context_summary or "(第一步，无前序上下文)",
        rubric_descriptions=rubric_descriptions,
    )


def judge_step(
    step: dict[str, Any],
    step_index: int,
    total_steps: int,
    context_summary: str,
    rubrics: list[Rubric],
    config: JudgeConfig | None = None,
    task_description: str = "",
) -> StepJudgment:
    """Judge a single step using LLM-as-Judge.

    Currently returns a stub result. When LLM dependencies are available,
    this will call the LLM with the constructed prompt.

    Args:
        step: Step dict with tool/params/output
        step_index: 1-based index of this step
        total_steps: Total number of steps
        context_summary: Summary of previous steps
        rubrics: List of rubrics to evaluate against
        config: Judge configuration
        task_description: Description of the overall task

    Returns:
        StepJudgment with scores and rationale
    """
    config = config or JudgeConfig()

    # Build the prompt (shows the template for inspection)
    _prompt = build_judge_prompt(
        step=step,
        step_index=step_index,
        total_steps=total_steps,
        context_summary=context_summary,
        rubrics=rubrics,
        task_description=task_description,
    )

    # --- Stub: LLM call would go here ---
    # In production, this would:
    # 1. Call the LLM with _prompt
    # 2. Parse the JSON response
    # 3. Return StepJudgment with actual scores
    #
    # For now, return neutral scores
    rubric_scores = {r.id: 0.5 for r in rubrics}
    total = sum(r.weight * rubric_scores[r.id] for r in rubrics)
    weight_sum = sum(r.weight for r in rubrics)
    overall = total / weight_sum if weight_sum > 0 else 0.5

    return StepJudgment(
        score=overall,
        rationale="(LLM judge not yet connected - returning neutral scores)",
        rubric_scores=rubric_scores,
    )


def judge_trajectory(
    trajectory: dict[str, Any],
    rubrics: list[Rubric],
    config: JudgeConfig | None = None,
) -> list[StepJudgment]:
    """Judge all steps in a trajectory using LLM-as-Judge.

    Args:
        trajectory: Dict with:
            - task: str (task description)
            - steps: list[dict] (list of steps)
        rubrics: List of model-based rubrics
        config: Judge configuration

    Returns:
        List of StepJudgment, one per step
    """
    config = config or JudgeConfig()
    steps = trajectory.get("steps", [])
    task_description = trajectory.get("task", "")
    total_steps = len(steps)
    judgments = []

    context_parts: list[str] = []

    for i, step in enumerate(steps):
        context_summary = "\n".join(context_parts[-5:]) if context_parts else ""

        judgment = judge_step(
            step=step,
            step_index=i + 1,
            total_steps=total_steps,
            context_summary=context_summary,
            rubrics=rubrics,
            config=config,
            task_description=task_description,
        )
        judgments.append(judgment)

        # Build context for next step
        tool = step.get("tool", "unknown")
        context_parts.append(
            f"Step {i + 1}: {tool} -> score={judgment.score:.2f}"
        )

    return judgments
