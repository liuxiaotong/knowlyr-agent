"""LLM-as-Judge scoring for agent trajectory steps.

Uses an LLM to evaluate each step against model-based rubrics,
providing fine-grained scores and rationale.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agentreward.rubrics import Rubric

logger = logging.getLogger(__name__)

# ── 可选 LLM 依赖 ──────────────────────────────────────────────────

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


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


# ── LLM 调用 ───────────────────────────────────────────────────────


def _call_anthropic(prompt: str, config: JudgeConfig) -> str:
    """通过 Anthropic API 调用 LLM."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.model,
        max_tokens=1024,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str, config: JudgeConfig) -> str:
    """通过 OpenAI API 调用 LLM."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=1024,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _call_llm(prompt: str, config: JudgeConfig) -> str:
    """根据 provider 选择对应的 LLM 调用.

    Args:
        prompt: 完整的评估提示词。
        config: Judge 配置。

    Returns:
        LLM 返回的原始文本。

    Raises:
        RuntimeError: provider 对应的库未安装或不支持。
    """
    if config.provider == "anthropic":
        if not _HAS_ANTHROPIC:
            raise RuntimeError(
                "Anthropic provider 需要安装 anthropic 库: "
                "pip install knowlyr-reward[llm]"
            )
        return _call_anthropic(prompt, config)
    elif config.provider == "openai":
        if not _HAS_OPENAI:
            raise RuntimeError(
                "OpenAI provider 需要安装 openai 库: "
                "pip install knowlyr-reward[llm]"
            )
        return _call_openai(prompt, config)
    else:
        raise RuntimeError(f"不支持的 provider: {config.provider}，支持: anthropic, openai")


def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 返回的文本中提取 JSON.

    支持纯 JSON、markdown 代码块包裹、以及混合文本中的 JSON。
    """
    # 尝试直接解析
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 ```json ... ``` 代码块提取
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找第一个 { ... } 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 响应中提取 JSON: {text[:200]}...")


def _parse_judgment(
    response_text: str,
    rubrics: list[Rubric],
) -> StepJudgment:
    """将 LLM 返回的 JSON 解析为 StepJudgment.

    Args:
        response_text: LLM 的原始响应文本。
        rubrics: 评估维度列表（用于校验 scores key 和计算加权分）。

    Returns:
        StepJudgment。
    """
    data = _extract_json(response_text)

    scores_raw = data.get("scores", {})
    rationale = data.get("rationale", "")

    # 校验并规范化每个 rubric 的分数
    rubric_scores: dict[str, float] = {}
    for r in rubrics:
        raw = scores_raw.get(r.id)
        if raw is not None:
            score = max(0.0, min(1.0, float(raw)))
        else:
            # LLM 漏给的 rubric 用 0.5 填充
            score = 0.5
            logger.warning("LLM 响应缺少 rubric '%s' 的分数，用 0.5 填充", r.id)
        rubric_scores[r.id] = score

    # 计算加权总分
    weighted_sum = sum(r.weight * rubric_scores[r.id] for r in rubrics)
    weight_sum = sum(r.weight for r in rubrics)
    overall = weighted_sum / weight_sum if weight_sum > 0 else 0.5

    return StepJudgment(
        score=overall,
        rationale=rationale,
        rubric_scores=rubric_scores,
    )


def _fallback_judgment(rubrics: list[Rubric], reason: str) -> StepJudgment:
    """LLM 调用失败时的降级结果."""
    rubric_scores = {r.id: 0.5 for r in rubrics}
    total = sum(r.weight * rubric_scores[r.id] for r in rubrics)
    weight_sum = sum(r.weight for r in rubrics)
    overall = total / weight_sum if weight_sum > 0 else 0.5

    return StepJudgment(
        score=overall,
        rationale=f"(降级: {reason})",
        rubric_scores=rubric_scores,
    )


# ── 公开 API ───────────────────────────────────────────────────────


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

    当 LLM 库可用时调用真实 LLM；不可用时返回中性分数 (0.5)。
    支持重试和降级。

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

    prompt = build_judge_prompt(
        step=step,
        step_index=step_index,
        total_steps=total_steps,
        context_summary=context_summary,
        rubrics=rubrics,
        task_description=task_description,
    )

    # 检查 LLM 是否可用
    has_llm = (
        (config.provider == "anthropic" and _HAS_ANTHROPIC)
        or (config.provider == "openai" and _HAS_OPENAI)
    )
    if not has_llm:
        logger.debug("LLM 库不可用 (provider=%s)，返回中性分数", config.provider)
        return _fallback_judgment(rubrics, "LLM 库未安装")

    # 带重试的 LLM 调用
    last_error = None
    for attempt in range(config.max_retries):
        try:
            response_text = _call_llm(prompt, config)
            return _parse_judgment(response_text, rubrics)
        except Exception as e:
            last_error = e
            logger.warning(
                "LLM judge 调用失败 (attempt %d/%d): %s",
                attempt + 1, config.max_retries, e,
            )

    # 所有重试失败，降级
    logger.error("LLM judge 全部重试失败，降级为中性分数: %s", last_error)
    return _fallback_judgment(rubrics, f"LLM 调用失败: {last_error}")


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
    logger.info("LLM Judge: %d 步, %d 个 rubric, model=%s",
                total_steps, len(rubrics), config.model)

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
