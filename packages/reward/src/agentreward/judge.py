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
- rationale 限制在 100 字以内

重要：只输出 JSON，不要输出其他任何文字。
"""

CONVERSATION_JUDGE_PROMPT = """你是一个 AI 对话质量评估专家。请根据以下评估维度，对 AI 助手的这一步操作进行打分。

## 用户请求
{task_description}

## 当前步骤 (Step {step_index}/{total_steps})
- 操作类型: {tool_name}
- 参数: {tool_params}
- 回复/输出:
{tool_output}

## 对话上下文
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

评分标准:
- 0.0 = 完全不满足，回复无用或有害
- 0.3 = 基本没满足，回复方向错误或严重缺失
- 0.5 = 部分满足，但有明显不足
- 0.7 = 基本满足，小有瑕疵
- 1.0 = 完全满足，回复优秀
- 重点关注: 回复对用户的实际帮助程度，而非工具调用的技术细节
- rationale 限制在 100 字以内

重要：只输出 JSON，不要输出其他任何文字。
"""

ENGINEERING_JUDGE_PROMPT = """你是一个工程能力评估专家。请根据以下评估维度，对 AI 工程师的这一步操作进行打分。

## 任务描述
{task_description}

## 当前步骤 (Step {step_index}/{total_steps})
- 操作类型: {tool_name}
- 参数: {tool_params}
- 输出:
{tool_output}

## 上下文
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

评分标准:
- 0.0 = 完全不满足
- 0.3 = 有明显技术错误或遗漏
- 0.5 = 基本正确但不够深入
- 0.7 = 正确且较全面
- 1.0 = 技术精准、分析深入
- 重点关注: 技术分析的准确性和工具使用的合理性
- rationale 限制在 100 字以内

重要：只输出 JSON，不要输出其他任何文字。
"""

ADVISORY_JUDGE_PROMPT = """你是一个专业顾问能力评估专家。请根据以下评估维度，对 AI 顾问的分析/建议进行打分。

## 用户请求
{task_description}

## 当前步骤 (Step {step_index}/{total_steps})
- 操作类型: {tool_name}
- 参数: {tool_params}
- 输出:
{tool_output}

## 上下文
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

评分标准:
- 0.0 = 分析空泛、建议无用
- 0.3 = 分析浅显、建议不具体
- 0.5 = 分析基本合理但缺少深度或证据
- 0.7 = 分析深入、建议可执行
- 1.0 = 洞察独到、建议精准可行
- 重点关注: 分析深度、建议的可操作性、是否有证据支撑
- rationale 限制在 100 字以内

重要：只输出 JSON，不要输出其他任何文字。
"""

DISCUSSION_JUDGE_PROMPT = """你是一个讨论质量评估专家。请根据以下评估维度，对 AI 参与者在讨论中的发言进行打分。

## 讨论主题
{task_description}

## 当前发言 (Step {step_index}/{total_steps})
- 操作类型: {tool_name}
- 参数: {tool_params}
- 发言内容:
{tool_output}

## 讨论上下文 (前序发言摘要)
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
  "rationale": "简要说明这位参与者的贡献质量",
  "overall_score": <加权总分>
}}

评分标准:
- 0.0 = 无实质贡献，只是附和或重复
- 0.3 = 有观点但浅显，未推动讨论
- 0.5 = 有一定贡献，但缺少专业深度或建设性
- 0.7 = 提供了有价值的专业观点或推动了结论
- 1.0 = 发言质量很高，提供了新视角或关键推进
- 重点关注: 是否提供了新信息、是否回应了他人、是否推动了讨论
- rationale 限制在 100 字以内

重要：只输出 JSON，不要输出其他任何文字。
"""


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""

    model: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.1
    max_retries: int = 3
    base_url: str | None = None
    api_key: str | None = None
    domain: str = "coding"


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
        max_tokens=2048,
        temperature=config.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str, config: JudgeConfig) -> str:
    """通过 OpenAI 兼容 API 调用 LLM（支持 moonshot 等）."""
    kwargs: dict[str, Any] = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=2048,
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

    支持纯 JSON、markdown 代码块包裹、混合文本中的 JSON、以及截断修复。
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
            # 代码块内的 JSON 可能被截断，尝试修复
            result = _try_fix_truncated(match.group(1).strip())
            if result is not None:
                return result

    # 尝试找第一个 { ... } 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            result = _try_fix_truncated(match.group(0))
            if result is not None:
                return result

    # 最后尝试：找第一个 { 开头，修复截断
    match = re.search(r"\{", text)
    if match:
        result = _try_fix_truncated(text[match.start():])
        if result is not None:
            return result

    raise ValueError(f"无法从 LLM 响应中提取 JSON: {text[:200]}...")


def _try_fix_truncated(text: str) -> dict[str, Any] | None:
    """尝试修复被 max_tokens 截断的 JSON.

    策略：截断 rationale 字符串，补齐缺失的括号。
    """
    # 如果 rationale 被截断（常见模式：值字符串没闭合）
    # 尝试在最后一个完整的 key-value 对后截断，补齐括号
    for trim in range(min(len(text), 500)):
        candidate = text[:len(text) - trim]
        # 补齐缺失的引号和括号
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        # 如果在字符串中间截断，先闭合字符串
        in_string = candidate.count('"') % 2 == 1
        if in_string:
            candidate += '"'
        candidate += "]" * max(0, open_brackets)
        candidate += "}" * max(0, open_braces)
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "scores" in result:
                logger.debug("修复截断 JSON 成功 (trimmed %d chars)", trim)
                return result
        except json.JSONDecodeError:
            continue
    return None


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
    domain: str = "coding",
) -> str:
    """Build the prompt for judging a single step.

    Args:
        step: Step dict with tool/params/output
        step_index: 1-based index of this step
        total_steps: Total number of steps
        context_summary: Summary of previous steps
        rubrics: List of rubrics to evaluate against
        task_description: Description of the overall task
        domain: 领域标识，conversation 使用对话专用 prompt

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

    # 选择 prompt 模板
    _domain_template_map = {
        "conversation": CONVERSATION_JUDGE_PROMPT,
        "engineering": ENGINEERING_JUDGE_PROMPT,
        "advisory": ADVISORY_JUDGE_PROMPT,
        "discussion": DISCUSSION_JUDGE_PROMPT,
    }
    template = _domain_template_map.get(domain, STEP_JUDGE_PROMPT)

    return template.format(
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
        domain=config.domain,
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
