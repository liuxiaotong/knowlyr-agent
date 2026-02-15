"""Rubric definitions for agent trajectory evaluation."""

from typing import Literal

from pydantic import BaseModel, Field


class Rubric(BaseModel):
    """A single evaluation rubric dimension.

    Attributes:
        id: Unique identifier for the rubric
        name: Display name
        description: What this rubric evaluates (used in LLM prompts)
        weight: Relative weight in final score (0.0-1.0)
        evaluator: Which layer evaluates this rubric
    """

    id: str
    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    evaluator: Literal["rule", "model", "human"] = "rule"


class RubricSet(BaseModel):
    """A collection of rubrics for trajectory evaluation.

    Weights across all rubrics should sum to 1.0.
    """

    rubrics: list[Rubric]

    def get_rule_rubrics(self) -> list[Rubric]:
        """Return rubrics evaluated by rule-based functions."""
        return [r for r in self.rubrics if r.evaluator == "rule"]

    def get_model_rubrics(self) -> list[Rubric]:
        """Return rubrics evaluated by LLM-as-Judge."""
        return [r for r in self.rubrics if r.evaluator == "model"]

    def get_human_rubrics(self) -> list[Rubric]:
        """Return rubrics evaluated by human annotators."""
        return [r for r in self.rubrics if r.evaluator == "human"]

    def get_by_id(self, rubric_id: str) -> Rubric | None:
        """Get a rubric by its ID."""
        for r in self.rubrics:
            if r.id == rubric_id:
                return r
        return None

    def total_weight(self) -> float:
        """Return sum of all rubric weights."""
        return sum(r.weight for r in self.rubrics)

    def to_prompt_description(self) -> str:
        """Convert rubric set to natural language for LLM prompting."""
        lines = ["评估维度 (Evaluation Rubrics):"]
        for r in self.rubrics:
            lines.append(f"- {r.id} ({r.name}, weight={r.weight}): {r.description}")
        return "\n".join(lines)


def get_default_rubric_set() -> RubricSet:
    """Return the default rubric set for code agent evaluation."""
    return RubricSet(
        rubrics=[
            Rubric(
                id="goal_progress",
                name="目标推进",
                description="这一步是否推进了任务目标？",
                weight=0.3,
                evaluator="model",
            ),
            Rubric(
                id="tool_choice",
                name="工具选择",
                description="选择的工具是否合理？",
                weight=0.2,
                evaluator="model",
            ),
            Rubric(
                id="param_correctness",
                name="参数正确性",
                description="工具调用的参数是否正确？",
                weight=0.2,
                evaluator="model",
            ),
            Rubric(
                id="info_utilization",
                name="信息利用",
                description="是否利用了之前获得的信息？",
                weight=0.15,
                evaluator="rule",
            ),
            Rubric(
                id="non_redundancy",
                name="非冗余性",
                description="这一步是否是非冗余操作？",
                weight=0.15,
                evaluator="rule",
            ),
        ]
    )


def get_conversation_rubric_set() -> RubricSet:
    """Return rubric set for conversation/dialogue agent evaluation.

    适用于对话类 AI 员工（如 CEO 助理），评估维度侧重回复质量而非工具使用。
    """
    return RubricSet(
        rubrics=[
            Rubric(
                id="relevance",
                name="相关性",
                description="回复是否直接回应了用户的请求或问题？是否切题？",
                weight=0.25,
                evaluator="model",
            ),
            Rubric(
                id="completeness",
                name="完整性",
                description="回复是否覆盖了请求的所有方面？是否有重要遗漏？",
                weight=0.20,
                evaluator="model",
            ),
            Rubric(
                id="clarity",
                name="清晰度",
                description="回复是否条理清晰、结构合理、易于理解？",
                weight=0.20,
                evaluator="model",
            ),
            Rubric(
                id="actionability",
                name="可操作性",
                description="回复是否提供了具体、可操作的信息或建议？而非泛泛而谈？",
                weight=0.15,
                evaluator="model",
            ),
            Rubric(
                id="tone_fit",
                name="语气匹配",
                description="回复的语气和风格是否符合角色设定和场景？是否自然？",
                weight=0.10,
                evaluator="model",
            ),
            Rubric(
                id="non_redundancy",
                name="非冗余性",
                description="这一步操作是否非冗余？是否避免了重复调用？",
                weight=0.10,
                evaluator="rule",
            ),
        ]
    )


def get_rubric_set_for_domain(domain: str) -> RubricSet:
    """根据领域获取对应的 RubricSet.

    Args:
        domain: 领域标识 (coding / conversation / 其他)

    Returns:
        对应领域的 RubricSet
    """
    if domain == "conversation":
        return get_conversation_rubric_set()
    return get_default_rubric_set()
