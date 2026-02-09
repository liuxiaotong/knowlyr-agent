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
