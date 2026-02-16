"""Reward configuration."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward computation.

    Attributes:
        rule_weight: Weight for rule-based reward scores (0.0-1.0)
        model_weight: Weight for model-based (LLM-as-Judge) scores (0.0-1.0)
        rubric_set: Name of the rubric set to use (default: "default")
        model_name: LLM model for judge scoring
        provider: LLM provider (anthropic, openai)
        temperature: Sampling temperature for judge calls
        max_retries: Maximum retries on LLM failure
        domain: Domain identifier (coding/conversation/engineering/advisory)
        base_url: Custom API base URL (for proxy or self-hosted models)
        api_key: API key override (default: from environment variable)

    典型配置::

        # 纯规则（快速，无网络调用）
        RewardConfig(rule_weight=1.0, model_weight=0.0)

        # 规则主导（推荐，平衡速度与质量）
        RewardConfig(rule_weight=0.7, model_weight=0.3, model_name="gpt-4o-mini")

        # 模型主导（语义评分为主，更准但慢）
        RewardConfig(rule_weight=0.3, model_weight=0.7, model_name="claude-sonnet-4-20250514")
    """

    rule_weight: float = 0.6
    model_weight: float = 0.4
    rubric_set: str = "default"
    model_name: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.1
    max_retries: int = 3
    domain: str = "coding"
    base_url: str | None = None
    api_key: str | None = None

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.rule_weight + self.model_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"rule_weight ({self.rule_weight}) + model_weight ({self.model_weight}) "
                f"= {total}, must sum to 1.0"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "rule_weight": self.rule_weight,
            "model_weight": self.model_weight,
            "rubric_set": self.rubric_set,
            "model_name": self.model_name,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "domain": self.domain,
        }
