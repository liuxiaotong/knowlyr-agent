"""环境注册表 — knowlyr.make() 发现机制.

借鉴 gymnasium.register() / gymnasium.make() 模式，
命名规范沿用 "namespace/env-name" 格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowlyrcore.env import AgentEnv


@dataclass
class EnvSpec:
    """环境规格 — 对应 gymnasium.EnvSpec.

    Attributes:
        id: 环境标识 (如 "knowlyr/sandbox", "swebench/django-11099")
        env_cls: 环境类
        kwargs: 默认构造参数
        domain: 领域标识
        description: 环境描述
    """

    id: str
    env_cls: type[AgentEnv]
    kwargs: dict[str, Any] = field(default_factory=dict)
    domain: str = "coding"
    description: str = ""


_REGISTRY: dict[str, EnvSpec] = {}


def register(
    id: str,
    env_cls: type[AgentEnv],
    *,
    domain: str = "coding",
    description: str = "",
    **kwargs: Any,
) -> None:
    """注册一个环境.

    Args:
        id: 环境标识 (如 "knowlyr/sandbox")
        env_cls: 环境类（必须继承 AgentEnv）
        domain: 领域标识
        description: 环境描述
        **kwargs: 默认构造参数

    Raises:
        ValueError: 环境 ID 已注册
    """
    if id in _REGISTRY:
        raise ValueError(f"环境已注册: {id!r}")
    _REGISTRY[id] = EnvSpec(
        id=id,
        env_cls=env_cls,
        kwargs=kwargs,
        domain=domain,
        description=description,
    )


def make(id: str, **override_kwargs: Any) -> AgentEnv:
    """创建环境实例 — 对应 gymnasium.make().

    Args:
        id: 环境标识
        **override_kwargs: 覆盖默认构造参数

    Returns:
        AgentEnv 实例

    Raises:
        KeyError: 环境未注册
    """
    if id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(空)"
        raise KeyError(f"未注册的环境: {id!r}。可用: {available}")
    env_spec = _REGISTRY[id]
    merged_kwargs = {**env_spec.kwargs, **override_kwargs}
    return env_spec.env_cls(**merged_kwargs)


def list_envs(domain: str | None = None) -> list[str]:
    """列出已注册的环境 ID.

    Args:
        domain: 按领域过滤（None 返回全部）

    Returns:
        环境 ID 列表
    """
    if domain is None:
        return sorted(_REGISTRY.keys())
    return sorted(k for k, v in _REGISTRY.items() if v.domain == domain)


def spec(id: str) -> EnvSpec | None:
    """获取环境规格.

    Args:
        id: 环境标识

    Returns:
        EnvSpec 或 None
    """
    return _REGISTRY.get(id)


def _clear_registry() -> None:
    """清空注册表（仅用于测试）."""
    _REGISTRY.clear()
