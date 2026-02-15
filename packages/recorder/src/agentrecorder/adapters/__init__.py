"""Agent 框架适配器 — 注册表模式.

通过 register_adapter() 注册自定义适配器，通过 get_adapter() 获取。
"""

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.adapters.crew import CrewAdapter
from agentrecorder.adapters.openhands import OpenHandsAdapter
from agentrecorder.adapters.sweagent import SWEAgentAdapter

_ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[BaseAdapter]) -> None:
    """注册一个适配器.

    Args:
        name: 适配器名称 (如 "openhands", "swe-agent")
        adapter_cls: 适配器类（必须继承 BaseAdapter）
    """
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> BaseAdapter | None:
    """按名称获取适配器实例.

    Args:
        name: 适配器名称

    Returns:
        适配器实例，未找到时返回 None
    """
    cls = _ADAPTER_REGISTRY.get(name)
    return cls() if cls else None


def list_adapters() -> list[str]:
    """列出所有已注册的适配器名称."""
    return list(_ADAPTER_REGISTRY.keys())


# 注册内置适配器
register_adapter("openhands", OpenHandsAdapter)
register_adapter("swe-agent", SWEAgentAdapter)
register_adapter("sweagent", SWEAgentAdapter)  # 别名
register_adapter("crew", CrewAdapter)

__all__ = [
    "BaseAdapter",
    "CrewAdapter",
    "OpenHandsAdapter",
    "SWEAgentAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
