"""Agent 框架适配器."""

from agentrecorder.adapters.base import BaseAdapter
from agentrecorder.adapters.openhands import OpenHandsAdapter
from agentrecorder.adapters.sweagent import SWEAgentAdapter

__all__ = ["BaseAdapter", "OpenHandsAdapter", "SWEAgentAdapter"]
