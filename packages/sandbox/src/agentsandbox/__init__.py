"""AgentSandbox - Code Agent 执行沙箱

提供可复现的 Docker 执行环境，支持代码任务的隔离执行与轨迹重放。
"""

__version__ = "0.1.0"

from agentsandbox.sandbox import Sandbox, SandboxConfig
from agentsandbox.tools import ToolResult

__all__ = ["Sandbox", "SandboxConfig", "ToolResult", "__version__"]
