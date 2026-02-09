"""AgentRecorder - Agent 轨迹录制工具

将任意 Agent 框架的执行日志转换为标准化轨迹格式。
"""

__version__ = "0.1.0"

from agentrecorder.recorder import Recorder
from agentrecorder.schema import Trajectory, Step, ToolCall, ToolResult

__all__ = ["Recorder", "Trajectory", "Step", "ToolCall", "ToolResult", "__version__"]
