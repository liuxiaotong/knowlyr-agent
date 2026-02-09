"""knowlyr-core — 共享数据模型

提供 knowlyr 生态各子包共用的基础数据类型。
"""

__version__ = "0.1.0"

from knowlyrcore.models import TaskInfo, ToolResult

__all__ = ["TaskInfo", "ToolResult", "__version__"]
