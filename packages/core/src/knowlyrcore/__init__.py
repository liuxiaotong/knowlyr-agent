"""knowlyr-core — 共享数据模型、领域配置与环境协议

提供 knowlyr 生态各子包共用的基础数据类型、领域 Profile、
Gymnasium 风格 AgentEnv 协议和环境注册表。
"""

__version__ = "0.1.0"

from knowlyrcore.domain import (
    BROWSER_PROFILE,
    CODING_PROFILE,
    GENERIC_PROFILE,
    DomainProfile,
    OutcomeSpec,
    ToolCategory,
    ToolSpec,
    get_domain_profile,
    list_domain_profiles,
    load_domain_profile,
)
from knowlyrcore.env import AgentEnv, EnvWrapper
from knowlyrcore.models import TaskInfo, ToolResult
from knowlyrcore.registry import (
    EnvSpec,
    list_envs,
    make,
    register,
    spec,
)
from knowlyrcore.timestep import TimeStep

__all__ = [
    # 数据模型
    "TaskInfo",
    "ToolResult",
    # 领域配置
    "DomainProfile",
    "ToolCategory",
    "ToolSpec",
    "OutcomeSpec",
    "CODING_PROFILE",
    "BROWSER_PROFILE",
    "GENERIC_PROFILE",
    "get_domain_profile",
    "load_domain_profile",
    "list_domain_profiles",
    # 环境协议
    "AgentEnv",
    "EnvWrapper",
    "TimeStep",
    # 注册表
    "EnvSpec",
    "register",
    "make",
    "list_envs",
    "spec",
    # 版本
    "__version__",
]
