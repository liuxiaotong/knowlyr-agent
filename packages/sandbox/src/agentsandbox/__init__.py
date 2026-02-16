"""AgentSandbox - Code Agent 执行沙箱

提供可复现的 Docker 执行环境，支持代码任务的隔离执行与轨迹重放。
同时包含对话类环境 (ConversationEnv)，支持 conversation/engineering/advisory 领域。
"""

import logging

__version__ = "0.1.0"

from agentsandbox.sandbox import Sandbox, SandboxConfig
from agentsandbox.env import SandboxEnv
from agentsandbox.conversation_env import ConversationEnv
from agentsandbox.tools import ToolResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "SandboxEnv",
    "ConversationEnv",
    "ToolResult",
    "__version__",
]


def _auto_register_envs() -> None:
    """自动注册内置环境到 knowlyrcore Registry.

    ConversationEnv 需要 domain 参数传给构造函数，但 register() 的 domain
    参数只写入 EnvSpec 元数据。通过 make() 时传 override_kwargs 或直接
    修改 EnvSpec.kwargs 解决。这里用一个小技巧：先注册，再手动设置 kwargs。
    """
    from knowlyrcore.registry import register, spec

    try:
        register(
            "knowlyr/sandbox", SandboxEnv,
            domain="coding", description="Docker 代码沙箱环境",
        )
    except ValueError:
        pass

    _conv_envs = [
        ("knowlyr/conversation", "conversation", "对话助手环境"),
        ("knowlyr/engineering", "engineering", "工程审查环境"),
        ("knowlyr/advisory", "advisory", "专业咨询环境"),
        ("knowlyr/discussion", "discussion", "讨论参与环境"),
    ]
    for env_id, env_domain, desc in _conv_envs:
        try:
            register(env_id, ConversationEnv, domain=env_domain, description=desc)
            # register 的 **kwargs 不含 domain（被 register 消费了）
            # 手动将 domain 添加到 EnvSpec.kwargs，make() 时会传给构造函数
            env_spec = spec(env_id)
            if env_spec is not None:
                env_spec.kwargs["domain"] = env_domain
        except ValueError:
            pass


try:
    _auto_register_envs()
except Exception:
    pass  # knowlyr-core 未安装时静默跳过
