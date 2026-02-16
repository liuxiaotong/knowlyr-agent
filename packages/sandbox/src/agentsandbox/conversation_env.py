"""ConversationEnv — 对话类 Agent 环境.

支持 conversation / engineering / advisory 三类领域。
不需要 Docker 隔离，环境状态 = 用户消息 + 对话历史。

Usage::

    from agentsandbox.conversation_env import ConversationEnv

    env = ConversationEnv(domain="conversation")
    ts = env.reset(task={"user_message": "你们的产品多少钱?"})
    while not ts.done:
        action = {"tool": "respond", "params": {"message": "产品定价..."}}
        ts = env.step(action)
    env.close()
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from knowlyrcore.domain import DomainProfile, get_domain_profile
from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

logger = logging.getLogger(__name__)

# 各领域的终止工具
_TERMINAL_TOOLS: dict[str, set[str]] = {
    "conversation": {"respond"},
    "engineering": {"submit", "finish"},
    "advisory": {"submit", "finish"},
    "discussion": {"respond", "submit"},
}


class ConversationEnv(AgentEnv):
    """对话类 Agent 环境.

    conversation/engineering/advisory 三类领域共享此环境。
    区别在于可用工具集和终止条件:

    - **conversation**: ``respond`` 即完成 (单轮问答)
    - **engineering**: ``submit`` 才完成 (需阅读代码、分析后提交)
    - **advisory**: ``submit`` 才完成 (需查资料、分析后提交建议)

    支持外部注入真实工具 (``tools`` 参数)，不传时使用模拟工具。

    Attributes:
        domain: 环境领域标识
        metadata: 环境元信息
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        domain: str = "conversation",
        tools: dict[str, Callable[..., dict[str, Any]]] | None = None,
        max_turns: int = 20,
    ):
        """初始化.

        Args:
            domain: 领域标识 (conversation/engineering/advisory/discussion)
            tools: 可用工具字典 {tool_name: tool_fn}。
                   tool_fn 签名: (params: dict) -> {"output": str, "exit_code": int}。
                   为 None 时使用默认模拟工具。
            max_turns: 最大对话轮数
        """
        self.domain = domain
        self._profile: DomainProfile = get_domain_profile(domain)
        self._external_tools = tools
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self._max_turns = max_turns
        self._terminal_tools = _TERMINAL_TOOLS.get(domain, {"submit", "finish"})
        self._turn_count = 0
        self._history: list[dict[str, Any]] = []
        self._task: Any = None

    def reset(self, *, task: Any | None = None, seed: int | None = None) -> TimeStep:
        """重置对话环境.

        Args:
            task: 任务信息。兼容多种格式:
                  - dict: {"user_message": "...", "description": "..."}
                  - 对象: task.user_message / task.description / task.context
                  - str: 直接作为用户消息
            seed: 未使用（保留接口兼容）

        Returns:
            初始 TimeStep，observation = 用户消息
        """
        self._turn_count = 0
        self._history = []
        self._task = task

        # 初始化工具集
        self._tools = (
            dict(self._external_tools)
            if self._external_tools is not None
            else self._build_default_tools()
        )

        # 从 task 提取用户消息
        user_message = self._extract_user_message(task)

        logger.info(
            "ConversationEnv(%s) 就绪: max_turns=%d, tools=%s",
            self.domain, self._max_turns, list(self._tools.keys()),
        )
        return TimeStep(
            observation=user_message,
            info={
                "domain": self.domain,
                "max_turns": self._max_turns,
                "available_tools": list(self._tools.keys()),
            },
        )

    def step(self, action: dict[str, Any]) -> TimeStep:
        """执行一步工具调用.

        Args:
            action: {"tool": "respond", "params": {"message": "..."}}

        Returns:
            TimeStep
        """
        tool = action.get("tool", "")
        params = action.get("params", {})

        # 执行工具
        if tool in self._tools:
            result = self._tools[tool](params)
        else:
            result = {
                "output": f"未知工具: {tool!r}。可用: {', '.join(self._tools.keys())}",
                "exit_code": 1,
            }

        self._turn_count += 1
        exit_code = result.get("exit_code", 0)

        # 记录历史
        self._history.append({
            "step": self._turn_count,
            "tool": tool,
            "params": params,
            "output": result.get("output", ""),
            "exit_code": exit_code,
        })

        # 终止判断
        terminated = tool in self._terminal_tools
        truncated = self._turn_count >= self._max_turns and not terminated

        return TimeStep(
            observation=result.get("output", ""),
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info={
                "exit_code": exit_code,
                "step": self._turn_count,
                "tool": tool,
            },
        )

    def close(self) -> None:
        """清理资源（对话环境无需特殊清理）."""
        self._history = []
        self._tools = {}
        logger.info("ConversationEnv(%s) 已关闭", self.domain)

    @property
    def available_tools(self) -> list[str]:
        """返回可用工具列表."""
        return list(self._tools.keys())

    @property
    def history(self) -> list[dict[str, Any]]:
        """当前 episode 的执行历史."""
        return list(self._history)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_user_message(self, task: Any) -> str:
        """从 task 提取用户消息，兼容多种格式."""
        if task is None:
            return ""
        if isinstance(task, str):
            return task
        if isinstance(task, dict):
            return (
                task.get("user_message")
                or task.get("context")
                or task.get("description")
                or ""
            )
        # 对象属性
        return (
            getattr(task, "user_message", None)
            or getattr(task, "context", None)
            or getattr(task, "description", None)
            or ""
        )

    def _build_default_tools(self) -> dict[str, Callable[..., dict[str, Any]]]:
        """构建默认模拟工具 (用于测试，实际部署应注入真实工具)."""
        tools: dict[str, Callable[..., dict[str, Any]]] = {}

        # 通用工具
        tools["respond"] = lambda p: {
            "output": p.get("message", ""),
            "exit_code": 0,
        }
        tools["think"] = lambda p: {
            "output": f"[思考] {p.get('thought', '')}",
            "exit_code": 0,
        }

        # 领域专属工具
        if self.domain == "conversation":
            tools["query_stats"] = lambda p: {
                "output": f"查询结果: {p.get('query', '')}",
                "exit_code": 0,
            }
            tools["send_message"] = lambda p: {
                "output": f"消息已发送: {p.get('target', '')}",
                "exit_code": 0,
            }
            tools["web_search"] = lambda p: {
                "output": f"搜索结果: {p.get('query', '')}",
                "exit_code": 0,
            }
        elif self.domain == "engineering":
            tools["read_file"] = lambda p: {
                "output": f"# 文件内容: {p.get('file_path', '')}\n...",
                "exit_code": 0,
            }
            tools["grep"] = lambda p: {
                "output": f"匹配结果: {p.get('pattern', '')}",
                "exit_code": 0,
            }
            tools["bash"] = lambda p: {
                "output": f"$ {p.get('command', '')}\n(模拟输出)",
                "exit_code": 0,
            }
            tools["git"] = lambda p: {
                "output": f"git {p.get('subcommand', 'status')}\n(模拟输出)",
                "exit_code": 0,
            }
            tools["submit"] = lambda p: {
                "output": p.get("conclusion", "已提交"),
                "exit_code": 0,
            }
        elif self.domain == "advisory":
            tools["knowledge_base"] = lambda p: {
                "output": f"知识库查询: {p.get('query', '')}",
                "exit_code": 0,
            }
            tools["web_search"] = lambda p: {
                "output": f"搜索结果: {p.get('query', '')}",
                "exit_code": 0,
            }
            tools["submit"] = lambda p: {
                "output": p.get("recommendation", "已提交"),
                "exit_code": 0,
            }
        elif self.domain == "discussion":
            tools["knowledge_base"] = lambda p: {
                "output": f"参考资料: {p.get('query', '')}",
                "exit_code": 0,
            }

        return tools
