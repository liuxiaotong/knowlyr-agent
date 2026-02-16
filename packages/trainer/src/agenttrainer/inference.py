"""推理桥 — 将训练后模型封装为 collect() 兼容的 agent 函数.

打通 Model → Env → Reward → Update 的闭环。训练完的 PyTorch 模型
通过 AgentInference 封装后，可直接传给 Hub 的 collect() 函数。

Usage::

    from agenttrainer.inference import AgentInference

    inference = AgentInference.from_pretrained("./checkpoints/step-1000")
    agent = inference.create_agent(system_prompt="你是代码审查员")

    from trajectoryhub import collect
    trajectories = collect("knowlyr/engineering", agent=agent, n_episodes=10)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentInference:
    """Agent 推理引擎.

    将 HuggingFace CausalLM 封装为 collect() 兼容的 agent 函数。
    复用 agenttrainer.models.loader 的模型加载逻辑。

    Attributes:
        model: HuggingFace CausalLM 模型
        tokenizer: 对应的 tokenizer
        temperature: 采样温度
        max_new_tokens: 最大生成 token 数
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        """初始化.

        Args:
            model: HuggingFace CausalLM (已加载到设备)
            tokenizer: 对应的 tokenizer
            temperature: 采样温度 (0.0 = greedy)
            max_new_tokens: 最大生成 token 数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        bf16: bool = True,
        device_map: str | None = None,
    ) -> AgentInference:
        """从 checkpoint 加载.

        复用 agenttrainer.models.loader.load_model()。

        Args:
            model_path: HuggingFace 模型名或本地 checkpoint 路径
            temperature: 采样温度
            max_new_tokens: 最大生成 token 数
            bf16: 是否使用 bfloat16
            device_map: 设备映射 (None = 自动选择)

        Returns:
            AgentInference 实例
        """
        from agenttrainer.models.loader import load_model

        model, tokenizer = load_model(
            model_path,
            bf16=bf16,
            device_map=device_map or _auto_device_map(),
        )
        model.eval()

        return cls(
            model,
            tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def create_agent(
        self,
        system_prompt: str = "",
    ) -> Callable[[str], dict[str, Any]]:
        """创建 collect() 兼容的 agent 函数.

        返回的函数签名: (observation: str) -> {"tool": ..., "params": {...}}
        内部通过闭包维护多轮对话历史。

        Args:
            system_prompt: system prompt (如员工角色描述)

        Returns:
            agent_fn: (observation: str) -> action dict
        """
        history: list[dict[str, str]] = []

        def agent_fn(observation: str) -> dict[str, Any]:
            # 构建 messages
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history)
            messages.append({"role": "user", "content": observation})

            # 生成
            action = self.generate_action(messages)

            # 更新历史
            history.append({"role": "user", "content": observation})
            history.append({"role": "assistant", "content": _action_to_text(action)})

            return action

        return agent_fn

    def generate_action(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """生成 action.

        Args:
            messages: chat messages 列表

        Returns:
            action dict: {"tool": str, "params": dict, "thought": str}
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        # 移到模型所在设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 解码新生成的部分
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return parse_action(generated_text)


# ── 动作解析 ──────────────────────────────────────────────────────


def parse_action(text: str) -> dict[str, Any]:
    """解析模型生成文本为 action dict.

    支持两种格式:

    1. 结构化格式 (XML-style)::

        <thought>分析一下代码</thought>
        <action>read_file</action>
        <params>{"file_path": "main.py"}</params>

    2. 简化格式::

        read_file main.py

    Args:
        text: 模型生成的原始文本

    Returns:
        {"tool": str, "params": dict, "thought": str}
    """
    text = text.strip()
    if not text:
        return {"tool": "think", "params": {"thought": ""}, "thought": ""}

    # 格式 1: 结构化标签
    action_match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if action_match:
        tool = action_match.group(1).strip()

        params: dict[str, Any] = {}
        params_match = re.search(r"<params>(.*?)</params>", text, re.DOTALL)
        if params_match:
            try:
                params = json.loads(params_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        thought = ""
        thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        return {"tool": tool, "params": params, "thought": thought}

    # 格式 2: 简化格式 "tool_name arg1 arg2"
    parts = text.split(maxsplit=1)
    if parts:
        tool = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        # 尝试解析 arg 为 JSON
        try:
            params = json.loads(arg)
            if isinstance(params, dict):
                return {"tool": tool, "params": params, "thought": ""}
        except (json.JSONDecodeError, ValueError):
            pass

        # 启发式参数映射
        params = _infer_params(tool, arg)
        return {"tool": tool, "params": params, "thought": ""}

    return {"tool": "think", "params": {"thought": text}, "thought": text}


def _infer_params(tool: str, arg: str) -> dict[str, Any]:
    """根据工具名推断参数结构."""
    _tool_param_map: dict[str, str] = {
        "read_file": "file_path",
        "file_read": "file_path",
        "edit_file": "file_path",
        "bash": "command",
        "shell": "command",
        "grep": "pattern",
        "search": "pattern",
        "glob": "file_pattern",
        "respond": "message",
        "reply": "message",
        "submit": "conclusion",
        "finish": "result",
        "think": "thought",
        "web_search": "query",
        "knowledge_base": "query",
        "query_stats": "query",
    }
    param_name = _tool_param_map.get(tool, "input")
    return {param_name: arg}


def _action_to_text(action: dict[str, Any]) -> str:
    """将 action dict 转为文本 (用于更新对话历史)."""
    parts: list[str] = []
    if action.get("thought"):
        parts.append(f"<thought>{action['thought']}</thought>")
    parts.append(f"<action>{action.get('tool', '')}</action>")
    if action.get("params"):
        params_str = json.dumps(action["params"], ensure_ascii=False)
        parts.append(f"<params>{params_str}</params>")
    return "\n".join(parts)


def _auto_device_map() -> str:
    """自动选择设备映射."""
    try:
        import torch
        if torch.cuda.is_available():
            return "auto"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "auto"
    except ImportError:
        pass
    return "cpu"
