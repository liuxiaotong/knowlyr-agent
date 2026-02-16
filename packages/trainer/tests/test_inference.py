"""测试推理桥 — parse_action + AgentInference."""

import json

import pytest

from agenttrainer.inference import (
    AgentInference,
    parse_action,
    _action_to_text,
    _infer_params,
)


# ── parse_action 测试 ──────────────────────────────────────────────


class TestParseAction:
    """测试动作解析."""

    def test_structured_format_full(self):
        """完整 XML 结构化格式."""
        text = """<thought>分析一下代码结构</thought>
<action>read_file</action>
<params>{"file_path": "main.py"}</params>"""

        result = parse_action(text)
        assert result["tool"] == "read_file"
        assert result["params"] == {"file_path": "main.py"}
        assert result["thought"] == "分析一下代码结构"

    def test_structured_format_no_thought(self):
        """XML 格式无 thought."""
        text = '<action>bash</action>\n<params>{"command": "ls -la"}</params>'
        result = parse_action(text)
        assert result["tool"] == "bash"
        assert result["params"] == {"command": "ls -la"}
        assert result["thought"] == ""

    def test_structured_format_no_params(self):
        """XML 格式无 params."""
        text = "<thought>完成了</thought>\n<action>submit</action>"
        result = parse_action(text)
        assert result["tool"] == "submit"
        assert result["params"] == {}

    def test_structured_format_invalid_json_params(self):
        """params 内容非法 JSON 时返回空 dict."""
        text = "<action>bash</action>\n<params>not json</params>"
        result = parse_action(text)
        assert result["tool"] == "bash"
        assert result["params"] == {}

    def test_simple_format_with_arg(self):
        """简化格式: tool_name arg."""
        result = parse_action("read_file main.py")
        assert result["tool"] == "read_file"
        assert result["params"] == {"file_path": "main.py"}

    def test_simple_format_bash(self):
        result = parse_action("bash ls -la /tmp")
        assert result["tool"] == "bash"
        assert result["params"] == {"command": "ls -la /tmp"}

    def test_simple_format_respond(self):
        result = parse_action("respond 你好，有什么需要帮助的?")
        assert result["tool"] == "respond"
        assert result["params"] == {"message": "你好，有什么需要帮助的?"}

    def test_simple_format_json_arg(self):
        """简化格式但参数是 JSON."""
        result = parse_action('grep {"pattern": "TODO", "path": "src/"}')
        assert result["tool"] == "grep"
        assert result["params"] == {"pattern": "TODO", "path": "src/"}

    def test_simple_format_no_arg(self):
        """只有工具名，无参数."""
        result = parse_action("submit")
        assert result["tool"] == "submit"
        assert result["params"] == {"conclusion": ""}

    def test_empty_text(self):
        result = parse_action("")
        assert result["tool"] == "think"

    def test_whitespace_only(self):
        result = parse_action("   \n  ")
        assert result["tool"] == "think"

    def test_unknown_tool_simple(self):
        """未知工具使用 'input' 作为参数名."""
        result = parse_action("my_custom_tool some_value")
        assert result["tool"] == "my_custom_tool"
        assert result["params"] == {"input": "some_value"}


# ── _infer_params 测试 ─────────────────────────────────────────────


class TestInferParams:
    """测试参数推断."""

    def test_file_tools(self):
        assert _infer_params("read_file", "main.py") == {"file_path": "main.py"}
        assert _infer_params("file_read", "main.py") == {"file_path": "main.py"}

    def test_command_tools(self):
        assert _infer_params("bash", "ls") == {"command": "ls"}
        assert _infer_params("shell", "pwd") == {"command": "pwd"}

    def test_search_tools(self):
        assert _infer_params("grep", "TODO") == {"pattern": "TODO"}
        assert _infer_params("web_search", "RL training") == {"query": "RL training"}

    def test_unknown_tool(self):
        assert _infer_params("mystery", "arg") == {"input": "arg"}


# ── _action_to_text 测试 ──────────────────────────────────────────


class TestActionToText:
    """测试 action → text 转换."""

    def test_full_action(self):
        action = {
            "tool": "read_file",
            "params": {"file_path": "main.py"},
            "thought": "看看代码",
        }
        text = _action_to_text(action)
        assert "<thought>看看代码</thought>" in text
        assert "<action>read_file</action>" in text
        assert "<params>" in text
        assert "main.py" in text

    def test_no_thought(self):
        action = {"tool": "submit", "params": {"conclusion": "ok"}}
        text = _action_to_text(action)
        assert "<thought>" not in text
        assert "<action>submit</action>" in text

    def test_no_params(self):
        action = {"tool": "think", "params": {}, "thought": "想想"}
        text = _action_to_text(action)
        assert "<thought>想想</thought>" in text
        assert "<params>" not in text  # 空 params 不输出

    def test_roundtrip(self):
        """action → text → parse_action 应还原."""
        original = {
            "tool": "bash",
            "params": {"command": "pytest tests/"},
            "thought": "跑一下测试",
        }
        text = _action_to_text(original)
        parsed = parse_action(text)
        assert parsed["tool"] == original["tool"]
        assert parsed["params"] == original["params"]
        assert parsed["thought"] == original["thought"]


# ── AgentInference 测试 ───────────────────────────────────────────


class TestAgentInference:
    """测试 AgentInference (使用 mock model)."""

    def _make_mock_inference(self):
        """创建 mock 的 AgentInference (不需要真实模型)."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        inference = AgentInference(
            model=mock_model,
            tokenizer=mock_tokenizer,
            temperature=0.7,
            max_new_tokens=256,
        )
        return inference

    def test_create_agent_returns_callable(self):
        """create_agent 应返回 callable."""
        inference = self._make_mock_inference()
        agent = inference.create_agent(system_prompt="你是助手")
        assert callable(agent)

    def test_create_agent_maintains_history(self):
        """agent 函数应维护对话历史."""
        from unittest.mock import MagicMock, patch

        inference = self._make_mock_inference()

        # Mock generate_action 返回固定值
        call_count = [0]

        def mock_generate(messages):
            call_count[0] += 1
            # 第一次调用 messages 只有 system + 1 user
            # 第二次调用 messages 有 system + user + assistant + user
            if call_count[0] == 1:
                assert len(messages) == 2  # system + user
            elif call_count[0] == 2:
                assert len(messages) == 4  # system + user + assistant + user
            return {"tool": "think", "params": {"thought": "ok"}, "thought": "ok"}

        inference.generate_action = mock_generate

        agent = inference.create_agent(system_prompt="你是助手")
        agent("第一条消息")
        agent("第二条消息")

        assert call_count[0] == 2

    def test_create_agent_without_system_prompt(self):
        """不传 system_prompt 时也应正常工作."""
        inference = self._make_mock_inference()

        call_count = [0]

        def mock_generate(messages):
            call_count[0] += 1
            # 无 system prompt，只有 user
            if call_count[0] == 1:
                assert len(messages) == 1
                assert messages[0]["role"] == "user"
            return {"tool": "respond", "params": {"message": "hi"}, "thought": ""}

        inference.generate_action = mock_generate

        agent = inference.create_agent()
        result = agent("你好")
        assert result["tool"] == "respond"
