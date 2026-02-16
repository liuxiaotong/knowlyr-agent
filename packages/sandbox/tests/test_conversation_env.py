"""测试 ConversationEnv — 对话类 Agent 环境."""


from knowlyrcore.env import AgentEnv
from knowlyrcore.timestep import TimeStep

from agentsandbox.conversation_env import ConversationEnv


# ── 基础接口 ──────────────────────────────────────────────────────


class TestConversationEnvInterface:
    """验证 ConversationEnv 实现 AgentEnv 协议."""

    def test_is_agent_env(self):
        assert issubclass(ConversationEnv, AgentEnv)

    def test_default_domain(self):
        env = ConversationEnv()
        assert env.domain == "conversation"

    def test_custom_domain(self):
        env = ConversationEnv(domain="engineering")
        assert env.domain == "engineering"

    def test_available_tools_conversation(self):
        env = ConversationEnv(domain="conversation")
        env.reset(task="你好")
        tools = env.available_tools
        assert "respond" in tools
        assert "think" in tools
        assert "web_search" in tools

    def test_available_tools_engineering(self):
        env = ConversationEnv(domain="engineering")
        env.reset(task="审查代码")
        tools = env.available_tools
        assert "read_file" in tools
        assert "grep" in tools
        assert "bash" in tools
        assert "submit" in tools

    def test_available_tools_advisory(self):
        env = ConversationEnv(domain="advisory")
        env.reset(task="分析市场")
        tools = env.available_tools
        assert "knowledge_base" in tools
        assert "web_search" in tools
        assert "submit" in tools


# ── Reset ────────────────────────────────────────────────────────


class TestConversationEnvReset:
    """测试 reset 方法."""

    def test_reset_with_string_task(self):
        env = ConversationEnv()
        ts = env.reset(task="你们的产品多少钱?")
        assert isinstance(ts, TimeStep)
        assert ts.observation == "你们的产品多少钱?"
        assert not ts.terminated
        assert not ts.truncated

    def test_reset_with_dict_task(self):
        env = ConversationEnv()
        ts = env.reset(task={"user_message": "帮我查一下", "description": "用户咨询"})
        assert ts.observation == "帮我查一下"

    def test_reset_with_dict_fallback(self):
        """dict 中没有 user_message 时 fallback 到 description."""
        env = ConversationEnv()
        ts = env.reset(task={"description": "分析报告"})
        assert ts.observation == "分析报告"

    def test_reset_with_object_task(self):
        """支持对象属性."""
        class MockTask:
            user_message = "对象消息"
            description = "任务描述"

        env = ConversationEnv()
        ts = env.reset(task=MockTask())
        assert ts.observation == "对象消息"

    def test_reset_with_none(self):
        env = ConversationEnv()
        ts = env.reset()
        assert ts.observation == ""
        assert not ts.done

    def test_reset_clears_history(self):
        env = ConversationEnv()
        env.reset(task="第一次")
        env.step({"tool": "think", "params": {"thought": "想一下"}})
        assert len(env.history) == 1

        env.reset(task="第二次")
        assert len(env.history) == 0

    def test_reset_info(self):
        env = ConversationEnv(domain="engineering", max_turns=10)
        ts = env.reset(task="审查")
        assert ts.info["domain"] == "engineering"
        assert ts.info["max_turns"] == 10
        assert isinstance(ts.info["available_tools"], list)


# ── Step ─────────────────────────────────────────────────────────


class TestConversationEnvStep:
    """测试 step 方法."""

    def test_step_basic(self):
        env = ConversationEnv()
        env.reset(task="你好")
        ts = env.step({"tool": "think", "params": {"thought": "分析一下"}})
        assert isinstance(ts, TimeStep)
        assert "[思考]" in ts.observation
        assert not ts.terminated  # think 不终止

    def test_step_unknown_tool(self):
        env = ConversationEnv()
        env.reset(task="你好")
        ts = env.step({"tool": "不存在", "params": {}})
        assert ts.info["exit_code"] == 1
        assert "未知工具" in ts.observation

    def test_step_records_history(self):
        env = ConversationEnv()
        env.reset(task="你好")
        env.step({"tool": "think", "params": {"thought": "a"}})
        env.step({"tool": "think", "params": {"thought": "b"}})
        assert len(env.history) == 2
        assert env.history[0]["step"] == 1
        assert env.history[1]["step"] == 2

    def test_step_reward_default_zero(self):
        env = ConversationEnv()
        env.reset(task="你好")
        ts = env.step({"tool": "think", "params": {}})
        assert ts.reward == 0.0


# ── 终止条件 ─────────────────────────────────────────────────────


class TestTermination:
    """测试不同领域的终止条件."""

    def test_conversation_respond_terminates(self):
        """conversation 领域: respond 即完成."""
        env = ConversationEnv(domain="conversation")
        env.reset(task="你好")
        ts = env.step({"tool": "respond", "params": {"message": "你好!"}})
        assert ts.terminated
        assert not ts.truncated

    def test_conversation_think_not_terminates(self):
        env = ConversationEnv(domain="conversation")
        env.reset(task="你好")
        ts = env.step({"tool": "think", "params": {"thought": "想想"}})
        assert not ts.terminated

    def test_engineering_submit_terminates(self):
        """engineering 领域: submit 才完成."""
        env = ConversationEnv(domain="engineering")
        env.reset(task="审查代码")
        ts = env.step({"tool": "read_file", "params": {"file_path": "main.py"}})
        assert not ts.terminated

        ts = env.step({"tool": "submit", "params": {"conclusion": "无问题"}})
        assert ts.terminated

    def test_engineering_respond_not_terminates(self):
        """engineering 领域: respond 不终止 (需要 submit)."""
        env = ConversationEnv(domain="engineering")
        env.reset(task="审查")
        # engineering 没有 respond 工具，但即使调用也不终止
        ts = env.step({"tool": "think", "params": {}})
        assert not ts.terminated

    def test_advisory_submit_terminates(self):
        env = ConversationEnv(domain="advisory")
        env.reset(task="分析市场")
        ts = env.step({"tool": "knowledge_base", "params": {"query": "市场数据"}})
        assert not ts.terminated

        ts = env.step({"tool": "submit", "params": {"recommendation": "建议扩张"}})
        assert ts.terminated

    def test_truncation_on_max_turns(self):
        """超过 max_turns 时 truncated."""
        env = ConversationEnv(domain="engineering", max_turns=2)
        env.reset(task="审查")
        ts = env.step({"tool": "think", "params": {}})
        assert not ts.truncated

        ts = env.step({"tool": "think", "params": {}})
        assert ts.truncated
        assert not ts.terminated

    def test_terminated_overrides_truncation(self):
        """最后一步同时 terminated + 到 max_turns，terminated 优先."""
        env = ConversationEnv(domain="engineering", max_turns=1)
        env.reset(task="审查")
        ts = env.step({"tool": "submit", "params": {}})
        assert ts.terminated
        assert not ts.truncated


# ── 自定义工具注入 ────────────────────────────────────────────────


class TestCustomTools:
    """测试外部工具注入."""

    def test_inject_custom_tools(self):
        custom_tools = {
            "my_tool": lambda p: {"output": f"result: {p.get('x', '')}", "exit_code": 0},
            "respond": lambda p: {"output": p.get("message", ""), "exit_code": 0},
        }
        env = ConversationEnv(tools=custom_tools)
        env.reset(task="测试")

        assert env.available_tools == ["my_tool", "respond"]

        ts = env.step({"tool": "my_tool", "params": {"x": "42"}})
        assert ts.observation == "result: 42"
        assert not ts.terminated

        ts = env.step({"tool": "respond", "params": {"message": "完成"}})
        assert ts.terminated

    def test_custom_tools_override_defaults(self):
        """自定义工具完全替代默认工具."""
        custom_tools = {
            "only_one": lambda p: {"output": "ok", "exit_code": 0},
        }
        env = ConversationEnv(tools=custom_tools)
        env.reset(task="测试")
        assert env.available_tools == ["only_one"]


# ── Close ────────────────────────────────────────────────────────


class TestConversationEnvClose:
    """测试 close 方法."""

    def test_close_clears_state(self):
        env = ConversationEnv()
        env.reset(task="你好")
        env.step({"tool": "think", "params": {}})
        env.close()
        assert env.available_tools == []
        assert env.history == []


# ── 完整生命周期 ──────────────────────────────────────────────────


class TestConversationEnvLifecycle:
    """端到端生命周期测试."""

    def test_conversation_lifecycle(self):
        """conversation: reset → think → respond → done."""
        env = ConversationEnv(domain="conversation")
        ts = env.reset(task="你们的产品多少钱?")
        assert ts.observation == "你们的产品多少钱?"

        ts = env.step({"tool": "think", "params": {"thought": "查询定价信息"}})
        assert not ts.done

        ts = env.step({"tool": "respond", "params": {"message": "标准版 299 元/月"}})
        assert ts.done
        assert ts.terminated
        assert ts.observation == "标准版 299 元/月"

        env.close()

    def test_engineering_lifecycle(self):
        """engineering: reset → read_file → grep → submit → done."""
        env = ConversationEnv(domain="engineering")
        ts = env.reset(task="审查 auth.py 的安全性")

        ts = env.step({"tool": "read_file", "params": {"file_path": "auth.py"}})
        assert not ts.done

        ts = env.step({"tool": "grep", "params": {"pattern": "password"}})
        assert not ts.done

        ts = env.step({"tool": "submit", "params": {"conclusion": "发现硬编码密码"}})
        assert ts.done
        assert ts.terminated

        assert len(env.history) == 3
        env.close()

    def test_advisory_lifecycle(self):
        """advisory: reset → web_search → knowledge_base → submit → done."""
        env = ConversationEnv(domain="advisory")
        ts = env.reset(task="分析竞品定价策略")

        ts = env.step({"tool": "web_search", "params": {"query": "竞品定价"}})
        assert not ts.done

        ts = env.step({"tool": "knowledge_base", "params": {"query": "市场数据"}})
        assert not ts.done

        ts = env.step({"tool": "submit", "params": {"recommendation": "建议降价 10%"}})
        assert ts.done

        env.close()


# ── Registry 集成 ────────────────────────────────────────────────


class TestRegistryIntegration:
    """测试环境注册和 make() 创建."""

    def test_make_conversation(self):
        from knowlyrcore.registry import make

        env = make("knowlyr/conversation")
        assert isinstance(env, ConversationEnv)
        assert env.domain == "conversation"

    def test_make_engineering(self):
        from knowlyrcore.registry import make

        env = make("knowlyr/engineering")
        assert isinstance(env, ConversationEnv)
        assert env.domain == "engineering"

    def test_make_advisory(self):
        from knowlyrcore.registry import make

        env = make("knowlyr/advisory")
        assert isinstance(env, ConversationEnv)
        assert env.domain == "advisory"

    def test_make_discussion(self):
        from knowlyrcore.registry import make

        env = make("knowlyr/discussion")
        assert isinstance(env, ConversationEnv)
        assert env.domain == "discussion"

    def test_list_envs_includes_new(self):
        from knowlyrcore.registry import list_envs

        all_envs = list_envs()
        assert "knowlyr/conversation" in all_envs
        assert "knowlyr/engineering" in all_envs
        assert "knowlyr/advisory" in all_envs

    def test_list_envs_by_domain(self):
        from knowlyrcore.registry import list_envs

        conv_envs = list_envs(domain="conversation")
        assert "knowlyr/conversation" in conv_envs
        assert "knowlyr/engineering" not in conv_envs
