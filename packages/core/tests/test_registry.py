"""测试环境注册表 — register / make / list_envs / spec."""

import pytest

from knowlyrcore.env import AgentEnv
from knowlyrcore.registry import (
    EnvSpec,
    _clear_registry,
    list_envs,
    make,
    register,
    spec,
)
from knowlyrcore.timestep import TimeStep


# ── 测试用环境 ─────────────────────────────────────────────────────


class DummyEnv(AgentEnv):
    """测试用环境."""

    domain = "test"

    def __init__(self, greeting: str = "hello"):
        self.greeting = greeting

    def reset(self, *, task=None, seed=None) -> TimeStep:
        return TimeStep(observation=self.greeting)

    def step(self, action: dict) -> TimeStep:
        return TimeStep(observation="step")


class BrowserEnv(AgentEnv):
    """测试用浏览器环境."""

    domain = "browser"

    def reset(self, *, task=None, seed=None) -> TimeStep:
        return TimeStep(observation="browser ready")

    def step(self, action: dict) -> TimeStep:
        return TimeStep(observation="clicked")


# ── 测试 ──────────────────────────────────────────────────────────


class TestRegister:
    """register() 测试."""

    def setup_method(self):
        _clear_registry()

    def test_register_basic(self):
        register("test/dummy", DummyEnv)
        assert "test/dummy" in list_envs()

    def test_register_with_kwargs(self):
        register("test/custom", DummyEnv, greeting="你好")
        env = make("test/custom")
        assert env.greeting == "你好"

    def test_register_with_domain(self):
        register("test/browser", BrowserEnv, domain="browser")
        envs = list_envs(domain="browser")
        assert "test/browser" in envs

    def test_register_with_description(self):
        register("test/desc", DummyEnv, description="测试环境")
        s = spec("test/desc")
        assert s.description == "测试环境"

    def test_register_duplicate_raises(self):
        register("test/dup", DummyEnv)
        with pytest.raises(ValueError, match="已注册"):
            register("test/dup", DummyEnv)


class TestMake:
    """make() 测试."""

    def setup_method(self):
        _clear_registry()

    def test_make_basic(self):
        register("test/dummy", DummyEnv)
        env = make("test/dummy")
        assert isinstance(env, DummyEnv)
        assert env.greeting == "hello"

    def test_make_with_override(self):
        register("test/dummy", DummyEnv, greeting="default")
        env = make("test/dummy", greeting="override")
        assert env.greeting == "override"

    def test_make_unknown_raises(self):
        with pytest.raises(KeyError, match="未注册"):
            make("nonexistent/env")

    def test_make_returns_fresh_instance(self):
        register("test/dummy", DummyEnv)
        env1 = make("test/dummy")
        env2 = make("test/dummy")
        assert env1 is not env2

    def test_make_env_works(self):
        """创建的环境应能正常 reset/step."""
        register("test/dummy", DummyEnv)
        env = make("test/dummy")
        ts = env.reset()
        assert ts.observation == "hello"
        ts = env.step({"tool": "test"})
        assert ts.observation == "step"


class TestListEnvs:
    """list_envs() 测试."""

    def setup_method(self):
        _clear_registry()

    def test_empty(self):
        assert list_envs() == []

    def test_list_all(self):
        register("a/env1", DummyEnv)
        register("b/env2", DummyEnv)
        assert sorted(list_envs()) == ["a/env1", "b/env2"]

    def test_filter_by_domain(self):
        register("test/coding", DummyEnv, domain="coding")
        register("test/browser", BrowserEnv, domain="browser")
        register("test/generic", DummyEnv, domain="generic")

        assert list_envs(domain="coding") == ["test/coding"]
        assert list_envs(domain="browser") == ["test/browser"]
        assert list_envs(domain="generic") == ["test/generic"]
        assert len(list_envs()) == 3


class TestSpec:
    """spec() 测试."""

    def setup_method(self):
        _clear_registry()

    def test_spec_found(self):
        register("test/dummy", DummyEnv, domain="test", description="测试")
        s = spec("test/dummy")
        assert isinstance(s, EnvSpec)
        assert s.id == "test/dummy"
        assert s.env_cls is DummyEnv
        assert s.domain == "test"
        assert s.description == "测试"

    def test_spec_not_found(self):
        assert spec("nonexistent") is None

    def test_spec_kwargs(self):
        register("test/custom", DummyEnv, greeting="hi")
        s = spec("test/custom")
        assert s.kwargs["greeting"] == "hi"
