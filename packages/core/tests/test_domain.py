"""测试领域配置 DomainProfile."""

import json

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


class TestToolCategory:
    """测试 ToolCategory 枚举."""

    def test_values(self):
        """所有分类值正确."""
        assert ToolCategory.READ == "read"
        assert ToolCategory.WRITE == "write"
        assert ToolCategory.SEARCH == "search"
        assert ToolCategory.EXECUTE == "execute"
        assert ToolCategory.NAVIGATE == "navigate"
        assert ToolCategory.SUBMIT == "submit"
        assert ToolCategory.THINK == "think"

    def test_count(self):
        """共 7 个分类."""
        assert len(ToolCategory) == 7


class TestToolSpec:
    """测试 ToolSpec."""

    def test_basic(self):
        spec = ToolSpec(name="click", category=ToolCategory.WRITE)
        assert spec.name == "click"
        assert spec.category == ToolCategory.WRITE
        assert spec.stateful_key == ""
        assert spec.aliases == []

    def test_with_aliases(self):
        spec = ToolSpec(
            name="read_file", category=ToolCategory.READ,
            stateful_key="file_path", aliases=["Read", "cat"],
        )
        assert spec.stateful_key == "file_path"
        assert spec.aliases == ["Read", "cat"]

    def test_serialization(self):
        spec = ToolSpec(name="bash", category=ToolCategory.EXECUTE)
        data = spec.model_dump()
        assert data["name"] == "bash"
        assert data["category"] == "execute"
        restored = ToolSpec.model_validate(data)
        assert restored == spec


class TestOutcomeSpec:
    """测试 OutcomeSpec."""

    def test_defaults(self):
        spec = OutcomeSpec()
        assert spec.success_field == "success"
        assert spec.score_field == ""
        assert spec.total_field == ""
        assert spec.partial_credit_field == "partial_credit"

    def test_coding(self):
        spec = OutcomeSpec(
            success_field="success",
            score_field="tests_passed",
            total_field="tests_total",
        )
        assert spec.score_field == "tests_passed"


class TestDomainProfile:
    """测试 DomainProfile."""

    def test_minimal(self):
        profile = DomainProfile(domain="test")
        assert profile.domain == "test"
        assert profile.tools == []
        assert profile.outcome_spec.success_field == "success"

    def test_serialization(self):
        profile = DomainProfile(
            domain="custom",
            display_name="Custom Agent",
            tools=[ToolSpec(name="do_thing", category=ToolCategory.EXECUTE)],
        )
        data = profile.model_dump()
        assert data["domain"] == "custom"
        assert len(data["tools"]) == 1
        restored = DomainProfile.model_validate(data)
        assert restored.domain == "custom"
        assert restored.tools[0].name == "do_thing"

    def test_json_roundtrip(self, tmp_path):
        """JSON 文件保存/加载往返."""
        profile = DomainProfile(
            domain="data_analysis",
            display_name="Data Analysis Agent",
            tools=[
                ToolSpec(name="query_db", category=ToolCategory.READ),
                ToolSpec(name="plot", category=ToolCategory.WRITE),
            ],
            outcome_spec=OutcomeSpec(success_field="completed"),
        )
        path = tmp_path / "profile.json"
        with open(path, "w") as f:
            json.dump(profile.model_dump(), f)

        loaded = load_domain_profile(str(path))
        assert loaded.domain == "data_analysis"
        assert len(loaded.tools) == 2
        assert loaded.outcome_spec.success_field == "completed"


class TestBuiltinProfiles:
    """测试内置 Profile."""

    def test_coding_profile(self):
        p = CODING_PROFILE
        assert p.domain == "coding"
        tool_names = {s.name for s in p.tools}
        assert "read_file" in tool_names
        assert "edit_file" in tool_names
        assert "bash" in tool_names
        assert "grep" in tool_names
        assert "git" in tool_names
        assert "submit" in tool_names

    def test_coding_aliases(self):
        """coding profile 的 aliases 覆盖 rules.py 中硬编码的工具名."""
        all_names: set[str] = set()
        for spec in CODING_PROFILE.tools:
            all_names.add(spec.name)
            all_names.update(spec.aliases)
        # rules.py 中使用的所有工具名都应被覆盖
        for name in ["read_file", "Read", "cat", "grep", "Grep", "Glob",
                      "find", "ls", "Edit", "edit_file", "sed", "write_file",
                      "Write", "Bash", "bash", "shell", "submit"]:
            assert name in all_names, f"{name} 未被 CODING_PROFILE 覆盖"

    def test_coding_outcome(self):
        spec = CODING_PROFILE.outcome_spec
        assert spec.score_field == "tests_passed"
        assert spec.total_field == "tests_total"

    def test_browser_profile(self):
        p = BROWSER_PROFILE
        assert p.domain == "browser"
        tool_names = {s.name for s in p.tools}
        assert "click" in tool_names
        assert "navigate" in tool_names
        assert "screenshot" in tool_names

    def test_generic_profile(self):
        p = GENERIC_PROFILE
        assert p.domain == "generic"
        assert p.tools == []


class TestGetDomainProfile:
    """测试 profile 查找."""

    def test_get_coding(self):
        assert get_domain_profile("coding") is CODING_PROFILE

    def test_get_browser(self):
        assert get_domain_profile("browser") is BROWSER_PROFILE

    def test_get_generic(self):
        assert get_domain_profile("generic") is GENERIC_PROFILE

    def test_unknown_returns_generic(self):
        assert get_domain_profile("unknown_domain") is GENERIC_PROFILE

    def test_list_profiles(self):
        names = list_domain_profiles()
        assert "coding" in names
        assert "browser" in names
        assert "generic" in names
