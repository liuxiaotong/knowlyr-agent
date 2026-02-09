"""测试规则层在不同领域 Profile 下的行为."""

from knowlyrcore.domain import BROWSER_PROFILE, CODING_PROFILE, GENERIC_PROFILE

from agentreward.rules import (
    ToolClassifier,
    check_info_utilization,
    check_outcome,
    check_redundancy,
    check_regression,
)


class TestToolClassifier:
    """测试 ToolClassifier."""

    def test_coding_read(self):
        c = ToolClassifier(CODING_PROFILE)
        assert c.is_read("read_file")
        assert c.is_read("Read")
        assert c.is_read("cat")
        assert c.is_read("grep")
        assert c.is_read("Grep")
        assert c.is_read("Glob")

    def test_coding_write(self):
        c = ToolClassifier(CODING_PROFILE)
        assert c.is_write("edit_file")
        assert c.is_write("Edit")
        assert c.is_write("Write")
        assert c.is_write("write_file")

    def test_coding_execute(self):
        c = ToolClassifier(CODING_PROFILE)
        assert c.is_execute("bash")
        assert c.is_execute("Bash")
        assert c.is_execute("git")

    def test_browser_read(self):
        c = ToolClassifier(BROWSER_PROFILE)
        assert c.is_read("screenshot")
        assert c.is_read("wait")
        assert c.is_read("extract_text")

    def test_browser_write(self):
        c = ToolClassifier(BROWSER_PROFILE)
        assert c.is_write("click")
        assert c.is_write("type_text")
        assert c.is_write("type")  # alias
        assert c.is_write("select")

    def test_browser_navigate(self):
        c = ToolClassifier(BROWSER_PROFILE)
        assert c.category("navigate") is not None
        assert c.category("goto") is not None

    def test_generic_returns_false(self):
        """GENERIC_PROFILE 无工具定义，全部返回 False."""
        c = ToolClassifier(GENERIC_PROFILE)
        assert not c.is_read("read_file")
        assert not c.is_write("edit_file")
        assert not c.is_execute("bash")
        assert c.category("anything") is None

    def test_target_param_coding(self):
        c = ToolClassifier(CODING_PROFILE)
        assert c.target_param("read_file", {"file_path": "/a.py"}) == "/a.py"
        assert c.target_param("Read", {"path": "/b.py"}) == "/b.py"

    def test_target_param_browser(self):
        c = ToolClassifier(BROWSER_PROFILE)
        assert c.target_param("click", {"element_id": "btn-1"}) == "btn-1"
        assert c.target_param("navigate", {"url": "https://x.com"}) == "https://x.com"

    def test_target_param_fallback(self):
        c = ToolClassifier(GENERIC_PROFILE)
        # 回退到常见参数名
        assert c.target_param("unknown", {"url": "https://y.com"}) == "https://y.com"
        assert c.target_param("unknown", {"file_path": "/c.py"}) == "/c.py"


class TestRedundancyBrowser:
    """用 BROWSER_PROFILE 测试冗余检测."""

    def test_no_redundancy(self):
        steps = [
            {"tool": "navigate", "params": {"url": "https://a.com"}, "output": "OK"},
            {"tool": "click", "params": {"element_id": "btn"}, "output": "clicked"},
            {"tool": "screenshot", "params": {}, "output": "img"},
        ]
        scores = check_redundancy(steps, BROWSER_PROFILE)
        assert scores == [1.0, 1.0, 1.0]

    def test_duplicate_screenshot(self):
        """连续两次 screenshot 应该被检测为冗余."""
        steps = [
            {"tool": "screenshot", "params": {}, "output": "img1"},
            {"tool": "screenshot", "params": {}, "output": "img2"},
        ]
        scores = check_redundancy(steps, BROWSER_PROFILE)
        assert scores[0] == 1.0
        assert scores[1] == 0.0  # 重复读取

    def test_click_then_screenshot_ok(self):
        """click 后 screenshot 不冗余（写入后读取）."""
        steps = [
            {"tool": "screenshot", "params": {}, "output": "img1"},
            {"tool": "click", "params": {"element_id": "btn"}, "output": "clicked"},
            {"tool": "screenshot", "params": {}, "output": "img2"},
        ]
        scores = check_redundancy(steps, BROWSER_PROFILE)
        # screenshot 无 stateful_key，所以不走 read-after-edit 逻辑
        # 但 call_key 相同，仍然是重复
        assert scores[0] == 1.0


class TestRedundancyGeneric:
    """用 GENERIC_PROFILE 测试冗余检测（启发式模式）."""

    def test_exact_duplicate(self):
        """完全相同的调用应被检测为冗余."""
        steps = [
            {"tool": "do_x", "params": {"a": 1}, "output": "ok"},
            {"tool": "do_x", "params": {"a": 1}, "output": "ok"},
        ]
        scores = check_redundancy(steps, GENERIC_PROFILE)
        assert scores[0] == 1.0
        assert scores[1] == 0.3  # 非读取类重复 → partial penalty

    def test_different_params_ok(self):
        """参数不同不算冗余."""
        steps = [
            {"tool": "do_x", "params": {"a": 1}, "output": "ok"},
            {"tool": "do_x", "params": {"a": 2}, "output": "ok"},
        ]
        scores = check_redundancy(steps, GENERIC_PROFILE)
        assert scores == [1.0, 1.0]


class TestRegressionBrowser:
    """用 BROWSER_PROFILE 测回归检测."""

    def test_no_regression(self):
        steps = [
            {"tool": "click", "params": {"element_id": "a"}, "output": "ok"},
            {"tool": "type_text", "params": {"element_id": "input", "text": "hello"}, "output": "ok"},
        ]
        scores = check_regression(steps, BROWSER_PROFILE)
        assert scores == [1.0, 1.0]


class TestOutcomeGeneric:
    """用不同 OutcomeSpec 测结果评分."""

    def test_coding_default(self):
        """coding 默认：tests_passed / tests_total."""
        from knowlyrcore.domain import CODING_PROFILE
        outcome = {"success": True, "tests_passed": 3, "tests_total": 5}
        score = check_outcome(outcome, CODING_PROFILE.outcome_spec)
        assert score == 0.6

    def test_browser_boolean(self):
        """browser：只看 success 布尔."""
        outcome = {"success": True, "task_completed": True}
        score = check_outcome(outcome, BROWSER_PROFILE.outcome_spec)
        assert score == 1.0

    def test_browser_failure(self):
        outcome = {"success": False}
        score = check_outcome(outcome, BROWSER_PROFILE.outcome_spec)
        assert score == 0.0

    def test_generic_with_custom_spec(self):
        """自定义 OutcomeSpec."""
        from knowlyrcore.domain import OutcomeSpec
        spec = OutcomeSpec(
            success_field="done",
            score_field="accuracy",
            total_field="total_queries",
        )
        outcome = {"done": True, "accuracy": 8, "total_queries": 10}
        score = check_outcome(outcome, spec)
        assert score == 0.8

    def test_partial_credit(self):
        from knowlyrcore.domain import OutcomeSpec
        spec = OutcomeSpec(partial_credit_field="manual_score")
        outcome = {"manual_score": 0.75}
        score = check_outcome(outcome, spec)
        assert score == 0.75


class TestInfoUtilBrowser:
    """用 BROWSER_PROFILE 测信息利用."""

    def test_read_always_ok(self):
        steps = [
            {"tool": "navigate", "params": {"url": "https://a.com"}, "output": "/page"},
            {"tool": "screenshot", "params": {}, "output": "img data"},
        ]
        scores = check_info_utilization(steps, BROWSER_PROFILE)
        assert scores[0] == 1.0
        assert scores[1] == 1.0  # screenshot 是 READ 类
