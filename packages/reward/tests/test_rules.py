"""Tests for rule-based reward functions."""

from agentreward.rules import (
    check_redundancy,
    check_efficiency,
    check_outcome,
    check_regression,
    check_info_utilization,
)


class TestCheckRedundancy:
    """Test redundancy detection."""

    def test_empty_steps(self):
        assert check_redundancy([]) == []

    def test_no_redundancy(self):
        steps = [
            {"tool": "Read", "params": {"file_path": "/a.py"}, "output": "content a"},
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "x", "new_string": "y"}},
            {"tool": "Read", "params": {"file_path": "/b.py"}, "output": "content b"},
        ]
        scores = check_redundancy(steps)
        assert len(scores) == 3
        assert all(s == 1.0 for s in scores)

    def test_duplicate_read(self):
        steps = [
            {"tool": "Read", "params": {"file_path": "/a.py"}, "output": "content"},
            {"tool": "Read", "params": {"file_path": "/a.py"}, "output": "content"},
        ]
        scores = check_redundancy(steps)
        assert scores[0] == 1.0
        assert scores[1] == 0.0  # Redundant

    def test_read_after_edit_is_ok(self):
        steps = [
            {"tool": "Read", "params": {"file_path": "/a.py"}, "output": "old"},
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "x", "new_string": "y"}},
            {"tool": "Read", "params": {"file_path": "/a.py"}, "output": "new"},
        ]
        scores = check_redundancy(steps)
        # Third read is after an edit, so it's justified
        assert scores[2] == 1.0


class TestCheckEfficiency:
    """Test efficiency scoring."""

    def test_equal_steps(self):
        steps = [{"tool": "a"}, {"tool": "b"}, {"tool": "c"}]
        assert check_efficiency(steps, 3) == 1.0

    def test_fewer_steps(self):
        steps = [{"tool": "a"}, {"tool": "b"}]
        assert check_efficiency(steps, 3) == 1.0

    def test_more_steps(self):
        steps = [{"tool": "a"}] * 6
        score = check_efficiency(steps, 3)
        assert score == 0.5

    def test_empty_steps(self):
        assert check_efficiency([], 3) == 0.0

    def test_zero_reference(self):
        assert check_efficiency([{"tool": "a"}], 0) == 1.0


class TestCheckOutcome:
    """Test outcome scoring."""

    def test_success(self):
        assert check_outcome({"success": True}) == 1.0

    def test_failure(self):
        assert check_outcome({"success": False}) == 0.0

    def test_empty(self):
        assert check_outcome({}) == 0.0

    def test_none(self):
        assert check_outcome(None) == 0.0

    def test_test_results(self):
        outcome = {"tests_passed": 8, "tests_total": 10}
        assert check_outcome(outcome) == 0.8

    def test_partial_credit(self):
        outcome = {"partial_credit": 0.65}
        assert check_outcome(outcome) == 0.65

    def test_partial_credit_clamped(self):
        assert check_outcome({"partial_credit": 1.5}) == 1.0
        assert check_outcome({"partial_credit": -0.5}) == 0.0


class TestCheckRegression:
    """Test regression (edit-then-revert) detection."""

    def test_empty_steps(self):
        assert check_regression([]) == []

    def test_no_regression(self):
        steps = [
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "x", "new_string": "y"}},
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "y", "new_string": "z"}},
        ]
        scores = check_regression(steps)
        assert all(s == 1.0 for s in scores)

    def test_revert_detected(self):
        steps = [
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "x", "new_string": "y"}},
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "y", "new_string": "x"}},
        ]
        scores = check_regression(steps)
        assert scores[0] == 1.0
        assert scores[1] == 0.0  # This is a revert

    def test_different_file_no_regression(self):
        steps = [
            {"tool": "Edit", "params": {"file_path": "/a.py", "old_string": "x", "new_string": "y"}},
            {"tool": "Edit", "params": {"file_path": "/b.py", "old_string": "y", "new_string": "x"}},
        ]
        scores = check_regression(steps)
        assert all(s == 1.0 for s in scores)


class TestCheckInfoUtilization:
    """Test information utilization scoring."""

    def test_empty_steps(self):
        assert check_info_utilization([]) == []

    def test_first_step_always_ok(self):
        steps = [{"tool": "Read", "params": {"file_path": "/a.py"}, "output": "content"}]
        scores = check_info_utilization(steps)
        assert scores[0] == 1.0

    def test_read_steps_always_ok(self):
        steps = [
            {"tool": "Read", "params": {"file_path": "/a.py"}, "output": "def foo():"},
            {"tool": "Grep", "params": {"pattern": "bar"}, "output": "found bar"},
            {"tool": "Glob", "params": {"pattern": "*.py"}, "output": "/a.py"},
        ]
        scores = check_info_utilization(steps)
        assert all(s == 1.0 for s in scores)
