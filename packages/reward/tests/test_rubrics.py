"""Tests for rubric definitions."""

from agentreward.rubrics import Rubric, RubricSet, get_default_rubric_set


class TestRubric:
    """Test Rubric model."""

    def test_create_rubric(self):
        rubric = Rubric(
            id="test",
            name="Test Rubric",
            description="A test rubric",
            weight=0.5,
            evaluator="rule",
        )
        assert rubric.id == "test"
        assert rubric.name == "Test Rubric"
        assert rubric.weight == 0.5
        assert rubric.evaluator == "rule"

    def test_rubric_default_evaluator(self):
        rubric = Rubric(
            id="test",
            name="Test",
            description="Test",
            weight=0.3,
        )
        assert rubric.evaluator == "rule"


class TestRubricSet:
    """Test RubricSet model."""

    def test_create_rubric_set(self):
        rubrics = [
            Rubric(id="r1", name="R1", description="D1", weight=0.5, evaluator="rule"),
            Rubric(id="r2", name="R2", description="D2", weight=0.3, evaluator="model"),
            Rubric(id="r3", name="R3", description="D3", weight=0.2, evaluator="human"),
        ]
        rs = RubricSet(rubrics=rubrics)
        assert len(rs.rubrics) == 3

    def test_get_rule_rubrics(self):
        rubrics = [
            Rubric(id="r1", name="R1", description="D1", weight=0.5, evaluator="rule"),
            Rubric(id="r2", name="R2", description="D2", weight=0.3, evaluator="model"),
            Rubric(id="r3", name="R3", description="D3", weight=0.2, evaluator="rule"),
        ]
        rs = RubricSet(rubrics=rubrics)
        rule_rubrics = rs.get_rule_rubrics()
        assert len(rule_rubrics) == 2
        assert all(r.evaluator == "rule" for r in rule_rubrics)

    def test_get_model_rubrics(self):
        rubrics = [
            Rubric(id="r1", name="R1", description="D1", weight=0.5, evaluator="rule"),
            Rubric(id="r2", name="R2", description="D2", weight=0.3, evaluator="model"),
        ]
        rs = RubricSet(rubrics=rubrics)
        model_rubrics = rs.get_model_rubrics()
        assert len(model_rubrics) == 1
        assert model_rubrics[0].id == "r2"

    def test_get_by_id(self):
        rubrics = [
            Rubric(id="r1", name="R1", description="D1", weight=0.5, evaluator="rule"),
            Rubric(id="r2", name="R2", description="D2", weight=0.3, evaluator="model"),
        ]
        rs = RubricSet(rubrics=rubrics)
        assert rs.get_by_id("r1") is not None
        assert rs.get_by_id("r1").name == "R1"
        assert rs.get_by_id("nonexistent") is None

    def test_total_weight(self):
        rubrics = [
            Rubric(id="r1", name="R1", description="D1", weight=0.6, evaluator="rule"),
            Rubric(id="r2", name="R2", description="D2", weight=0.4, evaluator="model"),
        ]
        rs = RubricSet(rubrics=rubrics)
        assert abs(rs.total_weight() - 1.0) < 1e-6

    def test_to_prompt_description(self):
        rubrics = [
            Rubric(id="r1", name="R1", description="Test desc", weight=0.5, evaluator="rule"),
        ]
        rs = RubricSet(rubrics=rubrics)
        desc = rs.to_prompt_description()
        assert "r1" in desc
        assert "Test desc" in desc


class TestDefaultRubricSet:
    """Test the default rubric set."""

    def test_default_rubric_count(self):
        rs = get_default_rubric_set()
        assert len(rs.rubrics) == 5

    def test_default_rubric_weights_sum(self):
        rs = get_default_rubric_set()
        assert abs(rs.total_weight() - 1.0) < 1e-6

    def test_default_rubric_ids(self):
        rs = get_default_rubric_set()
        ids = {r.id for r in rs.rubrics}
        expected = {"goal_progress", "tool_choice", "param_correctness",
                    "info_utilization", "non_redundancy"}
        assert ids == expected

    def test_default_has_rule_and_model(self):
        rs = get_default_rubric_set()
        assert len(rs.get_rule_rubrics()) > 0
        assert len(rs.get_model_rubrics()) > 0
