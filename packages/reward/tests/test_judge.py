"""测试 LLM-as-Judge 模块."""

import json
from unittest.mock import patch

import pytest

from agentreward.judge import (
    JudgeConfig,
    StepJudgment,
    _extract_json,
    _fallback_judgment,
    _parse_judgment,
    build_judge_prompt,
    judge_step,
    judge_trajectory,
)
from agentreward.rubrics import Rubric


# ── 测试数据 ──────────────────────────────────────────────────────


def _model_rubrics():
    """3 个模型层评估维度."""
    return [
        Rubric(
            id="goal_progress", name="目标推进",
            description="这一步是否推进了任务目标？",
            weight=0.3, evaluator="model",
        ),
        Rubric(
            id="tool_choice", name="工具选择",
            description="选择的工具是否合理？",
            weight=0.2, evaluator="model",
        ),
        Rubric(
            id="param_correctness", name="参数正确性",
            description="工具调用的参数是否正确？",
            weight=0.2, evaluator="model",
        ),
    ]


def _sample_step():
    """一个典型的步骤."""
    return {
        "tool": "bash",
        "params": {"command": "pytest tests/"},
        "output": "3 passed, 0 failed",
    }


def _good_llm_response(rubrics=None):
    """LLM 返回的合法 JSON."""
    rubrics = rubrics or _model_rubrics()
    scores = {r.id: 0.8 for r in rubrics}
    return json.dumps({
        "scores": scores,
        "rationale": "这一步很好地推进了目标",
        "overall_score": 0.8,
    })


# ── build_judge_prompt 测试 ───────────────────────────────────────


class TestBuildJudgePrompt:
    """测试提示词构建."""

    def test_basic_prompt(self):
        """基本提示词应包含关键信息."""
        prompt = build_judge_prompt(
            step=_sample_step(),
            step_index=1,
            total_steps=3,
            context_summary="",
            rubrics=_model_rubrics(),
            task_description="修复排序 bug",
        )

        assert "修复排序 bug" in prompt
        assert "Step 1/3" in prompt
        assert "bash" in prompt
        assert "pytest tests/" in prompt
        assert "goal_progress" in prompt
        assert "tool_choice" in prompt

    def test_no_task_description(self):
        """未提供任务描述时应显示占位符."""
        prompt = build_judge_prompt(
            step=_sample_step(),
            step_index=1,
            total_steps=1,
            context_summary="",
            rubrics=_model_rubrics(),
        )
        assert "未提供任务描述" in prompt

    def test_no_context(self):
        """第一步应显示无前序上下文."""
        prompt = build_judge_prompt(
            step=_sample_step(),
            step_index=1,
            total_steps=1,
            context_summary="",
            rubrics=_model_rubrics(),
        )
        assert "第一步，无前序上下文" in prompt

    def test_with_context(self):
        """有前序上下文时应包含."""
        prompt = build_judge_prompt(
            step=_sample_step(),
            step_index=2,
            total_steps=3,
            context_summary="Step 1: read_file -> score=0.80",
            rubrics=_model_rubrics(),
        )
        assert "Step 1: read_file" in prompt

    def test_long_output_truncated(self):
        """超长输出应被截断."""
        step = {"tool": "bash", "params": {}, "output": "x" * 3000}
        prompt = build_judge_prompt(
            step=step,
            step_index=1,
            total_steps=1,
            context_summary="",
            rubrics=_model_rubrics(),
        )
        assert "(truncated)" in prompt


# ── _extract_json 测试 ────────────────────────────────────────────


class TestExtractJson:
    """测试 JSON 提取."""

    def test_pure_json(self):
        """纯 JSON 应直接解析."""
        data = _extract_json('{"scores": {"a": 0.8}, "rationale": "ok"}')
        assert data["scores"]["a"] == 0.8

    def test_markdown_code_block(self):
        """```json ``` 包裹应提取."""
        text = '这是评估结果:\n```json\n{"scores": {"a": 0.9}}\n```\n完毕。'
        data = _extract_json(text)
        assert data["scores"]["a"] == 0.9

    def test_code_block_no_lang(self):
        """``` ``` 无语言标注也应提取."""
        text = '```\n{"scores": {"a": 0.7}}\n```'
        data = _extract_json(text)
        assert data["scores"]["a"] == 0.7

    def test_mixed_text(self):
        """混合文本中的 JSON 应提取."""
        text = '好的，这是我的评估：{"scores": {"a": 0.6}, "rationale": "一般"}'
        data = _extract_json(text)
        assert data["scores"]["a"] == 0.6

    def test_invalid_json(self):
        """无法提取时应抛出 ValueError."""
        with pytest.raises(ValueError, match="无法从 LLM 响应中提取 JSON"):
            _extract_json("这不是 JSON 格式的内容")


# ── _parse_judgment 测试 ──────────────────────────────────────────


class TestParseJudgment:
    """测试 LLM 响应解析."""

    def test_valid_response(self):
        """合法响应应正确解析."""
        rubrics = _model_rubrics()
        response = json.dumps({
            "scores": {
                "goal_progress": 0.9,
                "tool_choice": 0.7,
                "param_correctness": 0.8,
            },
            "rationale": "步骤执行良好",
        })
        judgment = _parse_judgment(response, rubrics)

        assert judgment.rubric_scores["goal_progress"] == 0.9
        assert judgment.rubric_scores["tool_choice"] == 0.7
        assert judgment.rationale == "步骤执行良好"
        # 加权: (0.3*0.9 + 0.2*0.7 + 0.2*0.8) / 0.7
        expected = (0.3 * 0.9 + 0.2 * 0.7 + 0.2 * 0.8) / 0.7
        assert abs(judgment.score - expected) < 0.001

    def test_missing_rubric_fills_default(self):
        """缺少的 rubric 应用 0.5 填充."""
        rubrics = _model_rubrics()
        response = json.dumps({
            "scores": {"goal_progress": 0.9},
            "rationale": "只给了一个分数",
        })
        judgment = _parse_judgment(response, rubrics)

        assert judgment.rubric_scores["goal_progress"] == 0.9
        assert judgment.rubric_scores["tool_choice"] == 0.5
        assert judgment.rubric_scores["param_correctness"] == 0.5

    def test_score_clamped(self):
        """超出 [0, 1] 的分数应被截断."""
        rubrics = _model_rubrics()
        response = json.dumps({
            "scores": {
                "goal_progress": 1.5,
                "tool_choice": -0.3,
                "param_correctness": 0.8,
            },
            "rationale": "异常分数",
        })
        judgment = _parse_judgment(response, rubrics)

        assert judgment.rubric_scores["goal_progress"] == 1.0
        assert judgment.rubric_scores["tool_choice"] == 0.0


# ── _fallback_judgment 测试 ───────────────────────────────────────


class TestFallbackJudgment:
    """测试降级结果."""

    def test_fallback_scores(self):
        """降级应返回中性分数."""
        rubrics = _model_rubrics()
        judgment = _fallback_judgment(rubrics, "测试降级")

        assert judgment.score == 0.5
        for r in rubrics:
            assert judgment.rubric_scores[r.id] == 0.5
        assert "降级" in judgment.rationale
        assert "测试降级" in judgment.rationale


# ── judge_step 测试 (mock LLM) ───────────────────────────────────


class TestJudgeStepMock:
    """测试 judge_step 使用 mock LLM."""

    def test_with_anthropic_mock(self):
        """Mock Anthropic 调用应返回正确评分."""
        import agentreward.judge as judge_mod

        original = judge_mod._HAS_ANTHROPIC
        try:
            judge_mod._HAS_ANTHROPIC = True

            with patch.object(judge_mod, "_call_anthropic", return_value=_good_llm_response()):
                config = JudgeConfig(provider="anthropic")
                judgment = judge_step(
                    step=_sample_step(),
                    step_index=1,
                    total_steps=3,
                    context_summary="",
                    rubrics=_model_rubrics(),
                    config=config,
                    task_description="修复 bug",
                )

                assert judgment.score > 0.0
                assert judgment.rationale == "这一步很好地推进了目标"
                assert judgment.rubric_scores["goal_progress"] == 0.8
        finally:
            judge_mod._HAS_ANTHROPIC = original

    def test_with_openai_mock(self):
        """Mock OpenAI 调用应返回正确评分."""
        import agentreward.judge as judge_mod

        original = judge_mod._HAS_OPENAI
        try:
            judge_mod._HAS_OPENAI = True

            with patch.object(judge_mod, "_call_openai", return_value=_good_llm_response()):
                config = JudgeConfig(provider="openai", model="gpt-4o")
                judgment = judge_step(
                    step=_sample_step(),
                    step_index=1,
                    total_steps=1,
                    context_summary="",
                    rubrics=_model_rubrics(),
                    config=config,
                )

                assert judgment.score > 0.0
                assert judgment.rubric_scores["tool_choice"] == 0.8
        finally:
            judge_mod._HAS_OPENAI = original

    def test_no_llm_fallback(self):
        """LLM 库不可用时应降级."""
        import agentreward.judge as judge_mod

        orig_a = judge_mod._HAS_ANTHROPIC
        orig_o = judge_mod._HAS_OPENAI
        try:
            judge_mod._HAS_ANTHROPIC = False
            judge_mod._HAS_OPENAI = False

            config = JudgeConfig(provider="anthropic")
            judgment = judge_step(
                step=_sample_step(),
                step_index=1,
                total_steps=1,
                context_summary="",
                rubrics=_model_rubrics(),
                config=config,
            )

            assert judgment.score == 0.5
            assert "LLM 库未安装" in judgment.rationale
        finally:
            judge_mod._HAS_ANTHROPIC = orig_a
            judge_mod._HAS_OPENAI = orig_o

    def test_retry_on_failure(self):
        """LLM 调用失败时应重试."""
        import agentreward.judge as judge_mod

        original = judge_mod._HAS_ANTHROPIC
        try:
            judge_mod._HAS_ANTHROPIC = True
            call_count = 0

            def failing_then_ok(prompt, config):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("API 错误")
                return _good_llm_response()

            with patch.object(judge_mod, "_call_anthropic", side_effect=failing_then_ok):
                config = JudgeConfig(provider="anthropic", max_retries=3)
                judgment = judge_step(
                    step=_sample_step(),
                    step_index=1,
                    total_steps=1,
                    context_summary="",
                    rubrics=_model_rubrics(),
                    config=config,
                )

                assert judgment.score > 0.0
                assert call_count == 3
        finally:
            judge_mod._HAS_ANTHROPIC = original

    def test_all_retries_fail(self):
        """所有重试失败应降级."""
        import agentreward.judge as judge_mod

        original = judge_mod._HAS_ANTHROPIC
        try:
            judge_mod._HAS_ANTHROPIC = True

            with patch.object(
                judge_mod, "_call_anthropic",
                side_effect=RuntimeError("API 故障"),
            ):
                config = JudgeConfig(provider="anthropic", max_retries=2)
                judgment = judge_step(
                    step=_sample_step(),
                    step_index=1,
                    total_steps=1,
                    context_summary="",
                    rubrics=_model_rubrics(),
                    config=config,
                )

                assert judgment.score == 0.5
                assert "LLM 调用失败" in judgment.rationale
        finally:
            judge_mod._HAS_ANTHROPIC = original

    def test_unsupported_provider(self):
        """不支持的 provider 应降级."""
        config = JudgeConfig(provider="unsupported")
        judgment = judge_step(
            step=_sample_step(),
            step_index=1,
            total_steps=1,
            context_summary="",
            rubrics=_model_rubrics(),
            config=config,
        )
        assert judgment.score == 0.5
        assert "LLM 库未安装" in judgment.rationale


# ── judge_trajectory 测试 ─────────────────────────────────────────


class TestJudgeTrajectory:
    """测试轨迹级评估."""

    def test_empty_trajectory(self):
        """空轨迹应返回空列表."""
        judgments = judge_trajectory(
            trajectory={"task": "修复 bug", "steps": []},
            rubrics=_model_rubrics(),
        )
        assert judgments == []

    def test_multi_step_trajectory(self):
        """多步轨迹应返回对应数量的 judgment."""
        steps = [
            {"tool": "bash", "params": {"command": "ls"}, "output": "file.py"},
            {"tool": "read_file", "params": {"path": "file.py"}, "output": "code..."},
            {"tool": "edit_file", "params": {"path": "file.py"}, "output": "File edited"},
        ]
        judgments = judge_trajectory(
            trajectory={"task": "修复 bug", "steps": steps},
            rubrics=_model_rubrics(),
        )
        assert len(judgments) == 3
        for j in judgments:
            assert isinstance(j, StepJudgment)
            assert 0.0 <= j.score <= 1.0

    def test_context_accumulates(self):
        """后续步骤应包含前序上下文."""
        import agentreward.judge as judge_mod

        original = judge_mod._HAS_ANTHROPIC
        try:
            judge_mod._HAS_ANTHROPIC = True
            prompts_seen = []

            def capture_prompt(prompt, config):
                prompts_seen.append(prompt)
                return _good_llm_response()

            with patch.object(judge_mod, "_call_anthropic", side_effect=capture_prompt):
                steps = [
                    {"tool": "bash", "params": {}, "output": "ok"},
                    {"tool": "read_file", "params": {}, "output": "code"},
                ]
                config = JudgeConfig(provider="anthropic")
                judge_trajectory(
                    trajectory={"task": "测试", "steps": steps},
                    rubrics=_model_rubrics(),
                    config=config,
                )

                # 第二步的 prompt 应包含第一步的上下文
                assert len(prompts_seen) == 2
                assert "Step 1: bash" in prompts_seen[1]
        finally:
            judge_mod._HAS_ANTHROPIC = original
