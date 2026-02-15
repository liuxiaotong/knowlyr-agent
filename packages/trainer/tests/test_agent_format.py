"""测试 data.agent_format — Agent 轨迹格式化 + 观察遮蔽 + 步骤加权."""

import pytest
import torch
from transformers import AutoTokenizer

from agenttrainer.data.agent_format import (
    AgentStep,
    parse_trajectory,
    parse_structured_steps,
    build_agent_messages,
    format_agent_sft,
)


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


# ── parse_trajectory ──────────────────────────────────────


class TestParseTrajectory:
    def test_parse_basic(self):
        text = (
            "Step 1:\n"
            "Thought: Let me read the file\n"
            "Action: read_file /sort.py\n"
            "Observation: def sort(arr): pass\n\n"
            "Step 2:\n"
            "Thought: Fix the bug\n"
            "Action: edit_file /sort.py\n"
            "Observation: File edited"
        )
        steps = parse_trajectory(text)
        assert len(steps) == 2
        assert steps[0].thought == "Let me read the file"
        assert steps[0].action == "read_file /sort.py"
        assert steps[0].observation == "def sort(arr): pass"
        assert steps[1].thought == "Fix the bug"

    def test_parse_no_thought(self):
        text = "Step 1:\nAction: bash ls\nObservation: file1.py"
        steps = parse_trajectory(text)
        assert len(steps) == 1
        assert steps[0].thought == ""
        assert steps[0].action == "bash ls"

    def test_parse_empty_string(self):
        steps = parse_trajectory("")
        assert len(steps) == 0

    def test_parse_no_steps_format(self):
        text = "Just some plain text without step markers"
        steps = parse_trajectory(text)
        assert len(steps) == 0


# ── parse_structured_steps ────────────────────────────────


class TestParseStructuredSteps:
    def test_parse_structured(self):
        data = [
            {"thought": "Read file", "action": "read_file /a.py", "observation": "code...", "reward": 0.7},
            {"thought": "Fix bug", "action": "edit_file /a.py", "observation": "Done", "reward": 0.9},
        ]
        steps = parse_structured_steps(data)
        assert len(steps) == 2
        assert steps[0].thought == "Read file"
        assert steps[0].reward == 0.7
        assert steps[1].action == "edit_file /a.py"

    def test_parse_missing_fields(self):
        data = [{"action": "bash ls"}]
        steps = parse_structured_steps(data)
        assert len(steps) == 1
        assert steps[0].thought == ""
        assert steps[0].observation == ""
        assert steps[0].reward == 0.0


# ── build_agent_messages ──────────────────────────────────


class TestBuildAgentMessages:
    def test_basic_messages(self):
        steps = [
            AgentStep(thought="Read it", action="read_file /a.py", observation="code here"),
            AgentStep(thought="Fix it", action="edit_file /a.py", observation="Done"),
        ]
        messages = build_agent_messages("Fix bug", "", steps)
        # user (task) + assistant (step1) + user (obs1) + assistant (step2) + user (obs2)
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert "Observation:" in messages[2]["content"]
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_with_input_text(self):
        steps = [AgentStep(thought="Think", action="act")]
        messages = build_agent_messages("Fix bug", "context info", steps)
        assert "context info" in messages[0]["content"]

    def test_no_observation(self):
        steps = [AgentStep(thought="Think", action="act")]
        messages = build_agent_messages("Task", "", steps)
        # user (task) + assistant (step1) — 无 observation
        assert len(messages) == 2


# ── format_agent_sft ──────────────────────────────────────


class TestFormatAgentSFT:
    def test_returns_required_keys(self, tokenizer):
        steps = [
            AgentStep(thought="Read", action="read_file /a.py", observation="code"),
            AgentStep(thought="Fix", action="edit_file /a.py", observation="done"),
        ]
        result = format_agent_sft(
            tokenizer, "Fix bug", "", steps, max_length=256
        )
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)

    def test_observation_masking(self, tokenizer):
        steps = [
            AgentStep(thought="Read", action="read_file /a.py", observation="code here"),
            AgentStep(thought="Fix", action="edit_file /a.py", observation="edited"),
        ]
        result = format_agent_sft(
            tokenizer, "Fix bug", "", steps, max_length=256, mask_observations=True
        )
        labels = result["labels"]
        # 应该有 -100 (masked) 和非 -100 (trainable)
        assert (labels == -100).any(), "应该有被遮蔽的 token"
        assert (labels != -100).any(), "应该有参与训练的 token"

    def test_no_observation_masking(self, tokenizer):
        steps = [
            AgentStep(thought="Read", action="read_file /a.py", observation="code"),
        ]
        result_masked = format_agent_sft(
            tokenizer, "Fix bug", "", steps, max_length=256, mask_observations=True
        )
        result_unmasked = format_agent_sft(
            tokenizer, "Fix bug", "", steps, max_length=256, mask_observations=False
        )
        # 不遮蔽模式下，更多 token 参与训练
        n_train_masked = (result_masked["labels"] != -100).sum().item()
        n_train_unmasked = (result_unmasked["labels"] != -100).sum().item()
        assert n_train_unmasked >= n_train_masked

    def test_step_weights(self, tokenizer):
        steps = [
            AgentStep(thought="Read", action="read_file", observation="code"),
            AgentStep(thought="Fix", action="edit_file", observation="done"),
        ]
        result = format_agent_sft(
            tokenizer, "Fix bug", "", steps, max_length=256,
            step_rewards=[0.5, 1.0],
        )
        assert "step_weights" in result
        assert isinstance(result["step_weights"], torch.Tensor)
        assert result["step_weights"].shape == result["input_ids"].shape

    def test_max_length_respected(self, tokenizer):
        steps = [
            AgentStep(
                thought="Think " * 100,
                action="act " * 100,
                observation="obs " * 100,
            ),
        ]
        result = format_agent_sft(
            tokenizer, "Task", "", steps, max_length=64
        )
        assert result["input_ids"].size(0) <= 64
