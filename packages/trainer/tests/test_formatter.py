"""测试 data.formatter - chat template 格式化."""

import pytest
import torch
from transformers import AutoTokenizer

from agenttrainer.data.formatter import format_sft, format_dpo, format_grpo


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


class TestFormatSFT:
    def test_returns_required_keys(self, tokenizer):
        result = format_sft(tokenizer, "Fix bug", "", "I fixed it", max_length=128)
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

    def test_input_ids_are_tensor(self, tokenizer):
        result = format_sft(tokenizer, "Fix bug", "", "I fixed it", max_length=128)
        assert isinstance(result["input_ids"], torch.Tensor)
        assert result["input_ids"].dim() == 1

    def test_labels_mask_prompt(self, tokenizer):
        result = format_sft(tokenizer, "Fix bug", "", "I fixed it", max_length=128)
        # labels 前面部分应该是 -100 (prompt masked)
        assert (result["labels"] == -100).any()
        # labels 后面部分应该有非 -100 的值 (response)
        assert (result["labels"] != -100).any()

    def test_max_length_truncation(self, tokenizer):
        long_response = "word " * 1000
        result = format_sft(tokenizer, "Fix bug", "", long_response, max_length=64)
        assert result["input_ids"].size(0) <= 64


class TestFormatDPO:
    def test_returns_required_keys(self, tokenizer):
        result = format_dpo(tokenizer, "Fix bug", "good response", "bad response", max_length=128)
        assert "input_ids_chosen" in result
        assert "input_ids_rejected" in result
        assert "labels_chosen" in result
        assert "labels_rejected" in result

    def test_chosen_rejected_different(self, tokenizer):
        result = format_dpo(tokenizer, "Fix bug", "good fix", "bad fix", max_length=128)
        # chosen 和 rejected 应该不完全相同
        assert not torch.equal(result["input_ids_chosen"], result["input_ids_rejected"])


class TestFormatGRPO:
    def test_returns_required_keys(self, tokenizer):
        result = format_grpo(tokenizer, "Fix bug", "I fixed it", max_length=128)
        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

    def test_labels_mask_prompt(self, tokenizer):
        result = format_grpo(tokenizer, "Fix bug", "I fixed it", max_length=128)
        assert (result["labels"] == -100).any()
        assert (result["labels"] != -100).any()
