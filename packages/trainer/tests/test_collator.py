"""测试 data.collator - padding 批次整理."""

import pytest
import torch
from transformers import AutoTokenizer

from agenttrainer.data.collator import SFTCollator, DPOCollator, GRPOCollator


def _get_tokenizer():
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


class TestSFTCollator:
    def test_pad_to_same_length(self):
        tokenizer = _get_tokenizer()
        collator = SFTCollator(tokenizer, max_length=128)

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
                "attention_mask": torch.ones(3),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7, 8]),
                "labels": torch.tensor([-100, -100, 6, 7, 8]),
                "attention_mask": torch.ones(5),
            },
        ]

        result = collator(batch)
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)
        assert result["attention_mask"].shape == (2, 5)

    def test_pad_value(self):
        tokenizer = _get_tokenizer()
        collator = SFTCollator(tokenizer, max_length=128)

        batch = [
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, 2])},
            {"input_ids": torch.tensor([3, 4, 5]), "labels": torch.tensor([-100, 4, 5])},
        ]

        result = collator(batch)
        # 第一条样本应该被 pad
        assert result["labels"][0, 2].item() == -100  # label pad = -100


class TestDPOCollator:
    def test_pad_chosen_and_rejected(self):
        tokenizer = _get_tokenizer()
        collator = DPOCollator(tokenizer, max_length=128)

        batch = [
            {
                "input_ids_chosen": torch.tensor([1, 2, 3]),
                "labels_chosen": torch.tensor([-100, 2, 3]),
                "attention_mask_chosen": torch.ones(3),
                "input_ids_rejected": torch.tensor([4, 5]),
                "labels_rejected": torch.tensor([-100, 5]),
                "attention_mask_rejected": torch.ones(2),
            },
        ]

        result = collator(batch)
        assert "input_ids_chosen" in result
        assert "input_ids_rejected" in result


class TestGRPOCollator:
    def test_pad_group(self):
        tokenizer = _get_tokenizer()
        collator = GRPOCollator(tokenizer, max_length=128)

        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]), "reward": 0.8},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([-100, 5]), "reward": 0.3},
        ]

        result = collator(batch)
        assert result["input_ids"].shape[0] == 2
        assert result["rewards"].shape == (2,)
        assert result["rewards"][0].item() == pytest.approx(0.8, abs=1e-6)
