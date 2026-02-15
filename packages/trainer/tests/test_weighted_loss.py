"""测试 weighted_cross_entropy loss."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttrainer.loss import weighted_cross_entropy


@pytest.fixture
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    return model, tok


class TestWeightedCrossEntropy:
    def test_uniform_weights_equals_standard(self, model_and_tokenizer):
        """权重全为 1.0 时应等于标准 CE loss."""
        model, tok = model_and_tokenizer
        text = "Hello world this is a test"
        ids = tok.encode(text, return_tensors="pt")
        labels = ids.clone()
        attention_mask = torch.ones_like(ids)

        # 标准 CE
        with torch.no_grad():
            standard_loss = model(input_ids=ids, labels=labels).loss

        # weighted CE with uniform weights
        weights = torch.ones_like(ids, dtype=torch.float32)
        with torch.no_grad():
            weighted_loss = weighted_cross_entropy(
                model, ids, labels, attention_mask, step_weights=weights
            )

        assert abs(standard_loss.item() - weighted_loss.item()) < 0.01

    def test_zero_weights_zero_loss(self, model_and_tokenizer):
        """权重全为 0 时 loss 应为 0."""
        model, tok = model_and_tokenizer
        text = "Hello world"
        ids = tok.encode(text, return_tensors="pt")
        labels = ids.clone()
        attention_mask = torch.ones_like(ids)
        weights = torch.zeros_like(ids, dtype=torch.float32)

        with torch.no_grad():
            loss = weighted_cross_entropy(
                model, ids, labels, attention_mask, step_weights=weights
            )

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_higher_weights_higher_loss(self, model_and_tokenizer):
        """更高权重应增大 loss（当权重不均匀时）."""
        model, tok = model_and_tokenizer
        text = "Hello world this is a test"
        ids = tok.encode(text, return_tensors="pt")
        labels = ids.clone()
        attention_mask = torch.ones_like(ids)

        weights_low = torch.ones_like(ids, dtype=torch.float32) * 0.5
        weights_high = torch.ones_like(ids, dtype=torch.float32) * 2.0

        with torch.no_grad():
            loss_low = weighted_cross_entropy(
                model, ids, labels, attention_mask, step_weights=weights_low
            )
            loss_high = weighted_cross_entropy(
                model, ids, labels, attention_mask, step_weights=weights_high
            )

        assert loss_high.item() > loss_low.item()

    def test_none_weights_works(self, model_and_tokenizer):
        """step_weights=None 应等于标准 CE."""
        model, tok = model_and_tokenizer
        text = "Hello world"
        ids = tok.encode(text, return_tensors="pt")
        labels = ids.clone()
        attention_mask = torch.ones_like(ids)

        with torch.no_grad():
            loss = weighted_cross_entropy(
                model, ids, labels, attention_mask, step_weights=None
            )

        assert loss.item() > 0
