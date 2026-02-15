"""模型评估 - perplexity 和 token accuracy."""

from __future__ import annotations

import logging
import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from agenttrainer.data.reader import read_sft
from agenttrainer.data.formatter import format_sft
from agenttrainer.data.collator import SFTCollator
from agenttrainer.models.loader import load_model

logger = logging.getLogger(__name__)


class _EvalDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        return format_sft(
            self.tokenizer,
            instruction=rec.get("instruction", ""),
            input_text=rec.get("input", ""),
            response=rec.get("response", ""),
            max_length=self.max_length,
        )


def evaluate(
    model_path: str,
    eval_file: str,
    max_length: int = 2048,
    batch_size: int = 4,
) -> dict:
    """评估模型.

    Args:
        model_path: 模型路径
        eval_file: 评估数据 JSONL (SFT 格式)
        max_length: 最大序列长度
        batch_size: batch size

    Returns:
        {"perplexity": float, "token_accuracy": float, "total_tokens": int, "samples": int}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(model_path, bf16=True)
    model.to(device)
    model.eval()

    records = read_sft(eval_file)
    dataset = _EvalDataset(records, tokenizer, max_length)
    collator = SFTCollator(tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估中"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            # Loss (已经是 mean over non-ignored tokens)
            mask = (batch["labels"] != -100)
            n_tokens = mask.sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

            # Token accuracy
            logits = outputs.logits[:, :-1, :]
            preds = logits.argmax(dim=-1)
            labels = batch["labels"][:, 1:]
            label_mask = (labels != -100)
            correct_tokens += ((preds == labels) & label_mask).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # cap 防止溢出
    accuracy = correct_tokens / max(total_tokens, 1)

    results = {
        "perplexity": perplexity,
        "token_accuracy": accuracy,
        "total_tokens": total_tokens,
        "samples": len(records),
        "avg_loss": avg_loss,
    }

    logger.info("评估结果: ppl=%.2f, acc=%.4f, tokens=%d", perplexity, accuracy, total_tokens)
    return results
