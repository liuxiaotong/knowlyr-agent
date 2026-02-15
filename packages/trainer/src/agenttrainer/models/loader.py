"""模型加载 - HuggingFace / 本地路径，可选 LoRA."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_model(
    name_or_path: str,
    *,
    tokenizer_name: str | None = None,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    device_map: str | None = None,
) -> tuple[Any, PreTrainedTokenizer]:
    """加载模型和 tokenizer.

    Args:
        name_or_path: HuggingFace 模型名或本地路径
        tokenizer_name: tokenizer 名，默认同 name_or_path
        bf16: 是否使用 bfloat16
        gradient_checkpointing: 是否启用梯度检查点
        use_lora: 是否使用 LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: LoRA 目标模块
        device_map: 设备映射（多 GPU 时使用）

    Returns:
        (model, tokenizer) 元组
    """
    dtype = torch.bfloat16 if bf16 else torch.float32

    logger.info("加载模型: %s (dtype=%s)", name_or_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("已启用梯度检查点")

    # tokenizer
    tok_name = tokenizer_name or name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("设置 pad_token = eos_token (%s)", tokenizer.pad_token)

    # LoRA
    if use_lora:
        model = _apply_lora(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "参数量: %.1fM 总 / %.1fM 可训练 (%.1f%%)",
        total_params / 1e6,
        trainable_params / 1e6,
        100 * trainable_params / total_params,
    )

    return model, tokenizer


def _apply_lora(
    model: Any,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> Any:
    """应用 LoRA 适配器."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("使用 LoRA 需要安装 peft: pip install knowlyr-trainer[peft]") from e

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    logger.info("已应用 LoRA (r=%d, alpha=%d, modules=%s)", r, alpha, target_modules)
    return model
