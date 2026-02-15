"""AgentTrainer CLI - 命令行界面."""

from __future__ import annotations

import json
from pathlib import Path

import click

from agenttrainer import __version__
from agenttrainer.config import SFTConfig, DPOConfig, GRPOConfig


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-trainer")
def main():
    """knowlyr-trainer - Agent 轨迹训练工具

    支持 SFT / DPO / GRPO 三种训练方法，无缝对接 knowlyr-hub 导出的数据集。
    """
    pass


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML 配置文件")
@click.option("--train-file", type=click.Path(exists=True), help="SFT 训练数据 (JSONL)")
@click.option("--model", "model_name", type=str, help="模型名称或路径")
@click.option("--output-dir", type=str, help="输出目录")
@click.option("--epochs", type=int, help="训练轮数")
@click.option("--batch-size", type=int, help="每设备 batch size")
@click.option("--lr", type=float, help="学习率")
@click.option("--max-length", type=int, help="最大序列长度")
@click.option("--lora/--no-lora", default=None, help="是否使用 LoRA")
@click.option("--bf16/--no-bf16", default=None, help="是否使用 bfloat16")
@click.option("--gradient-checkpointing/--no-gradient-checkpointing", default=None)
@click.option("--wandb-project", type=str, help="wandb 项目名")
@click.option("--seed", type=int, help="随机种子")
def sft(config_path, train_file, model_name, output_dir, epochs, batch_size, lr,
        max_length, lora, bf16, gradient_checkpointing, wandb_project, seed):
    """运行 SFT (Supervised Fine-Tuning) 训练.

    读取 knowlyr-hub export --format sft 导出的 JSONL 文件。
    """
    if config_path:
        cfg = SFTConfig.from_yaml(config_path)
    else:
        cfg = SFTConfig()

    # CLI 参数覆盖
    cfg = cfg.merge_cli(
        train_file=train_file,
        model_name_or_path=model_name,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        max_length=max_length,
        use_lora=lora,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        wandb_project=wandb_project,
        seed=seed,
    )

    if not cfg.train_file:
        raise click.UsageError("请指定 --train-file 或在配置文件中设置 train_file")

    click.echo("启动 SFT 训练")
    click.echo(f"  模型: {cfg.model_name_or_path}")
    click.echo(f"  数据: {cfg.train_file}")
    click.echo(f"  输出: {cfg.output_dir}")

    from agenttrainer.trainers.sft import SFTTrainer

    trainer = SFTTrainer(cfg)
    trainer.train()
    click.echo(f"训练完成! 模型: {cfg.output_dir}/final")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML 配置文件")
@click.option("--train-file", type=click.Path(exists=True), help="DPO 训练数据 (JSONL)")
@click.option("--model", "model_name", type=str, help="模型名称或路径")
@click.option("--output-dir", type=str, help="输出目录")
@click.option("--epochs", type=int, help="训练轮数")
@click.option("--batch-size", type=int, help="每设备 batch size")
@click.option("--lr", type=float, help="学习率")
@click.option("--beta", type=float, help="DPO beta 参数")
@click.option("--lora/--no-lora", default=None, help="是否使用 LoRA")
@click.option("--bf16/--no-bf16", default=None, help="是否使用 bfloat16")
@click.option("--wandb-project", type=str, help="wandb 项目名")
def dpo(config_path, train_file, model_name, output_dir, epochs, batch_size, lr,
        beta, lora, bf16, wandb_project):
    """运行 DPO (Direct Preference Optimization) 训练.

    读取 knowlyr-hub export --format dpo 导出的 JSONL 文件。
    """
    if config_path:
        cfg = DPOConfig.from_yaml(config_path)
    else:
        cfg = DPOConfig()

    cfg = cfg.merge_cli(
        train_file=train_file,
        model_name_or_path=model_name,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        beta=beta,
        use_lora=lora,
        bf16=bf16,
        wandb_project=wandb_project,
    )

    if not cfg.train_file:
        raise click.UsageError("请指定 --train-file 或在配置文件中设置 train_file")

    click.echo("启动 DPO 训练")
    click.echo(f"  模型: {cfg.model_name_or_path}")
    click.echo(f"  数据: {cfg.train_file}")
    click.echo(f"  Beta: {cfg.beta}")

    from agenttrainer.trainers.dpo import DPOTrainer

    trainer = DPOTrainer(cfg)
    trainer.train()
    click.echo(f"训练完成! 模型: {cfg.output_dir}/final")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML 配置文件")
@click.option("--train-file", type=click.Path(exists=True), help="GRPO 分组数据 (JSONL)")
@click.option("--model", "model_name", type=str, help="模型名称或路径")
@click.option("--output-dir", type=str, help="输出目录")
@click.option("--epochs", type=int, help="训练轮数")
@click.option("--group-size", type=int, help="每组轨迹数量")
@click.option("--clip-epsilon", type=float, help="PPO clip 范围")
@click.option("--kl-coef", type=float, help="KL penalty 系数")
@click.option("--lora/--no-lora", default=None, help="是否使用 LoRA")
@click.option("--bf16/--no-bf16", default=None, help="是否使用 bfloat16")
@click.option("--wandb-project", type=str, help="wandb 项目名")
def grpo(config_path, train_file, model_name, output_dir, epochs, group_size,
         clip_epsilon, kl_coef, lora, bf16, wandb_project):
    """运行 GRPO (Group Relative Policy Optimization) 训练.

    Phase 1: 离线模式，读取 knowlyr-hub export --format grpo 导出的分组数据。
    """
    if config_path:
        cfg = GRPOConfig.from_yaml(config_path)
    else:
        cfg = GRPOConfig()

    cfg = cfg.merge_cli(
        train_file=train_file,
        model_name_or_path=model_name,
        output_dir=output_dir,
        num_epochs=epochs,
        group_size=group_size,
        clip_epsilon=clip_epsilon,
        kl_coef=kl_coef,
        use_lora=lora,
        bf16=bf16,
        wandb_project=wandb_project,
    )

    if not cfg.train_file:
        raise click.UsageError("请指定 --train-file 或在配置文件中设置 train_file")

    click.echo("启动 GRPO 训练 (离线模式)")
    click.echo(f"  模型: {cfg.model_name_or_path}")
    click.echo(f"  数据: {cfg.train_file}")
    click.echo(f"  Group size: {cfg.group_size}")

    from agenttrainer.trainers.grpo import GRPOTrainer

    trainer = GRPOTrainer(cfg)
    trainer.train()
    click.echo(f"训练完成! 模型: {cfg.output_dir}/final")


@main.command("eval")
@click.option("--model", "model_path", type=click.Path(exists=True), required=True, help="模型路径")
@click.option("--eval-file", type=click.Path(exists=True), required=True, help="评估数据 (JSONL)")
@click.option("-o", "--output", type=click.Path(), default="./eval_results.json", help="结果输出")
@click.option("--max-length", type=int, default=2048, help="最大序列长度")
@click.option("--batch-size", type=int, default=4, help="Batch size")
def eval_cmd(model_path, eval_file, output, max_length, batch_size):
    """评估模型.

    计算 perplexity 和 token accuracy。
    """
    click.echo(f"评估模型: {model_path}")
    click.echo(f"数据: {eval_file}")

    from agenttrainer.eval.evaluator import evaluate

    results = evaluate(
        model_path=model_path,
        eval_file=eval_file,
        max_length=max_length,
        batch_size=batch_size,
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    click.echo(f"Perplexity: {results.get('perplexity', 'N/A'):.4f}")
    click.echo(f"Token Accuracy: {results.get('token_accuracy', 'N/A'):.4f}")
    click.echo(f"结果: {output}")


if __name__ == "__main__":
    main()
