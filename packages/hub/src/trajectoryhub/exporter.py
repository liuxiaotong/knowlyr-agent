"""Dataset export - 数据集导出.

支持将轨迹数据导出为多种训练格式：
- SFT: 监督微调格式 (instruction/response 对)
- DPO: 偏好学习格式 (chosen/rejected 对)
- Benchmark: 评测基准格式
- HuggingFace: 推送到 HuggingFace Hub
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """导出结果.

    Attributes:
        success: 是否成功
        output_path: 输出文件路径
        total_records: 导出记录数
        format: 导出格式
        error: 错误信息 (如果失败)
    """

    success: bool = False
    output_path: str = ""
    total_records: int = 0
    format: str = ""
    error: Optional[str] = None


class DatasetExporter:
    """数据集导出器.

    将 Pipeline 产出的轨迹和偏好对导出为各种训练格式。

    Usage:
        exporter = DatasetExporter(
            trajectories_dir="./output/trajectories.jsonl",
            preferences_dir="./output/preferences.jsonl",
        )

        # 导出 SFT 格式
        result = exporter.export_sft("./export/sft_train.jsonl")

        # 导出 DPO 格式
        result = exporter.export_dpo("./export/dpo_train.jsonl")
    """

    def __init__(
        self,
        trajectories_dir: str,
        preferences_dir: Optional[str] = None,
    ) -> None:
        self.trajectories_dir = Path(trajectories_dir)
        self.preferences_dir = Path(preferences_dir) if preferences_dir else None

    def export_sft(self, output_path: str) -> ExportResult:
        """导出为 SFT 训练格式.

        将成功的轨迹转换为 instruction/response 对，适用于监督微调。

        SFT 格式 (每行一个 JSON):
        ```json
        {
            "instruction": "任务描述",
            "input": "任务上下文 (repo, commit 等)",
            "response": "Agent 执行步骤序列",
            "task_id": "原始任务 ID",
            "reward": 0.85,
            "metadata": {...}
        }
        ```

        只导出成功的轨迹 (success=True)，按 reward 从高到低排序。

        Args:
            output_path: 输出文件路径

        Returns:
            ExportResult: 导出结果
        """
        try:
            trajectories = self._load_trajectories()
            logger.info("导出 SFT: 加载 %d 条轨迹", len(trajectories))
            # 只选成功的轨迹，按 reward 排序
            successful = sorted(
                [t for t in trajectories if t.get("success", False)],
                key=lambda t: t.get("reward", 0.0),
                reverse=True,
            )

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, "w", encoding="utf-8") as f:
                for traj in successful:
                    # 将步骤序列转为 response 文本
                    response_parts = []
                    for i, step in enumerate(traj.get("steps", []), 1):
                        action = step.get("action", "")
                        observation = step.get("observation", "")
                        thought = step.get("thought", "")

                        step_text = f"Step {i}:"
                        if thought:
                            step_text += f"\nThought: {thought}"
                        if action:
                            step_text += f"\nAction: {action}"
                        if observation:
                            step_text += f"\nObservation: {observation}"
                        response_parts.append(step_text)

                    metadata = traj.get("metadata", {})
                    record = {
                        "instruction": metadata.get(
                            "task_description", f"Solve task: {traj.get('task_id', '')}"
                        ),
                        "input": json.dumps(
                            {
                                "repo": metadata.get("repo", ""),
                                "base_commit": metadata.get("base_commit", ""),
                                "test_command": metadata.get("test_command", ""),
                            },
                            ensure_ascii=False,
                        ),
                        "response": "\n\n".join(response_parts) if response_parts else "",
                        "task_id": traj.get("task_id", ""),
                        "reward": traj.get("reward", 0.0),
                        "metadata": {
                            "agent_framework": traj.get("agent_framework", ""),
                            "agent_model": traj.get("agent_model", ""),
                            "total_steps": traj.get("total_steps", 0),
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info("SFT 导出完成: %d 条记录 -> %s", len(successful), output)
            return ExportResult(
                success=True,
                output_path=str(output),
                total_records=len(successful),
                format="sft",
            )

        except Exception as e:
            logger.exception("SFT 导出失败")
            return ExportResult(
                success=False,
                format="sft",
                error=str(e),
            )

    def export_dpo(self, output_path: str) -> ExportResult:
        """导出为 DPO 训练格式.

        将偏好对转换为 DPO 训练格式。

        DPO 格式 (每行一个 JSON):
        ```json
        {
            "prompt": "任务描述",
            "chosen": "高 reward 的执行轨迹",
            "rejected": "低 reward 的执行轨迹",
            "task_id": "原始任务 ID",
            "reward_margin": 0.3,
            "metadata": {...}
        }
        ```

        Args:
            output_path: 输出文件路径

        Returns:
            ExportResult: 导出结果
        """
        if not self.preferences_dir or not self.preferences_dir.exists():
            return ExportResult(
                success=False,
                format="dpo",
                error="偏好对文件不存在，请先运行 Pipeline 生成偏好对",
            )

        try:
            preferences = self._load_preferences()
            logger.info("导出 DPO: 加载 %d 个偏好对", len(preferences))

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, "w", encoding="utf-8") as f:
                for pref in preferences:
                    # 将步骤序列转为文本
                    chosen_text = self._steps_to_text(pref.get("chosen", {}).get("steps", []))
                    rejected_text = self._steps_to_text(
                        pref.get("rejected", {}).get("steps", [])
                    )

                    record = {
                        "prompt": f"Solve the following task:\n\nTask ID: {pref.get('task_id', '')}",
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                        "task_id": pref.get("task_id", ""),
                        "reward_margin": pref.get("reward_margin", 0.0),
                        "metadata": {
                            "chosen_model": pref.get("chosen", {}).get("agent_model", ""),
                            "rejected_model": pref.get("rejected", {}).get("agent_model", ""),
                            "chosen_reward": pref.get("chosen", {}).get("reward", 0.0),
                            "rejected_reward": pref.get("rejected", {}).get("reward", 0.0),
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info("DPO 导出完成: %d 条记录 -> %s", len(preferences), output)
            return ExportResult(
                success=True,
                output_path=str(output),
                total_records=len(preferences),
                format="dpo",
            )

        except Exception as e:
            logger.exception("DPO 导出失败")
            return ExportResult(
                success=False,
                format="dpo",
                error=str(e),
            )

    def export_benchmark(self, output_path: str) -> ExportResult:
        """导出为评测基准格式.

        将任务和轨迹导出为可复现的评测基准格式。

        Benchmark 格式:
        ```json
        {
            "task_id": "...",
            "description": "...",
            "repo": "...",
            "base_commit": "...",
            "test_command": "...",
            "reference_trajectories": [...],
            "difficulty": "medium",
            "expected_reward_range": [0.6, 0.9]
        }
        ```

        Args:
            output_path: 输出文件路径

        Returns:
            ExportResult: 导出结果
        """
        try:
            trajectories = self._load_trajectories()
            logger.info("导出 Benchmark: 加载 %d 条轨迹", len(trajectories))

            # 按 task_id 分组
            task_groups: Dict[str, List[Dict[str, Any]]] = {}
            for traj in trajectories:
                tid = traj.get("task_id", "")
                if tid not in task_groups:
                    task_groups[tid] = []
                task_groups[tid].append(traj)

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            records = []
            with open(output, "w", encoding="utf-8") as f:
                for task_id, group in task_groups.items():
                    rewards = [t.get("reward", 0.0) for t in group]
                    metadata = group[0].get("metadata", {})

                    record = {
                        "task_id": task_id,
                        "description": metadata.get("task_description", ""),
                        "repo": metadata.get("repo", ""),
                        "base_commit": metadata.get("base_commit", ""),
                        "test_command": metadata.get("test_command", ""),
                        "reference_trajectories": [
                            {
                                "agent_model": t.get("agent_model", ""),
                                "reward": t.get("reward", 0.0),
                                "success": t.get("success", False),
                                "total_steps": t.get("total_steps", 0),
                            }
                            for t in group
                        ],
                        "difficulty": metadata.get("difficulty", "unknown"),
                        "expected_reward_range": [
                            round(min(rewards), 2) if rewards else 0.0,
                            round(max(rewards), 2) if rewards else 0.0,
                        ],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)

            logger.info("Benchmark 导出完成: %d 条记录 -> %s", len(records), output)
            return ExportResult(
                success=True,
                output_path=str(output),
                total_records=len(records),
                format="benchmark",
            )

        except Exception as e:
            logger.exception("Benchmark 导出失败")
            return ExportResult(
                success=False,
                format="benchmark",
                error=str(e),
            )

    def export_huggingface(self, repo_id: str) -> ExportResult:
        """推送数据集到 HuggingFace Hub.

        将轨迹和偏好对打包上传到 HuggingFace，附带自动生成的 Dataset Card。

        Args:
            repo_id: HuggingFace 仓库 ID (e.g. "username/dataset-name")

        Returns:
            ExportResult: 导出结果
        """
        # TODO: 接入 huggingface_hub 库上传
        # from huggingface_hub import HfApi
        # api = HfApi()
        # api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        # api.upload_file(...)

        return ExportResult(
            success=False,
            format="huggingface",
            error="HuggingFace 导出功能待实现。请先安装: pip install huggingface_hub",
        )

    def generate_datacard(self) -> str:
        """生成数据集卡片 (Dataset Card).

        Returns:
            str: Markdown 格式的数据集说明
        """
        trajectories = self._load_trajectories()

        total = len(trajectories)
        successful = sum(1 for t in trajectories if t.get("success", False))
        avg_reward = (
            sum(t.get("reward", 0.0) for t in trajectories) / total if total > 0 else 0.0
        )
        avg_steps = (
            sum(t.get("total_steps", 0) for t in trajectories) / total if total > 0 else 0.0
        )

        models = set(t.get("agent_model", "") for t in trajectories)
        frameworks = set(t.get("agent_framework", "") for t in trajectories)

        card = f"""---
language:
- en
- zh
tags:
- agent-trajectory
- code-agent
- sft
- dpo
- reinforcement-learning
license: mit
---

# Agent Trajectory Dataset

## Dataset Description

Agent 执行轨迹数据集，包含 Code Agent 在真实代码仓库上的执行过程记录。
每条轨迹包含完整的步骤序列、过程级 Reward 打分和偏好对。

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total trajectories | {total} |
| Successful | {successful} |
| Success rate | {successful / total * 100:.1f}% |
| Average reward | {avg_reward:.3f} |
| Average steps | {avg_steps:.1f} |
| Agent models | {', '.join(models)} |
| Agent frameworks | {', '.join(frameworks)} |

## Data Format

### Trajectories (SFT)

Each record contains:
- `instruction`: Task description
- `input`: Task context (repo, commit, test command)
- `response`: Agent execution steps
- `reward`: Process-level reward score

### Preferences (DPO)

Each record contains:
- `prompt`: Task description
- `chosen`: Higher-reward trajectory
- `rejected`: Lower-reward trajectory
- `reward_margin`: Reward difference

## Usage

```python
from datasets import load_dataset
dataset = load_dataset("path/to/dataset")
```

## License

MIT

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{agent-trajectory-hub,
    title={{Agent Trajectory Hub}},
    author={{Liu Kai}},
    year={{2026}},
    url={{https://github.com/liuxiaotong/agent-trajectory-hub}}
}}
```
"""
        return card

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载轨迹."""
        records = []
        if not self.trajectories_dir.exists():
            return records

        with open(self.trajectories_dir, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _load_preferences(self) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载偏好对."""
        records = []
        if not self.preferences_dir or not self.preferences_dir.exists():
            return records

        with open(self.preferences_dir, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _steps_to_text(self, steps: List[Dict[str, Any]]) -> str:
        """将步骤列表转为文本格式."""
        parts = []
        for i, step in enumerate(steps, 1):
            action = step.get("action", "")
            observation = step.get("observation", "")
            thought = step.get("thought", "")

            step_text = f"Step {i}:"
            if thought:
                step_text += f"\nThought: {thought}"
            if action:
                step_text += f"\nAction: {action}"
            if observation:
                step_text += f"\nObservation: {observation}"
            parts.append(step_text)

        return "\n\n".join(parts) if parts else ""
