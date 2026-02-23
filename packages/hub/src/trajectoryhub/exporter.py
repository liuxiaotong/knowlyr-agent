"""Dataset export - 数据集导出.

支持将轨迹数据导出为多种训练格式：
- SFT: 监督微调格式 (instruction/response 对)
- DPO: 偏好学习格式 (chosen/rejected 对)
- Benchmark: 评测基准格式
- HuggingFace: 推送到 HuggingFace Hub

Phase 3 新增：
- export_sft_split / export_dpo_split: 按比例分割导出 train/val/test
"""

import json
import logging
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi

    _HAS_HF = True
except ImportError:
    _HAS_HF = False


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
        # 从 JSONL 文件
        exporter = DatasetExporter(
            trajectories_dir="./output/trajectories.jsonl",
            preferences_dir="./output/preferences.jsonl",
        )

        # 从 CAS 存储（按 GDI 排名导出）
        exporter = DatasetExporter.from_store("./data/index.sqlite")

        # 导出 SFT 格式
        result = exporter.export_sft("./export/sft_train.jsonl")
    """

    def __init__(
        self,
        trajectories_dir: str,
        preferences_dir: Optional[str] = None,
    ) -> None:
        self.trajectories_dir = Path(trajectories_dir)
        self.preferences_dir = Path(preferences_dir) if preferences_dir else None
        self._store = None  # CAStore 实例（from_store 模式）

    @classmethod
    def from_store(
        cls,
        store_path: str,
        *,
        min_gdi: float = 0.0,
        limit: int = 10000,
    ) -> "DatasetExporter":
        """从 CAS 存储创建导出器（按 GDI 排名）.

        Args:
            store_path: SQLite 存储路径.
            min_gdi: 最低 GDI 阈值，低于此值的轨迹不导出.
            limit: 最多导出条数.

        Returns:
            DatasetExporter 实例.
        """
        from trajectoryhub.cas import CAStore

        instance = cls.__new__(cls)
        instance.trajectories_dir = Path(store_path)
        instance.preferences_dir = None
        instance._store = CAStore(store_path)
        instance._store_min_gdi = min_gdi
        instance._store_limit = limit
        return instance

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
                    record = self._format_sft_record(traj)
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
                    record = self._format_dpo_record(pref)
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

    def export_grpo(self, output_path: str, group_size: int = 8) -> ExportResult:
        """导出为 GRPO 分组训练格式.

        将同一 task_id 的多条轨迹分组，用于 Group Relative Policy Optimization 训练。
        每组至少 2 条轨迹才能计算 group advantage。

        GRPO 格式 (每行一个 JSON):
        ```json
        {
            "task_id": "任务 ID",
            "prompt": "任务描述",
            "trajectories": [
                {"response": "执行轨迹文本", "reward": 0.85},
                {"response": "执行轨迹文本", "reward": 0.55}
            ]
        }
        ```

        Args:
            output_path: 输出文件路径
            group_size: 每组最大轨迹数

        Returns:
            ExportResult: 导出结果
        """
        try:
            trajectories = self._load_trajectories()
            logger.info("导出 GRPO: 加载 %d 条轨迹", len(trajectories))

            # 按 task_id 分组
            task_groups: Dict[str, List[Dict[str, Any]]] = {}
            for traj in trajectories:
                tid = traj.get("task_id", "")
                if not tid:
                    continue
                if tid not in task_groups:
                    task_groups[tid] = []
                task_groups[tid].append(traj)

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            total_groups = 0
            with open(output, "w", encoding="utf-8") as f:
                for task_id, group in task_groups.items():
                    if len(group) < 2:
                        continue  # 至少需要 2 条才能计算 advantage

                    # 按 reward 排序，取 top group_size 条
                    sorted_group = sorted(
                        group, key=lambda t: t.get("reward", 0.0), reverse=True
                    )
                    selected = sorted_group[:group_size]

                    metadata = selected[0].get("metadata", {})
                    record = {
                        "task_id": task_id,
                        "prompt": metadata.get(
                            "task_description", f"Solve task: {task_id}"
                        ),
                        "trajectories": [
                            {
                                "response": self._steps_to_text(t.get("steps", [])),
                                "reward": t.get("reward", 0.0),
                            }
                            for t in selected
                        ],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_groups += 1

            logger.info("GRPO 导出完成: %d 组 -> %s", total_groups, output)
            return ExportResult(
                success=True,
                output_path=str(output),
                total_records=total_groups,
                format="grpo",
            )

        except Exception as e:
            logger.exception("GRPO 导出失败")
            return ExportResult(
                success=False,
                format="grpo",
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
        需要安装 huggingface-hub 并已通过 huggingface-cli login 认证。

        上传文件:
        - README.md (Dataset Card)
        - sft_train.jsonl (SFT 格式)
        - dpo_train.jsonl (DPO 格式，如有偏好对)
        - benchmark.jsonl (评测基准格式)

        Args:
            repo_id: HuggingFace 仓库 ID (e.g. "username/dataset-name")

        Returns:
            ExportResult: 导出结果
        """
        if not _HAS_HF:
            return ExportResult(
                success=False,
                format="huggingface",
                error="需要安装 huggingface-hub: pip install knowlyr-hub[hf]",
            )

        try:
            api = HfApi()

            # 创建 dataset repo（已存在则跳过）
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            logger.info("HuggingFace repo 已就绪: %s", repo_id)

            uploaded_count = 0

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # 生成 Dataset Card
                card = self.generate_datacard()
                readme_path = tmp_path / "README.md"
                readme_path.write_text(card, encoding="utf-8")

                # 导出 SFT
                sft_path = tmp_path / "sft_train.jsonl"
                sft_result = self.export_sft(str(sft_path))
                if sft_result.success:
                    uploaded_count += sft_result.total_records

                # 导出 DPO（如有偏好对）
                if self.preferences_dir and self.preferences_dir.exists():
                    dpo_path = tmp_path / "dpo_train.jsonl"
                    dpo_result = self.export_dpo(str(dpo_path))
                    if dpo_result.success:
                        uploaded_count += dpo_result.total_records

                # 导出 Benchmark
                bench_path = tmp_path / "benchmark.jsonl"
                bench_result = self.export_benchmark(str(bench_path))
                if bench_result.success:
                    uploaded_count += bench_result.total_records

                # 上传整个目录
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(tmp_path),
                    repo_type="dataset",
                )

            logger.info(
                "HuggingFace 上传完成: %s (%d 条记录)", repo_id, uploaded_count,
            )
            return ExportResult(
                success=True,
                output_path=f"https://huggingface.co/datasets/{repo_id}",
                total_records=uploaded_count,
                format="huggingface",
            )

        except Exception as e:
            logger.exception("HuggingFace 导出失败")
            return ExportResult(
                success=False,
                format="huggingface",
                error=str(e),
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
    # Phase 3: 分割导出
    # ------------------------------------------------------------------

    def export_sft_split(
        self,
        output_dir: str,
        split_ratios: dict[str, float] | None = None,
        seed: int = 42,
    ) -> dict[str, "ExportResult"]:
        """导出 SFT 数据并按比例分割为 train/val/test.

        Args:
            output_dir: 输出目录，会生成 train.jsonl / val.jsonl / test.jsonl
            split_ratios: 分割比例，默认 {"train": 0.8, "val": 0.1, "test": 0.1}
            seed: 随机种子

        Returns:
            各分割的 ExportResult
        """
        ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
        try:
            trajectories = self._load_trajectories()
            # 只选成功的轨迹，按 reward 排序后 shuffle
            successful = sorted(
                [t for t in trajectories if t.get("success", False)],
                key=lambda t: t.get("reward", 0.0),
                reverse=True,
            )

            # 格式化为 SFT 记录
            records = [self._format_sft_record(t) for t in successful]

            return self._split_and_write(records, output_dir, ratios, seed, "sft")

        except Exception as e:
            logger.exception("SFT 分割导出失败")
            return {
                name: ExportResult(success=False, format="sft", error=str(e))
                for name in ratios
            }

    def export_dpo_split(
        self,
        output_dir: str,
        split_ratios: dict[str, float] | None = None,
        seed: int = 42,
    ) -> dict[str, "ExportResult"]:
        """导出 DPO 数据并按比例分割为 train/val/test.

        Args:
            output_dir: 输出目录，会生成 train.jsonl / val.jsonl / test.jsonl
            split_ratios: 分割比例，默认 {"train": 0.8, "val": 0.1, "test": 0.1}
            seed: 随机种子

        Returns:
            各分割的 ExportResult
        """
        ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}

        if not self.preferences_dir or not self.preferences_dir.exists():
            return {
                name: ExportResult(
                    success=False,
                    format="dpo",
                    error="偏好对文件不存在，请先运行 Pipeline 生成偏好对",
                )
                for name in ratios
            }

        try:
            preferences = self._load_preferences()

            records = [self._format_dpo_record(p) for p in preferences]

            return self._split_and_write(records, output_dir, ratios, seed, "dpo")

        except Exception as e:
            logger.exception("DPO 分割导出失败")
            return {
                name: ExportResult(success=False, format="dpo", error=str(e))
                for name in ratios
            }

    def _format_sft_record(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """将单条轨迹格式化为 SFT 记录."""
        response_text = self._steps_to_text(traj.get("steps", []))
        response_parts = [response_text] if response_text else []

        metadata = traj.get("metadata", {})
        return {
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

    def _format_dpo_record(self, pref: Dict[str, Any]) -> Dict[str, Any]:
        """将单条偏好对格式化为 DPO 记录."""
        chosen_text = self._steps_to_text(pref.get("chosen", {}).get("steps", []))
        rejected_text = self._steps_to_text(
            pref.get("rejected", {}).get("steps", [])
        )

        # 优先使用 task_description（信息更丰富），与 reader.py _cas_to_dpo 保持一致
        metadata = pref.get("metadata", {})
        chosen_meta = pref.get("chosen", {}).get("metadata", {})
        task_desc = (
            metadata.get("task_description")
            or chosen_meta.get("task_description")
        )
        prompt = task_desc if task_desc else f"Solve the following task:\n\nTask ID: {pref.get('task_id', '')}"

        return {
            "prompt": prompt,
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

    def _split_and_write(
        self,
        records: List[Dict[str, Any]],
        output_dir: str,
        ratios: dict[str, float],
        seed: int,
        fmt: str,
    ) -> dict[str, "ExportResult"]:
        """对记录列表做可复现 shuffle + 按比例分割写入."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # 可复现 shuffle
        indices = list(range(len(records)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        shuffled = [records[i] for i in indices]

        # 按比例分割
        splits: dict[str, list[Dict[str, Any]]] = {}
        total = len(shuffled)
        offset = 0
        split_names = list(ratios.keys())
        for i, name in enumerate(split_names):
            if i == len(split_names) - 1:
                # 最后一份取余
                splits[name] = shuffled[offset:]
            else:
                count = int(total * ratios[name])
                splits[name] = shuffled[offset : offset + count]
                offset += count

        results: dict[str, ExportResult] = {}
        for name, split_records in splits.items():
            out_path = Path(output_dir) / f"{name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in split_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results[name] = ExportResult(
                success=True,
                output_path=str(out_path),
                total_records=len(split_records),
                format=fmt,
            )
            logger.info(
                "%s 分割 %s: %d 条 -> %s", fmt.upper(), name, len(split_records), out_path
            )

        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """从 JSONL 文件或 CAS 存储加载轨迹."""
        # CAS 模式：按 GDI 排名读取
        if self._store is not None:
            rows = self._store.list(
                order_by="gdi_score",
                limit=getattr(self, "_store_limit", 10000),
            )
            min_gdi = getattr(self, "_store_min_gdi", 0.0)
            records = [r for r in rows if r.get("gdi_score", 0) >= min_gdi]
            # 引用计数 +1
            for r in records:
                h = r.get("content_hash", "")
                if h:
                    self._store.increment_export(h)
            logger.info("CAS 加载 %d 条轨迹 (min_gdi=%.2f)", len(records), min_gdi)
            return records

        # JSONL 模式
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
        """加载偏好对，兼容 JSON 数组和 JSONL 两种格式."""
        records = []
        if not self.preferences_dir or not self.preferences_dir.exists():
            return records

        with open(self.preferences_dir, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return records

        # 尝试 JSON 数组格式（knowlyr-reward preferences 输出）
        if content.startswith("["):
            data = json.loads(content)
            if isinstance(data, list):
                return data

        # 回退到 JSONL 格式（Pipeline 内部生成）
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def validate_dataset(
        self,
        format: str = "sft",
        max_length: int = 2048,
    ) -> Dict[str, Any]:
        """验证导出数据集的质量.

        检查项目:
        - 缺失率: 必填字段的缺失比例
        - 长度分布: response/chosen/rejected 的 token 估算长度
        - reward 异常: NaN/Inf/全零检测
        - 空内容: 空 response 或空步骤的比例

        Args:
            format: 数据格式 ("sft", "dpo", "grpo")
            max_length: 序列长度阈值，超过此值会发出警告

        Returns:
            验证结果字典::

                {
                    "total_records": 100,
                    "issues": [...],          # 问题列表
                    "missing_rate": {...},     # 各字段缺失率
                    "length_stats": {...},     # 长度统计
                    "reward_stats": {...},     # reward 统计
                    "is_valid": True/False,    # 总体是否通过
                }
        """
        if format == "dpo":
            return self._validate_dpo(max_length)
        return self._validate_sft(max_length)

    def _validate_sft(self, max_length: int) -> Dict[str, Any]:
        """验证 SFT 数据集."""
        trajectories = self._load_trajectories()
        issues: List[str] = []

        if not trajectories:
            return {
                "total_records": 0,
                "issues": ["数据集为空"],
                "missing_rate": {},
                "length_stats": {},
                "reward_stats": {},
                "is_valid": False,
            }

        # 统计
        total = len(trajectories)
        missing: Dict[str, int] = {
            "task_id": 0,
            "steps": 0,
            "success": 0,
            "reward": 0,
        }
        rewards: List[float] = []
        step_counts: List[int] = []
        response_lengths: List[int] = []  # 估算 char 长度
        empty_responses = 0

        for traj in trajectories:
            if not traj.get("task_id"):
                missing["task_id"] += 1
            steps = traj.get("steps", [])
            if not steps:
                missing["steps"] += 1
                empty_responses += 1
            else:
                step_counts.append(len(steps))
                # 估算 response 长度 (字符数 ≈ token 数 * 1.5 for 中文)
                text_len = sum(
                    len(str(s.get("action", ""))) + len(str(s.get("observation", "")))
                    for s in steps
                )
                response_lengths.append(text_len)

            if "success" not in traj and "success" not in traj.get("outcome", {}):
                missing["success"] += 1

            reward = traj.get("reward")
            if reward is None:
                missing["reward"] += 1
            else:
                try:
                    r = float(reward)
                    if r != r:  # NaN check
                        issues.append(f"task={traj.get('task_id', '?')}: reward 为 NaN")
                    elif abs(r) == float("inf"):
                        issues.append(f"task={traj.get('task_id', '?')}: reward 为 Inf")
                    else:
                        rewards.append(r)
                except (ValueError, TypeError):
                    issues.append(f"task={traj.get('task_id', '?')}: reward 无法解析")

        # 缺失率
        missing_rate = {k: v / total for k, v in missing.items()}
        for field, rate in missing_rate.items():
            if rate > 0.05:
                issues.append(f"字段 '{field}' 缺失率 {rate:.1%} (超过 5%)")

        # 长度统计
        length_stats: Dict[str, Any] = {}
        if response_lengths:
            over_limit = sum(1 for ln in response_lengths if ln > max_length * 4)
            length_stats = {
                "min": min(response_lengths),
                "max": max(response_lengths),
                "mean": sum(response_lengths) / len(response_lengths),
                "over_max_length": over_limit,
                "over_max_length_rate": over_limit / len(response_lengths),
            }
            if over_limit > 0:
                issues.append(
                    f"{over_limit} 条记录 ({over_limit/len(response_lengths):.1%}) "
                    f"超过 max_length={max_length} 的估算阈值"
                )

        # Reward 统计
        reward_stats: Dict[str, Any] = {}
        if rewards:
            all_zero = all(r == 0.0 for r in rewards)
            reward_stats = {
                "min": min(rewards),
                "max": max(rewards),
                "mean": sum(rewards) / len(rewards),
                "all_zero": all_zero,
            }
            if all_zero:
                issues.append("所有 reward 均为 0.0，检查 reward 计算是否正常")

        # 空内容
        if empty_responses > 0:
            rate = empty_responses / total
            issues.append(f"{empty_responses} 条记录 ({rate:.1%}) 无步骤数据")

        return {
            "total_records": total,
            "issues": issues,
            "missing_rate": missing_rate,
            "length_stats": length_stats,
            "reward_stats": reward_stats,
            "is_valid": len(issues) == 0,
        }

    def _validate_dpo(self, max_length: int) -> Dict[str, Any]:
        """验证 DPO 数据集."""
        preferences = self._load_preferences()
        issues: List[str] = []

        if not preferences:
            return {
                "total_records": 0,
                "issues": ["偏好对数据集为空"],
                "missing_rate": {},
                "length_stats": {},
                "reward_stats": {},
                "is_valid": False,
            }

        total = len(preferences)
        missing_chosen = 0
        missing_rejected = 0
        reward_margins: List[float] = []

        for pref in preferences:
            if not pref.get("chosen"):
                missing_chosen += 1
            if not pref.get("rejected"):
                missing_rejected += 1

            margin = pref.get("reward_margin")
            if margin is not None:
                try:
                    m = float(margin)
                    if m < 0:
                        issues.append(
                            f"task={pref.get('task_id', '?')}: reward_margin < 0 ({m:.3f})"
                        )
                    reward_margins.append(m)
                except (ValueError, TypeError):
                    pass

        missing_rate = {
            "chosen": missing_chosen / total,
            "rejected": missing_rejected / total,
        }

        reward_stats: Dict[str, Any] = {}
        if reward_margins:
            reward_stats = {
                "min_margin": min(reward_margins),
                "max_margin": max(reward_margins),
                "mean_margin": sum(reward_margins) / len(reward_margins),
            }
            if all(m == 0.0 for m in reward_margins):
                issues.append("所有 reward_margin 均为 0.0")

        for field, rate in missing_rate.items():
            if rate > 0.0:
                issues.append(f"字段 '{field}' 缺失率 {rate:.1%}")

        return {
            "total_records": total,
            "issues": issues,
            "missing_rate": missing_rate,
            "length_stats": {},
            "reward_stats": reward_stats,
            "is_valid": len(issues) == 0,
        }

    def _steps_to_text(self, steps: List[Dict[str, Any]]) -> str:
        """将步骤列表转为文本格式.

        # NOTE: 此函数与 trainer/reader.py 的同名函数逻辑同步。
        # 修改任一处时请同步另一处，或考虑迁移至 knowlyr-core 包。

        兼容两种字段约定:
        - 标准 (recorder): action / observation
        - Crew 轨迹:       tool_call.name+params / tool_result.output
        - Hub 内部:        tool / params / output
        """
        parts = []
        for i, step in enumerate(steps, 1):
            thought = step.get("thought", "")

            # --- action ---
            action = step.get("action", "")
            if not action:
                # hub 内部格式: tool + params
                tool = step.get("tool", "")
                if tool:
                    params = step.get("params")
                    action = f"{tool}({json.dumps(params, ensure_ascii=False)})" if params else tool
            if not action:
                # crew 轨迹格式: tool_call dict
                tc = step.get("tool_call")
                if isinstance(tc, dict):
                    name = tc.get("name", "")
                    params = tc.get("parameters")
                    action = f"{name}({json.dumps(params, ensure_ascii=False)})" if params else name

            # --- observation ---
            observation = step.get("observation", "")
            if not observation:
                observation = step.get("output", "")
            if not observation:
                tr = step.get("tool_result")
                if isinstance(tr, dict):
                    observation = tr.get("output", "")

            step_text = f"Step {i}:"
            if thought:
                step_text += f"\nThought: {thought}"
            if action:
                step_text += f"\nAction: {action}"
            if observation:
                step_text += f"\nObservation: {observation}"
            parts.append(step_text)

        return "\n\n".join(parts) if parts else ""
