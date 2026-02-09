"""Core pipeline orchestrator - 核心流水线编排器.

编排 Task -> Sandbox -> Recorder -> Reward -> Export 全流程。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from trajectoryhub.config import AgentConfig, PipelineConfig
from trajectoryhub.tasks import TaskInfo, TaskLoader

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """单条执行轨迹.

    Attributes:
        task_id: 任务 ID
        agent_framework: Agent 框架名
        agent_model: Agent 使用的模型
        steps: 执行步骤列表
        total_steps: 总步数
        success: 任务是否成功完成
        reward: 综合 Reward 分数
        step_rewards: 每步的 Reward 分数
        duration_seconds: 执行时长 (秒)
        metadata: 额外元数据
    """

    task_id: str = ""
    agent_framework: str = ""
    agent_model: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    success: bool = False
    reward: float = 0.0
    step_rewards: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Pipeline 执行结果.

    Attributes:
        total_tasks: 总任务数
        completed: 完成数
        failed: 失败数
        trajectories_path: 轨迹输出路径
        preferences_path: 偏好对输出路径
        quality_report_path: 质检报告路径
        duration_seconds: 总执行时长 (秒)
    """

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    trajectories_path: Optional[str] = None
    preferences_path: Optional[str] = None
    quality_report_path: Optional[str] = None
    duration_seconds: float = 0.0


class Pipeline:
    """核心流水线编排器.

    串联整个 Agent 轨迹数据生产流水线：
    1. 加载任务 (Task)
    2. 对每个任务 x 每个 Agent：创建沙箱 -> 运行 Agent -> 录制轨迹 -> 打分
    3. 构建偏好对 (同任务的多条轨迹按 reward 排序)
    4. 运行数据质检 (data-check)
    5. 返回 PipelineResult

    Usage:
        config = PipelineConfig(
            task_source=TaskSource(path="tasks.jsonl"),
            agents=[
                AgentConfig(framework="openhands", model="claude-sonnet-4-20250514"),
                AgentConfig(framework="openhands", model="gpt-4o"),
            ],
            output_dir="./output",
        )

        pipeline = Pipeline(config)
        result = pipeline.run()
        print(f"完成: {result.completed}/{result.total_tasks}")
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._checkpoint_data: Dict[str, Any] = {}

    def run(self) -> PipelineResult:
        """运行完整 Pipeline.

        完整流程：
        1. 加载任务列表
        2. 对每个 (task, agent) 组合执行 run_single
        3. 构建偏好对 (同任务多轨迹按 reward 排序)
        4. 运行质检
        5. 保存结果

        Returns:
            PipelineResult: 执行结果汇总
        """
        start_time = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 加载任务
        tasks = self._load_tasks()
        logger.info("加载 %d 个任务，%d 个 agent 配置", len(tasks), len(self.config.agents))

        # Step 2: 对每个 task x agent 执行
        all_trajectories: List[Trajectory] = []
        completed = 0
        failed = 0
        total_combinations = len(tasks) * len(self.config.agents)

        for task in tasks:
            for agent_config in self.config.agents:
                try:
                    trajectory = self.run_single(task, agent_config)
                    all_trajectories.append(trajectory)
                    completed += 1
                except Exception:
                    failed += 1
                    logger.exception(
                        "任务执行失败: task=%s, agent=%s/%s",
                        task.task_id, agent_config.framework, agent_config.model,
                    )

                if (completed + failed) % 10 == 0:
                    logger.info("进度: %d/%d (成功=%d, 失败=%d)",
                                completed + failed, total_combinations, completed, failed)

                # Checkpoint
                if (completed + failed) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(output_dir, all_trajectories, completed, failed)
                    logger.debug("Checkpoint 已保存: %d 条轨迹", len(all_trajectories))

        # Step 3: 保存轨迹
        trajectories_path = output_dir / "trajectories.jsonl"
        self._save_trajectories(trajectories_path, all_trajectories)
        logger.info("轨迹已保存: %s (%d 条)", trajectories_path, len(all_trajectories))

        # Step 4: 构建偏好对
        preferences_path = output_dir / "preferences.jsonl"
        self._build_preference_pairs(all_trajectories, preferences_path)

        # Step 5: 运行质检
        quality_report_path = output_dir / "quality_report.json"
        self._run_quality_check(trajectories_path, quality_report_path)

        duration = time.time() - start_time
        logger.info("Pipeline 完成: %d/%d 成功, 耗时 %.1fs", completed, len(tasks), duration)

        return PipelineResult(
            total_tasks=len(tasks),
            completed=completed,
            failed=failed,
            trajectories_path=str(trajectories_path),
            preferences_path=str(preferences_path),
            quality_report_path=str(quality_report_path),
            duration_seconds=duration,
        )

    def run_single(self, task: TaskInfo, agent_config: AgentConfig) -> Trajectory:
        """运行单个任务.

        流程：
        1. 创建沙箱环境 (agent-sandbox)
        2. 在沙箱中运行 Agent (按 agent_config 配置)
        3. 录制执行轨迹 (agent-recorder)
        4. 计算 Reward (agent-reward)
        5. 返回 Trajectory

        Args:
            task: 要执行的任务
            agent_config: Agent 配置

        Returns:
            Trajectory: 执行轨迹及其 reward
        """
        # TODO: 接入 agent-sandbox 创建沙箱
        # sandbox = Sandbox(self.config.sandbox_config)
        # sandbox.setup(task.repo, task.base_commit)

        # TODO: 接入 Agent 框架运行任务
        # agent = AgentRunner(agent_config)
        # raw_log = agent.run(task, sandbox)

        # TODO: 接入 agent-recorder 转换为标准轨迹
        # recorder = Recorder()
        # steps = recorder.parse(raw_log, framework=agent_config.framework)

        # TODO: 接入 agent-reward 计算分数
        # reward_calculator = RewardCalculator(self.config.reward_config)
        # step_rewards = reward_calculator.score_steps(steps)
        # total_reward = reward_calculator.aggregate(step_rewards)

        # 当前返回空轨迹（等待原子模块就绪后替换）
        return Trajectory(
            task_id=task.task_id,
            agent_framework=agent_config.framework,
            agent_model=agent_config.model,
            steps=[],
            total_steps=0,
            success=False,
            reward=0.0,
            step_rewards=[],
            duration_seconds=0.0,
            metadata={
                "task_description": task.description,
                "task_type": task.type,
            },
        )

    def resume(self, checkpoint_path: str) -> PipelineResult:
        """从 checkpoint 恢复执行.

        读取 checkpoint 文件中已完成的任务，跳过这些任务后继续执行剩余任务。

        Args:
            checkpoint_path: checkpoint 文件路径

        Returns:
            PipelineResult: 执行结果汇总
        """
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

        with open(checkpoint_file, "r", encoding="utf-8") as f:
            self._checkpoint_data = json.load(f)

        # 获取已完成的 task_id 列表
        completed_ids = set(self._checkpoint_data.get("completed_task_ids", []))

        # 加载任务并过滤已完成的
        all_tasks = self._load_tasks()
        remaining_tasks = [t for t in all_tasks if t.task_id not in completed_ids]
        logger.info("从 checkpoint 恢复: 已完成 %d, 剩余 %d 个任务",
                     len(completed_ids), len(remaining_tasks))

        # 加载已有轨迹
        existing_trajectories: List[Trajectory] = []
        existing_path = self._checkpoint_data.get("trajectories_path")
        if existing_path and Path(existing_path).exists():
            existing_trajectories = self._load_trajectories(Path(existing_path))

        start_time = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_trajectories = list(existing_trajectories)
        completed = self._checkpoint_data.get("completed", 0)
        failed = self._checkpoint_data.get("failed", 0)

        for task in remaining_tasks:
            for agent_config in self.config.agents:
                try:
                    trajectory = self.run_single(task, agent_config)
                    all_trajectories.append(trajectory)
                    completed += 1
                except Exception:
                    failed += 1
                    logger.exception(
                        "任务执行失败 (resume): task=%s, agent=%s/%s",
                        task.task_id, agent_config.framework, agent_config.model,
                    )

        # 保存最终结果
        trajectories_path = output_dir / "trajectories.jsonl"
        self._save_trajectories(trajectories_path, all_trajectories)

        preferences_path = output_dir / "preferences.jsonl"
        self._build_preference_pairs(all_trajectories, preferences_path)

        quality_report_path = output_dir / "quality_report.json"
        self._run_quality_check(trajectories_path, quality_report_path)

        duration = time.time() - start_time

        return PipelineResult(
            total_tasks=len(all_tasks),
            completed=completed,
            failed=failed,
            trajectories_path=str(trajectories_path),
            preferences_path=str(preferences_path),
            quality_report_path=str(quality_report_path),
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_tasks(self) -> List[TaskInfo]:
        """根据配置加载任务列表."""
        source = self.config.task_source
        loader = TaskLoader()

        if source.source_type == "swebench" and source.dataset_name:
            tasks = loader.load_from_swebench(
                dataset_name=source.dataset_name,
                split=source.split,
                limit=source.limit,
            )
        elif source.path:
            tasks = loader.load_from_jsonl(source.path)
        else:
            return []

        # 应用过滤条件
        if source.filters:
            tasks = loader.filter_tasks(
                tasks,
                language=source.filters.get("language"),
                difficulty=source.filters.get("difficulty"),
                task_type=source.filters.get("type"),
            )

        if source.limit and len(tasks) > source.limit:
            tasks = tasks[: source.limit]

        return tasks

    def _save_trajectories(self, path: Path, trajectories: List[Trajectory]) -> None:
        """将轨迹列表保存为 JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                line = json.dumps(
                    {
                        "task_id": traj.task_id,
                        "agent_framework": traj.agent_framework,
                        "agent_model": traj.agent_model,
                        "steps": traj.steps,
                        "total_steps": traj.total_steps,
                        "success": traj.success,
                        "reward": traj.reward,
                        "step_rewards": traj.step_rewards,
                        "duration_seconds": traj.duration_seconds,
                        "metadata": traj.metadata,
                    },
                    ensure_ascii=False,
                )
                f.write(line + "\n")

    def _load_trajectories(self, path: Path) -> List[Trajectory]:
        """从 JSONL 文件加载轨迹列表."""
        trajectories = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                trajectories.append(
                    Trajectory(
                        task_id=data.get("task_id", ""),
                        agent_framework=data.get("agent_framework", ""),
                        agent_model=data.get("agent_model", ""),
                        steps=data.get("steps", []),
                        total_steps=data.get("total_steps", 0),
                        success=data.get("success", False),
                        reward=data.get("reward", 0.0),
                        step_rewards=data.get("step_rewards", []),
                        duration_seconds=data.get("duration_seconds", 0.0),
                        metadata=data.get("metadata", {}),
                    )
                )
        return trajectories

    def _build_preference_pairs(
        self, trajectories: List[Trajectory], output_path: Path
    ) -> None:
        """构建偏好对.

        同一任务的多条轨迹按 reward 排序，形成 (chosen, rejected) 对。
        """
        # 按 task_id 分组
        task_groups: Dict[str, List[Trajectory]] = {}
        for traj in trajectories:
            if traj.task_id not in task_groups:
                task_groups[traj.task_id] = []
            task_groups[traj.task_id].append(traj)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for task_id, group in task_groups.items():
                if len(group) < 2:
                    continue

                # 按 reward 排序
                sorted_group = sorted(group, key=lambda t: t.reward, reverse=True)

                # 两两配对: 高 reward 为 chosen, 低 reward 为 rejected
                for i in range(len(sorted_group) - 1):
                    chosen = sorted_group[i]
                    rejected = sorted_group[i + 1]

                    if chosen.reward == rejected.reward:
                        continue

                    pair = {
                        "task_id": task_id,
                        "chosen": {
                            "agent_framework": chosen.agent_framework,
                            "agent_model": chosen.agent_model,
                            "steps": chosen.steps,
                            "reward": chosen.reward,
                            "success": chosen.success,
                        },
                        "rejected": {
                            "agent_framework": rejected.agent_framework,
                            "agent_model": rejected.agent_model,
                            "steps": rejected.steps,
                            "reward": rejected.reward,
                            "success": rejected.success,
                        },
                        "reward_margin": chosen.reward - rejected.reward,
                    }
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def _run_quality_check(self, trajectories_path: Path, report_path: Path) -> None:
        """运行数据质检.

        TODO: 接入 data-check (knowlyr-datacheck) 进行质检。
        当前生成一个占位报告。
        """
        report = {
            "status": "pending",
            "message": "质检功能待接入 knowlyr-datacheck",
            "trajectories_path": str(trajectories_path),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def _save_checkpoint(
        self,
        output_dir: Path,
        trajectories: List[Trajectory],
        completed: int,
        failed: int,
    ) -> None:
        """保存 checkpoint 以支持断点恢复."""
        checkpoint_path = output_dir / "checkpoint.json"
        trajectories_path = output_dir / "trajectories_checkpoint.jsonl"

        self._save_trajectories(trajectories_path, trajectories)

        checkpoint = {
            "completed": completed,
            "failed": failed,
            "completed_task_ids": list({t.task_id for t in trajectories}),
            "trajectories_path": str(trajectories_path),
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
