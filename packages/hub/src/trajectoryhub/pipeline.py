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

# ── 可选依赖 (try/except) ──────────────────────────────────────────

try:
    import agentsandbox  # noqa: F401

    _HAS_SANDBOX = True
except ImportError:
    _HAS_SANDBOX = False

try:
    from agentrecorder import Recorder
    from agentrecorder.adapters import get_adapter as _get_recorder_adapter

    _HAS_RECORDER = True
except ImportError:
    _HAS_RECORDER = False

    def _get_recorder_adapter(name: str):  # type: ignore[misc]
        return None

try:
    from agentreward.reward import RewardEngine

    _HAS_REWARD = True
except ImportError:
    _HAS_REWARD = False

try:
    from knowlyrcore.domain import get_domain_profile

    _HAS_CORE_DOMAIN = True
except ImportError:
    _HAS_CORE_DOMAIN = False

    def get_domain_profile(domain: str):  # type: ignore[misc]
        return None


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
    content_hash: str = ""


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

        # Step 5: 存入 CAS + 计算 GDI（如果配置了 store_path）
        if self.config.store_path:
            self._store_and_score(all_trajectories)

        # Step 6: 运行质检
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
        1. 创建沙箱环境 (knowlyr-sandbox)
        2. 在沙箱中运行 Agent (按 agent_config 配置)
        3. 录制执行轨迹 (knowlyr-recorder)
        4. 计算 Reward (knowlyr-reward)
        5. 返回 Trajectory

        当 Agent 框架未安装时返回空轨迹。
        如果只有日志文件而无需运行 Agent，请使用 run_from_log()。

        Args:
            task: 要执行的任务
            agent_config: Agent 配置

        Returns:
            Trajectory: 执行轨迹及其 reward
        """
        start_time = time.time()

        # 当前 Agent 框架执行部分仍需外部集成
        # 如需处理已有日志，请使用 run_from_log()
        logger.debug(
            "run_single: task=%s, agent=%s/%s (Agent 执行需外部集成)",
            task.task_id, agent_config.framework, agent_config.model,
        )

        return Trajectory(
            task_id=task.task_id,
            agent_framework=agent_config.framework,
            agent_model=agent_config.model,
            steps=[],
            total_steps=0,
            success=False,
            reward=0.0,
            step_rewards=[],
            duration_seconds=time.time() - start_time,
            metadata={
                "task_description": task.description,
                "task_type": task.type,
            },
        )

    def run_from_log(
        self,
        log_path: str | Path,
        framework: str,
        task: Optional[TaskInfo] = None,
    ) -> Trajectory:
        """从已有日志文件生成带评分的轨迹.

        这是处理已有 Agent 日志的主要入口。流程：
        1. 使用 recorder 适配器解析日志 → 标准 Trajectory
        2. 使用 reward engine 计算分数
        3. 合并为 hub Trajectory 返回

        Args:
            log_path: Agent 日志文件路径。
            framework: Agent 框架名 (openhands / sweagent)。
            task: 可选的任务信息。不传则从日志中自动提取。

        Returns:
            Trajectory: 带 reward 评分的标准轨迹。

        Raises:
            RuntimeError: recorder 或 reward 未安装。
            ValueError: 不支持的框架类型。
        """
        start_time = time.time()
        log_path = Path(log_path)

        # Step 1: 解析日志
        recorder_traj = self._parse_log(log_path, framework)
        logger.info("日志解析完成: %d 步 (%s)", len(recorder_traj.steps), framework)

        # 合并 task 信息
        if task is not None:
            recorder_traj.task = task

        # Step 2: Reward 评分
        reward_result = self._score_trajectory(recorder_traj)

        duration = time.time() - start_time

        # Step 3: 转换为 hub Trajectory
        return self._recorder_to_hub_trajectory(
            recorder_traj, reward_result, duration,
        )

    def run_batch_from_logs(
        self,
        log_dir: str | Path,
        framework: str,
        pattern: str = "*",
    ) -> List[Trajectory]:
        """批量处理日志目录.

        Args:
            log_dir: 包含日志文件的目录。
            framework: Agent 框架名。
            pattern: 文件匹配模式。

        Returns:
            Trajectory 列表。
        """
        if not _HAS_RECORDER:
            raise RuntimeError("批量处理需要安装 knowlyr-recorder: pip install knowlyr-recorder")

        adapter = _get_recorder_adapter(framework)
        if adapter is None:
            raise ValueError(f"不支持的框架: {framework}")

        recorder = Recorder(adapter)
        recorder_trajs = recorder.convert_batch(str(log_dir), pattern)

        trajectories = []
        for rtraj in recorder_trajs:
            reward_result = self._score_trajectory(rtraj)
            traj = self._recorder_to_hub_trajectory(rtraj, reward_result, 0.0)
            trajectories.append(traj)

        logger.info("批量处理完成: %d 条轨迹", len(trajectories))
        return trajectories

    def run_from_trajectories(
        self,
        jsonl_path: str | Path,
    ) -> PipelineResult:
        """从已标准化的 Trajectory JSONL 直接走评分 -> 导出管线.

        适用于 Crew 等已经输出标准 Trajectory 格式的数据源，
        跳过 adapter 解析步骤，直接加载 → 评分 → 偏好对 → CAS → 质检。

        同一份 JSONL 跑两遍结果一样（幂等）。

        Args:
            jsonl_path: 标准 Trajectory JSONL 文件路径。

        Returns:
            PipelineResult: 执行结果汇总。
        """
        start_time = time.time()
        jsonl_path = Path(jsonl_path)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 加载 JSONL
        raw_trajectories = self._load_trajectories(jsonl_path)
        logger.info("从 JSONL 加载 %d 条轨迹: %s", len(raw_trajectories), jsonl_path)

        # Step 2: 逐条评分（如果 reward engine 可用且尚无 reward）
        for traj in raw_trajectories:
            if traj.reward != 0.0:
                continue  # 已有评分，保持幂等
            score = self._score_trajectory_from_hub(traj)
            if score is not None:
                traj.reward = score.total_score
                traj.step_rewards = [sr.total_score for sr in score.step_rewards]

        # Step 3: 保存带评分的轨迹
        trajectories_path = output_dir / "trajectories.jsonl"
        self._save_trajectories(trajectories_path, raw_trajectories)
        logger.info("轨迹已保存: %s (%d 条)", trajectories_path, len(raw_trajectories))

        # Step 4: 构建偏好对
        preferences_path = output_dir / "preferences.jsonl"
        self._build_preference_pairs(raw_trajectories, preferences_path)

        # Step 5: CAS + GDI
        if self.config.store_path:
            self._store_and_score(raw_trajectories)

        # Step 6: 质检
        quality_report_path = output_dir / "quality_report.json"
        self._run_quality_check(trajectories_path, quality_report_path)

        duration = time.time() - start_time
        completed = len(raw_trajectories)
        logger.info(
            "run_from_trajectories 完成: %d 条轨迹, 耗时 %.1fs",
            completed, duration,
        )

        return PipelineResult(
            total_tasks=completed,
            completed=completed,
            failed=0,
            trajectories_path=str(trajectories_path),
            preferences_path=str(preferences_path),
            quality_report_path=str(quality_report_path),
            duration_seconds=duration,
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
    # 集成方法 — recorder / reward
    # ------------------------------------------------------------------

    def _parse_log(self, log_path: Path, framework: str):
        """使用 recorder 适配器解析日志.

        Returns:
            agentrecorder.schema.Trajectory (recorder 格式)
        """
        if not _HAS_RECORDER:
            raise RuntimeError(
                "日志解析需要安装 knowlyr-recorder: pip install knowlyr-recorder"
            )

        adapter = _get_recorder_adapter(framework)
        if adapter is None:
            raise ValueError(f"不支持的框架: {framework}")

        recorder = Recorder(adapter)
        return recorder.convert(str(log_path))

    def _score_trajectory(self, recorder_traj) -> Optional[dict]:
        """使用 reward engine 评分.

        Args:
            recorder_traj: agentrecorder.schema.Trajectory

        Returns:
            RewardResult 或 None (reward 未安装时)
        """
        if not _HAS_REWARD:
            logger.debug("knowlyr-reward 未安装，跳过评分")
            return None

        # 传递 domain profile 给 RewardEngine
        profile = None
        if _HAS_CORE_DOMAIN:
            domain = self.config.domain or "coding"
            profile = get_domain_profile(domain)

        engine = RewardEngine(profile=profile)

        # 将 recorder 格式转换为 reward engine 输入
        steps = [
            {
                "tool": step.tool_call.name,
                "params": step.tool_call.parameters,
                "output": step.tool_result.output if step.tool_result else "",
            }
            for step in recorder_traj.steps
        ]

        outcome = {
            "success": recorder_traj.outcome.success,
            "tests_passed": recorder_traj.outcome.tests_passed,
            "tests_total": (
                recorder_traj.outcome.tests_passed + recorder_traj.outcome.tests_failed
            ),
        }

        result = engine.score({
            "task": recorder_traj.task.description,
            "steps": steps,
            "outcome": outcome,
        })

        return result

    def _recorder_to_hub_trajectory(
        self,
        recorder_traj,
        reward_result,
        duration: float,
    ) -> Trajectory:
        """将 recorder Trajectory + reward 结果合并为 hub Trajectory."""
        # 将 recorder steps 转为 dict 列表
        steps_dicts = []
        for step in recorder_traj.steps:
            steps_dicts.append({
                "thought": step.thought,
                "tool": step.tool_call.name,
                "params": step.tool_call.parameters,
                "output": step.tool_result.output if step.tool_result else "",
                "exit_code": step.tool_result.exit_code if step.tool_result else 0,
            })

        # 提取 reward
        total_reward = 0.0
        step_rewards = []
        if reward_result is not None:
            total_reward = reward_result.total_score
            step_rewards = [sr.total_score for sr in reward_result.step_rewards]

        return Trajectory(
            task_id=recorder_traj.task.task_id,
            agent_framework=recorder_traj.agent,
            agent_model=recorder_traj.model,
            steps=steps_dicts,
            total_steps=len(steps_dicts),
            success=recorder_traj.outcome.success,
            reward=total_reward,
            step_rewards=step_rewards,
            duration_seconds=duration,
            metadata={
                "task_description": recorder_traj.task.description,
                "outcome_tests_passed": recorder_traj.outcome.tests_passed,
                "outcome_tests_failed": recorder_traj.outcome.tests_failed,
                "outcome_total_tokens": recorder_traj.outcome.total_tokens,
                **recorder_traj.metadata,
            },
        )

    def _score_trajectory_from_hub(self, traj: Trajectory) -> Optional[dict]:
        """对已加载的 hub Trajectory 评分.

        与 _score_trajectory (接受 recorder 格式) 不同，本方法接受
        hub 内部的 Trajectory dataclass，从 steps dict 中提取字段。

        Args:
            traj: hub Trajectory dataclass

        Returns:
            RewardResult 或 None (reward 未安装时)
        """
        if not _HAS_REWARD:
            logger.debug("knowlyr-reward 未安装，跳过评分")
            return None

        profile = None
        if _HAS_CORE_DOMAIN:
            domain = self.config.domain or "coding"
            profile = get_domain_profile(domain)

        engine = RewardEngine(profile=profile)

        steps = []
        for step in traj.steps:
            # 兼容 hub 内部格式 (tool/params/output) 和 crew 格式 (tool_call/tool_result)
            tool = step.get("tool", "")
            params = step.get("params", {})
            output = step.get("output", "")

            if not tool:
                tc = step.get("tool_call")
                if isinstance(tc, dict):
                    tool = tc.get("name", "")
                    params = tc.get("parameters", {})

            if not output:
                tr = step.get("tool_result")
                if isinstance(tr, dict):
                    output = tr.get("output", "")

            steps.append({"tool": tool, "params": params, "output": output})

        outcome = {
            "success": traj.success,
            "total_steps": traj.total_steps,
        }

        result = engine.score({
            "task": traj.metadata.get("task_description", traj.task_id),
            "steps": steps,
            "outcome": outcome,
        })

        return result

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
        from trajectoryhub.cas import content_hash as _content_hash

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for traj in trajectories:
                # 计算 content_hash（如果还没有）
                if not traj.content_hash:
                    traj.content_hash = _content_hash(traj.steps)
                line = json.dumps(
                    {
                        "content_hash": traj.content_hash,
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
                        content_hash=data.get("content_hash", ""),
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

    def _store_and_score(self, trajectories: List[Trajectory]) -> None:
        """将轨迹存入 CAS 并计算 GDI 分数."""
        from trajectoryhub.cas import CAStore
        from trajectoryhub.gdi import GDIScorer

        store_path = Path(self.config.store_path)  # type: ignore[arg-type]
        store = CAStore(store_path)
        scorer = GDIScorer()

        try:
            hashes = []
            for traj in trajectories:
                h = store.put(traj)
                traj.content_hash = h
                hashes.append(h)

            # 批量计算 GDI
            gdi_batch: Dict[str, float] = {}
            for h in hashes:
                row = store.get(h)
                if row is None:
                    continue
                gdi = scorer.score(
                    reward=row.get("reward", 0.0),
                    export_count=row.get("export_count", 0),
                    created_at=row.get("created_at"),
                )
                gdi_batch[h] = gdi.total

            store.update_gdi_batch(gdi_batch)
            stats = store.stats()
            logger.info(
                "CAS 存储完成: %d 条轨迹, 去重后 %d 条, 平均 GDI %.3f",
                len(trajectories), stats["total_trajectories"], stats["avg_gdi"],
            )
        finally:
            store.close()

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
