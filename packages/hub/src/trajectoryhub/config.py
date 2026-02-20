"""Pipeline configuration - 流水线配置."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskSource(BaseModel):
    """任务来源配置.

    Attributes:
        path: 任务定义文件路径 (JSONL 格式)
        source_type: 任务来源类型 (jsonl / swebench / custom)
        dataset_name: SWE-bench 数据集名称 (仅 swebench 类型使用)
        split: 数据集分割 (train / test / dev)
        limit: 限制加载的任务数量
        filters: 过滤条件 (language, difficulty, type 等)
    """

    path: Optional[str] = None
    source_type: str = "jsonl"
    dataset_name: Optional[str] = None
    split: str = "test"
    limit: Optional[int] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent 配置.

    Attributes:
        framework: Agent 框架 (openhands / sweagent / custom)
        model: 使用的 LLM 模型 (e.g. claude-sonnet-4-20250514, gpt-4o)
        max_steps: 最大执行步数
        temperature: 采样温度
        extra_args: 传给 Agent 框架的额外参数
    """

    framework: str = "openhands"
    model: str = "claude-sonnet-4-20250514"
    max_steps: int = 30
    temperature: float = 0.0
    domain: str = "coding"
    extra_args: Dict[str, Any] = Field(default_factory=dict)


class SandboxConfig(BaseModel):
    """沙箱环境配置.

    Attributes:
        image: Docker 镜像名
        timeout: 单任务超时时间 (秒)
        memory_limit: 内存限制 (e.g. "4g")
        network_enabled: 是否允许网络访问
    """

    image: str = "knowlyr/sandbox:latest"
    timeout: int = 600
    memory_limit: str = "4g"
    network_enabled: bool = False


class RewardConfig(BaseModel):
    """Reward 计算配置.

    Attributes:
        enable_rule_layer: 启用规则层 (使用 data-check)
        enable_model_layer: 启用模型层 (LLM-as-Judge)
        judge_model: Judge 使用的 LLM 模型
        rubric_weights: Rubric 各维度的权重
            - goal_progress: 目标推进
            - tool_selection: 工具选择
            - param_correctness: 参数正确性
            - info_utilization: 信息利用
            - non_redundancy: 非冗余性
    """

    enable_rule_layer: bool = True
    enable_model_layer: bool = False
    judge_model: str = "claude-sonnet-4-20250514"
    rubric_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "goal_progress": 0.3,
            "tool_selection": 0.2,
            "param_correctness": 0.2,
            "info_utilization": 0.15,
            "non_redundancy": 0.15,
        }
    )


class PipelineConfig(BaseModel):
    """Pipeline 全局配置.

    Attributes:
        task_source: 任务来源配置
        agents: Agent 配置列表 (支持多 Agent 对比)
        sandbox_config: 沙箱环境配置
        reward_config: Reward 计算配置
        output_dir: 输出目录
        parallel_workers: 并行工作进程数
        checkpoint_interval: 每隔多少个任务做一次 checkpoint
        resume_from: 从 checkpoint 恢复时的路径
    """

    task_source: TaskSource = Field(default_factory=TaskSource)
    agents: List[AgentConfig] = Field(default_factory=lambda: [AgentConfig()])
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    reward_config: RewardConfig = Field(default_factory=RewardConfig)
    output_dir: str = "./output"
    parallel_workers: int = 1
    checkpoint_interval: int = 10
    resume_from: Optional[str] = None
    domain: str = "coding"
    store_path: Optional[str] = None  # SQLite CAS 存储路径，设置后自动启用内容寻址+GDI
