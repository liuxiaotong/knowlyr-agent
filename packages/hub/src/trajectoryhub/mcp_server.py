"""MCP Server - Model Context Protocol 服务."""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from trajectoryhub.config import PipelineConfig, TaskSource, AgentConfig
from trajectoryhub.pipeline import Pipeline
from trajectoryhub.exporter import DatasetExporter


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-hub[mcp]")

    server = Server("trajectoryhub")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="run_pipeline",
                description="运行完整的 Agent 轨迹数据 Pipeline (Task -> Sandbox -> Recorder -> Reward -> Export)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_source": {
                            "type": "string",
                            "description": "任务定义文件路径 (JSONL 格式)",
                        },
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "framework": {
                                        "type": "string",
                                        "description": "Agent 框架 (openhands / sweagent)",
                                    },
                                    "model": {
                                        "type": "string",
                                        "description": "LLM 模型",
                                    },
                                },
                            },
                            "description": "Agent 配置列表",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "输出目录 (默认: ./output)",
                            "default": "./output",
                        },
                        "parallel_workers": {
                            "type": "integer",
                            "description": "并行工作进程数 (默认: 1)",
                            "default": 1,
                        },
                    },
                    "required": ["task_source"],
                },
            ),
            Tool(
                name="export_dataset",
                description="将轨迹数据导出为指定的训练格式 (SFT / DPO / Benchmark / HuggingFace)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "trajectories_path": {
                            "type": "string",
                            "description": "轨迹文件路径 (JSONL)",
                        },
                        "preferences_path": {
                            "type": "string",
                            "description": "偏好对文件路径 (JSONL，DPO 导出时必需)",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["sft", "dpo", "benchmark", "huggingface"],
                            "description": "导出格式",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "输出文件路径",
                        },
                        "repo_id": {
                            "type": "string",
                            "description": "HuggingFace 仓库 ID (仅 huggingface 格式)",
                        },
                    },
                    "required": ["trajectories_path", "format", "output_path"],
                },
            ),
            Tool(
                name="process_log",
                description="处理单个 Agent 日志文件，解析并评分生成标准轨迹",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "log_path": {
                            "type": "string",
                            "description": "Agent 日志文件路径",
                        },
                        "framework": {
                            "type": "string",
                            "enum": ["openhands", "sweagent", "swe-agent"],
                            "description": "Agent 框架",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "输出目录 (默认: ./output)",
                            "default": "./output",
                        },
                    },
                    "required": ["log_path", "framework"],
                },
            ),
            Tool(
                name="process_logs_batch",
                description="批量处理 Agent 日志目录，解析并评分生成标准轨迹",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "log_dir": {
                            "type": "string",
                            "description": "包含日志文件的目录",
                        },
                        "framework": {
                            "type": "string",
                            "enum": ["openhands", "sweagent", "swe-agent"],
                            "description": "Agent 框架",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "文件匹配模式 (默认: *)",
                            "default": "*",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "输出目录 (默认: ./output)",
                            "default": "./output",
                        },
                    },
                    "required": ["log_dir", "framework"],
                },
            ),
            Tool(
                name="pipeline_status",
                description="查看 Pipeline 执行状态和进度",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_dir": {
                            "type": "string",
                            "description": "Pipeline 输出目录",
                        },
                    },
                    "required": ["output_dir"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """调用工具."""

        if name == "run_pipeline":
            # 构建配置
            agents = []
            for agent_data in arguments.get("agents", [{"framework": "openhands"}]):
                agents.append(
                    AgentConfig(
                        framework=agent_data.get("framework", "openhands"),
                        model=agent_data.get("model", "claude-sonnet-4-20250514"),
                    )
                )

            config = PipelineConfig(
                task_source=TaskSource(
                    path=arguments["task_source"],
                    source_type="jsonl",
                ),
                agents=agents,
                output_dir=arguments.get("output_dir", "./output"),
                parallel_workers=arguments.get("parallel_workers", 1),
            )

            pipeline = Pipeline(config)
            result = pipeline.run()

            return [
                TextContent(
                    type="text",
                    text=f"Pipeline 执行完成:\n"
                    f"- 总任务数: {result.total_tasks}\n"
                    f"- 完成: {result.completed}\n"
                    f"- 失败: {result.failed}\n"
                    f"- 轨迹输出: {result.trajectories_path}\n"
                    f"- 偏好对输出: {result.preferences_path}\n"
                    f"- 质检报告: {result.quality_report_path}\n"
                    f"- 耗时: {result.duration_seconds:.1f}s",
                )
            ]

        elif name == "export_dataset":
            exporter = DatasetExporter(
                trajectories_dir=arguments["trajectories_path"],
                preferences_dir=arguments.get("preferences_path"),
            )

            export_format = arguments["format"]
            output_path = arguments["output_path"]

            if export_format == "sft":
                result = exporter.export_sft(output_path)
            elif export_format == "dpo":
                result = exporter.export_dpo(output_path)
            elif export_format == "benchmark":
                result = exporter.export_benchmark(output_path)
            elif export_format == "huggingface":
                repo_id = arguments.get("repo_id", "")
                result = exporter.export_huggingface(repo_id)
            else:
                return [TextContent(type="text", text=f"不支持的导出格式: {export_format}")]

            if result.success:
                return [
                    TextContent(
                        type="text",
                        text=f"导出成功:\n"
                        f"- 格式: {result.format}\n"
                        f"- 输出路径: {result.output_path}\n"
                        f"- 导出记录数: {result.total_records}",
                    )
                ]
            else:
                return [TextContent(type="text", text=f"导出失败: {result.error}")]

        elif name == "process_log":
            log_path = arguments["log_path"]
            framework = arguments["framework"]
            output_dir = arguments.get("output_dir", "./output")

            config = PipelineConfig(output_dir=output_dir)
            pipeline = Pipeline(config)

            try:
                traj = pipeline.run_from_log(log_path, framework)
            except (RuntimeError, ValueError) as e:
                return [TextContent(type="text", text=f"处理失败: {e}")]

            return [
                TextContent(
                    type="text",
                    text=f"日志处理完成:\n"
                    f"- 任务 ID: {traj.task_id}\n"
                    f"- 框架: {traj.agent_framework}\n"
                    f"- 模型: {traj.agent_model}\n"
                    f"- 步数: {traj.total_steps}\n"
                    f"- 成功: {traj.success}\n"
                    f"- Reward: {traj.reward:.3f}\n"
                    f"- 耗时: {traj.duration_seconds:.1f}s",
                )
            ]

        elif name == "process_logs_batch":
            log_dir = arguments["log_dir"]
            framework = arguments["framework"]
            pattern = arguments.get("pattern", "*")
            output_dir = arguments.get("output_dir", "./output")

            config = PipelineConfig(output_dir=output_dir)
            pipeline = Pipeline(config)

            try:
                trajectories = pipeline.run_batch_from_logs(log_dir, framework, pattern)
            except (RuntimeError, ValueError) as e:
                return [TextContent(type="text", text=f"批量处理失败: {e}")]

            if not trajectories:
                return [TextContent(type="text", text="没有找到匹配的日志文件。")]

            # 保存轨迹
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            traj_path = out_path / "trajectories.jsonl"
            with open(traj_path, "w", encoding="utf-8") as f:
                for traj in trajectories:
                    f.write(json.dumps({
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
                    }, ensure_ascii=False) + "\n")

            success_count = sum(1 for t in trajectories if t.success)
            avg_reward = (
                sum(t.reward for t in trajectories) / len(trajectories)
                if trajectories else 0.0
            )

            return [
                TextContent(
                    type="text",
                    text=f"批量处理完成:\n"
                    f"- 轨迹数: {len(trajectories)}\n"
                    f"- 成功率: {success_count}/{len(trajectories)}\n"
                    f"- 平均 Reward: {avg_reward:.3f}\n"
                    f"- 输出: {traj_path}",
                )
            ]

        elif name == "pipeline_status":
            output_dir = Path(arguments["output_dir"])

            # 检查各个文件是否存在
            checkpoint_path = output_dir / "checkpoint.json"
            trajectories_path = output_dir / "trajectories.jsonl"
            preferences_path = output_dir / "preferences.jsonl"
            report_path = output_dir / "quality_report.json"

            status_parts = [f"Pipeline 状态 ({output_dir}):"]

            if checkpoint_path.exists():
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                status_parts.append(
                    f"- Checkpoint: 已完成 {checkpoint.get('completed', 0)} 个, "
                    f"失败 {checkpoint.get('failed', 0)} 个"
                )
            else:
                status_parts.append("- Checkpoint: 无")

            if trajectories_path.exists():
                with open(trajectories_path, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                status_parts.append(f"- 轨迹文件: {count} 条")
            else:
                status_parts.append("- 轨迹文件: 未生成")

            if preferences_path.exists():
                with open(preferences_path, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                status_parts.append(f"- 偏好对文件: {count} 对")
            else:
                status_parts.append("- 偏好对文件: 未生成")

            if report_path.exists():
                status_parts.append("- 质检报告: 已生成")
            else:
                status_parts.append("- 质检报告: 未生成")

            return [TextContent(type="text", text="\n".join(status_parts))]

        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-hub[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """主入口."""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
