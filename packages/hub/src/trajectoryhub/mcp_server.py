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
