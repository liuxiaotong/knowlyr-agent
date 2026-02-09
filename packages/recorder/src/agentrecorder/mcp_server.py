"""AgentRecorder MCP Server - Model Context Protocol 服务."""

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

from agentrecorder.schema import Trajectory


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-recorder[mcp]")

    server = Server("agentrecorder")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="convert_logs",
                description="将 Agent 日志转换为标准化轨迹格式",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "log_path": {
                            "type": "string",
                            "description": "Agent 日志文件路径",
                        },
                        "framework": {
                            "type": "string",
                            "enum": ["openhands", "swe-agent"],
                            "description": "Agent 框架名称",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "输出 JSONL 文件路径（可选）",
                        },
                    },
                    "required": ["log_path", "framework"],
                },
            ),
            Tool(
                name="validate_logs",
                description="验证日志文件是否为指定的 Agent 框架格式",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "log_path": {
                            "type": "string",
                            "description": "待验证的日志文件路径",
                        },
                        "framework": {
                            "type": "string",
                            "enum": ["openhands", "swe-agent"],
                            "description": "要验证的 Agent 框架名称（可选，不指定则自动检测）",
                        },
                    },
                    "required": ["log_path"],
                },
            ),
            Tool(
                name="get_schema",
                description="返回标准化轨迹的 JSON Schema 定义",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """调用工具."""

        if name == "convert_logs":
            log_path = arguments["log_path"]
            framework = arguments["framework"]
            output_path = arguments.get("output_path")

            path = Path(log_path)
            if not path.exists():
                return [TextContent(type="text", text=f"日志文件不存在: {log_path}")]

            # 获取适配器
            adapter = _get_adapter(framework)
            if adapter is None:
                return [TextContent(type="text", text=f"不支持的框架: {framework}")]

            if not adapter.validate(log_path):
                return [
                    TextContent(
                        type="text",
                        text=f"日志格式不匹配 {framework} 格式: {log_path}",
                    )
                ]

            try:
                from agentrecorder.recorder import Recorder

                recorder = Recorder(adapter)
                trajectory = recorder.convert(log_path)

                if output_path:
                    trajectory.to_jsonl(output_path)
                    return [
                        TextContent(
                            type="text",
                            text=f"转换成功:\n"
                            f"- 输出路径: {output_path}\n"
                            f"- Agent: {trajectory.agent}\n"
                            f"- 步骤数: {len(trajectory.steps)}\n"
                            f"- 结果: {'成功' if trajectory.outcome.success else '失败'}",
                        )
                    ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=trajectory.model_dump_json(indent=2),
                        )
                    ]
            except NotImplementedError as e:
                return [TextContent(type="text", text=f"适配器尚未实现: {e}")]
            except Exception as e:
                return [TextContent(type="text", text=f"转换失败: {e}")]

        elif name == "validate_logs":
            log_path = arguments["log_path"]
            framework = arguments.get("framework")

            path = Path(log_path)
            if not path.exists():
                return [TextContent(type="text", text=f"文件不存在: {log_path}")]

            if framework:
                adapter = _get_adapter(framework)
                if adapter is None:
                    return [TextContent(type="text", text=f"不支持的框架: {framework}")]
                is_valid = adapter.validate(log_path)
                return [
                    TextContent(
                        type="text",
                        text=f"验证结果: {'有效' if is_valid else '无效'}\n"
                        f"- 文件: {log_path}\n"
                        f"- 框架: {framework}",
                    )
                ]
            else:
                # 自动检测
                results = []
                for fw_name in ["openhands", "swe-agent"]:
                    adapter = _get_adapter(fw_name)
                    if adapter and adapter.validate(log_path):
                        results.append(fw_name)

                if results:
                    return [
                        TextContent(
                            type="text",
                            text=f"检测到匹配的框架: {', '.join(results)}\n- 文件: {log_path}",
                        )
                    ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"未检测到匹配的框架格式: {log_path}",
                        )
                    ]

        elif name == "get_schema":
            schema_json = Trajectory.schema_json_example()
            return [
                TextContent(
                    type="text",
                    text=f"## Trajectory JSON Schema\n\n```json\n{schema_json}\n```",
                )
            ]

        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


def _get_adapter(framework: str):
    """根据框架名称获取适配器实例."""
    if framework == "openhands":
        from agentrecorder.adapters.openhands import OpenHandsAdapter

        return OpenHandsAdapter()
    elif framework == "swe-agent":
        from agentrecorder.adapters.sweagent import SWEAgentAdapter

        return SWEAgentAdapter()
    return None


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-recorder[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """主入口."""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
