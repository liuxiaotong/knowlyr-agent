"""AgentSandbox MCP Server - Model Context Protocol 服务."""

import json
from typing import Any, Dict, List

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.replay import Trajectory


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-sandbox[mcp]")

    server = Server("agentsandbox")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="create_sandbox",
                description="创建 Docker 沙箱执行环境",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "Git 仓库 URL",
                        },
                        "base_commit": {
                            "type": "string",
                            "description": "起始 commit SHA",
                        },
                        "language": {
                            "type": "string",
                            "description": "编程语言 (默认: python)",
                            "default": "python",
                        },
                        "image": {
                            "type": "string",
                            "description": "Docker 镜像 (默认: python:3.11-slim)",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "超时秒数 (默认: 300)",
                            "default": 300,
                        },
                    },
                    "required": ["repo_url", "base_commit"],
                },
            ),
            Tool(
                name="execute_tool",
                description="在沙箱中执行工具 (file_read, file_write, shell, search, git)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sandbox_id": {
                            "type": "string",
                            "description": "沙箱 ID",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "工具名称",
                            "enum": ["file_read", "file_write", "shell", "search", "git"],
                        },
                        "params": {
                            "type": "object",
                            "description": "工具参数",
                        },
                    },
                    "required": ["sandbox_id", "tool_name", "params"],
                },
            ),
            Tool(
                name="reset_sandbox",
                description="重置沙箱到初始状态",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sandbox_id": {
                            "type": "string",
                            "description": "沙箱 ID",
                        },
                    },
                    "required": ["sandbox_id"],
                },
            ),
            Tool(
                name="replay_trajectory",
                description="在沙箱中重放 Agent 执行轨迹",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sandbox_id": {
                            "type": "string",
                            "description": "沙箱 ID",
                        },
                        "trajectory": {
                            "type": "object",
                            "description": "轨迹数据 (包含 steps 和 metadata)",
                        },
                    },
                    "required": ["sandbox_id", "trajectory"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """调用工具."""

        if name == "create_sandbox":
            sandbox_config = SandboxConfig(
                image=arguments.get("image", "python:3.11-slim"),
                timeout=arguments.get("timeout", 300),
            )
            task_config = TaskConfig(
                repo_url=arguments["repo_url"],
                base_commit=arguments["base_commit"],
                language=arguments.get("language", "python"),
            )

            # Validate task config
            errors = task_config.validate()
            if errors:
                return [TextContent(type="text", text=f"配置错误:\n" + "\n".join(errors))]

            # TODO: 实际创建沙箱
            return [
                TextContent(
                    type="text",
                    text="[未实现] create_sandbox 将创建 Docker 沙箱:\n"
                    f"- 仓库: {task_config.repo_url}\n"
                    f"- Commit: {task_config.base_commit}\n"
                    f"- 镜像: {sandbox_config.image}\n"
                    f"- 超时: {sandbox_config.timeout}s",
                )
            ]

        elif name == "execute_tool":
            sandbox_id = arguments["sandbox_id"]
            tool_name = arguments["tool_name"]
            params = arguments.get("params", {})

            # TODO: 在实际沙箱中执行工具
            return [
                TextContent(
                    type="text",
                    text=f"[未实现] execute_tool 将在沙箱 {sandbox_id} 中执行:\n"
                    f"- 工具: {tool_name}\n"
                    f"- 参数: {json.dumps(params, ensure_ascii=False)}",
                )
            ]

        elif name == "reset_sandbox":
            sandbox_id = arguments["sandbox_id"]

            # TODO: 重置沙箱
            return [
                TextContent(
                    type="text",
                    text=f"[未实现] reset_sandbox 将重置沙箱 {sandbox_id} 到初始状态",
                )
            ]

        elif name == "replay_trajectory":
            sandbox_id = arguments["sandbox_id"]
            trajectory_data = arguments["trajectory"]

            trajectory = Trajectory.from_dict(trajectory_data)
            step_count = len(trajectory.steps)

            # TODO: 实际重放轨迹
            return [
                TextContent(
                    type="text",
                    text=f"[未实现] replay_trajectory 将在沙箱 {sandbox_id} 中重放:\n"
                    f"- 步骤数: {step_count}\n"
                    f"- 元数据: {json.dumps(trajectory.metadata, ensure_ascii=False)}",
                )
            ]

        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-sandbox[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """主入口."""
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
