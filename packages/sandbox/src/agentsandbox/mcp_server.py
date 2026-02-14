"""AgentSandbox MCP Server - Model Context Protocol 服务."""

import json
import logging
from typing import Any, Dict, List

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from agentsandbox.config import SandboxConfig, TaskConfig
from agentsandbox.replay import Trajectory, replay_trajectory
from agentsandbox.sandbox import Sandbox

logger = logging.getLogger(__name__)

# 活跃沙箱映射: sandbox_id -> Sandbox
_sandboxes: Dict[str, Sandbox] = {}


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
            Tool(
                name="sandbox_snapshot",
                description="保存沙箱当前状态快照（文件系统 diff + 环境信息）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sandbox_id": {
                            "type": "string",
                            "description": "沙箱 ID",
                        },
                        "label": {
                            "type": "string",
                            "description": "快照标签（可选）",
                        },
                    },
                    "required": ["sandbox_id"],
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

            errors = task_config.validate()
            if errors:
                return [TextContent(type="text", text="配置错误:\n" + "\n".join(errors))]

            try:
                sandbox = Sandbox.create(sandbox_config, task_config)
                sandbox_id = sandbox.container_id
                _sandboxes[sandbox_id] = sandbox
                logger.info("MCP 创建沙箱: %s", sandbox_id)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "sandbox_id": sandbox_id,
                            "image": sandbox_config.image,
                            "repo_url": task_config.repo_url,
                            "base_commit": task_config.base_commit,
                        }, ensure_ascii=False),
                    )
                ]
            except Exception as e:
                logger.exception("MCP 创建沙箱失败")
                return [TextContent(type="text", text=f"创建沙箱失败: {e}")]

        elif name == "execute_tool":
            sandbox_id = arguments["sandbox_id"]
            tool_name = arguments["tool_name"]
            params = arguments.get("params", {})

            sandbox = _sandboxes.get(sandbox_id)
            if not sandbox:
                return [TextContent(type="text", text=f"沙箱不存在: {sandbox_id}")]

            result = sandbox.execute_tool(tool_name, params)
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "output": result.output,
                        "exit_code": result.exit_code,
                        "error": result.error,
                    }, ensure_ascii=False),
                )
            ]

        elif name == "reset_sandbox":
            sandbox_id = arguments["sandbox_id"]

            sandbox = _sandboxes.get(sandbox_id)
            if not sandbox:
                return [TextContent(type="text", text=f"沙箱不存在: {sandbox_id}")]

            try:
                sandbox.reset()
                return [TextContent(type="text", text=f"沙箱已重置: {sandbox_id}")]
            except Exception as e:
                logger.exception("MCP 重置沙箱失败: %s", sandbox_id)
                return [TextContent(type="text", text=f"重置失败: {e}")]

        elif name == "replay_trajectory":
            sandbox_id = arguments["sandbox_id"]
            trajectory_data = arguments["trajectory"]

            sandbox = _sandboxes.get(sandbox_id)
            if not sandbox:
                return [TextContent(type="text", text=f"沙箱不存在: {sandbox_id}")]

            trajectory = Trajectory.from_dict(trajectory_data)
            result = replay_trajectory(sandbox, trajectory)

            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": result.success,
                        "total_steps": result.total_steps,
                        "completed_steps": result.completed_steps,
                        "divergence_step": result.divergence_step,
                        "details": result.details,
                    }, ensure_ascii=False),
                )
            ]

        elif name == "sandbox_snapshot":
            sandbox_id = arguments["sandbox_id"]
            label = arguments.get("label", "")

            sandbox = _sandboxes.get(sandbox_id)
            if not sandbox:
                return [TextContent(type="text", text=f"沙箱不存在: {sandbox_id}")]

            try:
                # Get diff from base commit
                diff_result = sandbox.exec("git diff --stat HEAD 2>/dev/null || echo 'no git'")
                env_result = sandbox.exec("python3 --version 2>/dev/null && pip list --format=columns 2>/dev/null | head -20 || echo 'no python'")

                snapshot = {
                    "sandbox_id": sandbox_id,
                    "label": label,
                    "file_diff": diff_result.output if diff_result.success else "(无法获取)",
                    "environment": env_result.output if env_result.success else "(无法获取)",
                }

                lines = [
                    f"## 沙箱快照: {sandbox_id}",
                    f"**标签**: {label or '(无)'}", "",
                    "### 文件变更",
                    f"```\n{snapshot['file_diff']}\n```", "",
                    "### 环境信息",
                    f"```\n{snapshot['environment']}\n```",
                ]
                return [TextContent(type="text", text="\n".join(lines))]
            except Exception as e:
                return [TextContent(type="text", text=f"快照失败: {e}")]

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
