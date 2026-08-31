"""
mcp_provider.py — 社区标准 MCP 客户端封装。

读取项目根目录的 .mcp.json，使用 mcp SDK 连接 MCP server，
自动发现工具并暴露为统一 ToolDef。

支持 transport:
- stdio: 本地子进程（如 python -m mcp_simple_arxiv）
- streamable-http: 远程 HTTP MCP server
"""

from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from . import ToolDef, ToolProvider


# ---- config loader ----

def _expand_env(value: str) -> str:
    """替换 ${VAR} 和 ${VAR:-default} 占位符。"""
    import re

    def _repl(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(2)
        return os.environ.get(var, default or "")

    return re.sub(r'\$\{(\w+)(?::-(.*?))?\}', _repl, value)


def _find_config() -> Path | None:
    """向上查找 .mcp.json。"""
    d = Path.cwd()
    for _ in range(5):
        p = d / ".mcp.json"
        if p.exists():
            return p
        if d.parent == d:
            break
        d = d.parent
    return None


def load_mcp_config(path: str | Path | None = None) -> dict[str, dict]:
    """加载 MCP 配置，返回 {server_name: {command, args, transport, url, headers, env}}。

    默认查找项目根目录 .mcp.json。若文件不存在，返回空 dict（MCP 可选）。
    """
    if path is None:
        found = _find_config()
        if found is None:
            return {}
        path = found

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    servers = raw.get("mcpServers", {})
    result: dict[str, dict] = {}
    for name, cfg in servers.items():
        if isinstance(cfg, dict):
            entry: dict = {}
            # stdio
            if "command" in cfg:
                entry["command"] = _expand_env(cfg["command"])
                entry["args"] = [_expand_env(a) for a in cfg.get("args", [])]
                env_vars = cfg.get("env", {})
                entry["env"] = {k: _expand_env(v) for k, v in env_vars.items()} if env_vars else None
            # http
            if "url" in cfg:
                entry["url"] = _expand_env(cfg["url"])
                headers = cfg.get("headers", {})
                entry["headers"] = {k: _expand_env(v) for k, v in headers.items()} if headers else None
            # common
            entry["transport"] = cfg.get("transport", "stdio")
            entry["disabled"] = cfg.get("disabled", False)
            result[name] = entry
    return result


# ---- provider ----

class MCPProvider(ToolProvider):
    """管理多个 MCP server 连接，将其工具暴露为统一 ToolDef。

    特性：
    - 惰性连接：首次 list_tools() 时才启动子进程/建立连接
    - 自动重连：call_tool 失败时重试一次
    - 优雅关闭：close() 清理所有子进程
    """

    name = "mcp"

    def __init__(self, config: dict[str, dict] | None = None,
                 config_path: str | Path | None = None):
        """
        Args:
            config: 直接传入 server 配置字典
            config_path: .mcp.json 路径，默认自动查找
        """
        self._servers: dict[str, dict] = config or load_mcp_config(config_path)
        self._sessions: dict[str, tuple[Any, Any]] = {}  # name → (session, stack)
        self._tool_index: dict[str, tuple[ToolDef, str]] = {}  # tool_name → (ToolDef, server_name)

    # ---- lifecycle ----

    async def _connect(self, server_name: str) -> Any:
        """连接指定 MCP server，返回 ClientSession。"""
        if server_name in self._sessions:
            session, _stack = self._sessions[server_name]
            return session

        cfg = self._servers.get(server_name)
        if not cfg:
            raise ValueError(f"MCP server '{server_name}' not found in config")

        stack = AsyncExitStack()
        transport = cfg.get("transport", "stdio")

        try:
            if transport == "stdio":
                from mcp.client.stdio import stdio_client, StdioServerParameters

                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg.get("args", []),
                    env=cfg.get("env"),
                )
                read, write = await stack.enter_async_context(stdio_client(params))

            elif transport in ("streamable-http", "sse"):
                from mcp.client.streamable_http import streamablehttp_client

                url = cfg["url"]
                headers = cfg.get("headers")
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(url, headers=headers or {})
                )

            else:
                raise ValueError(f"Unsupported MCP transport: {transport}")

            from mcp.client.session import ClientSession

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[server_name] = (session, stack)
            return session

        except Exception:
            await stack.aclose()
            raise

    async def close(self) -> None:
        """关闭所有 MCP 连接。"""
        for name in list(self._sessions):
            _, stack = self._sessions.pop(name)
            try:
                await stack.aclose()
            except Exception:
                pass
        self._tool_index.clear()

    # ---- tool discovery ----

    async def _discover(self) -> None:
        """连接所有未禁用的 server 并发现工具。幂等。"""
        if self._tool_index:
            return

        for srv_name, cfg in self._servers.items():
            if cfg.get("disabled"):
                continue
            try:
                session = await self._connect(srv_name)
                result = await session.list_tools()
                for t in result.tools:
                    td = ToolDef(
                        name=f"{srv_name}__{t.name}",
                        description=t.description or t.name,
                        parameters=t.inputSchema,
                        source="mcp",
                        annotations={
                            "readOnlyHint": getattr(t.annotations, "readOnlyHint", None) if t.annotations else None,
                            "destructiveHint": getattr(t.annotations, "destructiveHint", None) if t.annotations else None,
                            "idempotentHint": getattr(t.annotations, "idempotentHint", None) if t.annotations else None,
                            "openWorldHint": getattr(t.annotations, "openWorldHint", None) if t.annotations else None,
                        },
                    )
                    self._tool_index[td.name] = (td, srv_name)
            except Exception as e:
                from agent.observability import log_event
                log_event("mcp_connect_failed", node="mcp", level="warning",
                          server=srv_name, error=f"{type(e).__name__}: {e}")

    async def list_tools(self) -> list[ToolDef]:
        await self._discover()
        return [td for td, _ in self._tool_index.values()]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        await self._discover()

        if name not in self._tool_index:
            raise KeyError(f"MCP tool '{name}' not found")

        _td, server_name = self._tool_index[name]
        # name 格式: "servername__toolname"，拆分回原始 tool name
        mcp_tool_name = name.split("__", 1)[1] if "__" in name else name

        async def _call() -> Any:
            session = await self._connect(server_name)
            result = await session.call_tool(mcp_tool_name, arguments)
            # result.content 是 list[ContentBlock]，提取文本
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts) if parts else str(result.content)

        try:
            return await _call()
        except Exception:
            # 重连一次
            if server_name in self._sessions:
                _, stack = self._sessions.pop(server_name)
                try:
                    await stack.aclose()
                except Exception:
                    pass
            return await _call()
