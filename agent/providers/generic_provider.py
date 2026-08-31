"""
generic_provider.py — 通用工具集（文件/网络/系统横切能力）。

参考 CowAgent `agent/tools/` 的成熟逻辑（路径 resolve + 越界拒绝、输出截断、
SSRF 守卫），但保持本项目 ToolDef + ToolProvider 骨架，不移植 BaseTool 类。

工具（第一批只读 + 第二批破坏性/网络）：
- read_file   (readOnly)  读工作区内文本文件
- list_dir    (readOnly)  列目录
- get_time    (readOnly)  当前本地时间
- calculator  (readOnly)  ast 白名单算术求值
- write_file  (破坏性)    写文本文件（覆盖，权限门覆盖）
- fetch_url   (readOnly)  抓取 URL 正文（SSRF 守卫）

web_search / run_shell 观察期，见 docs/generic-tools-plan.md §4。

注册面：仅 _EXPOSED_TOOLS（read_file / list_dir / write_file）暴露给
研究 agent；get_time / calculator / fetch_url 代码保留、不注册。
"""

from __future__ import annotations

import ast
import ipaddress
import json
import operator
import os
import socket
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import ToolDef, ToolProvider
from agent.safety import tool_allowed

# 工作区根 = 项目根。相对路径基于此解析，越界即拒。
# ponytail: 只允许项目根，不细分 data/；需要更严时收紧 _resolve_path。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_MAX_READ_BYTES = 32_000
_MAX_FETCH_BYTES = 40_000


# ---- 错误/成功约定 ----
# 沿用 builtin_provider 的 {"ok":false,"error_type":...} 契约，dispatcher 透传。

def _err(error_type: str, detail: str, next_action: str) -> str:
    return json.dumps({
        "ok": False, "error": detail, "next": next_action,
        "error_type": error_type,
    }, ensure_ascii=False)


def _ok(data: dict) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


# ---- 共享辅助 ----

def resolve_workspace_path(path: str) -> Path:
    """展开 ~ / 相对→绝对，拒绝逃出工作区（含 symlink / .. 穿越）。

    工作区 root = 项目根，相对路径基于此解析，越界即拒。此守卫是
    generic 工具与 web/api workspace 端点共用的唯一路径安全边界。
    """
    p = Path(os.path.expanduser(path or "."))
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    real = p.resolve()
    try:
        real.relative_to(_PROJECT_ROOT.resolve())
    except ValueError:
        raise PermissionError(f"path escapes workspace: {path}")
    return real


# 向后兼容别名（模块内部旧调用点）
_resolve_path = resolve_workspace_path


def _truncate(text: str, max_bytes: int = _MAX_READ_BYTES) -> str:
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    cut = b[:max_bytes].decode("utf-8", errors="ignore")
    return f"{cut}\n...[truncated — {len(b)} bytes total, {max_bytes} shown]"


def _is_private_host(hostname: str) -> bool:
    """SSRF 守卫：拒绝内网/回环/保留地址。解析不了的保守拒绝。"""
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        except OSError:
            return True
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


# ---- 工具实现 ----

async def _read_file(path: str) -> str:
    try:
        p = _resolve_path(path)
    except PermissionError as e:
        return _err("permission_denied", str(e), "Use a path inside the workspace.")
    if not p.is_file():
        return _err("param_error", f"not a file: {path}", "Use list_dir to see contents.")
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return _err("param_error", f"read failed: {e}", "Check the path.")
    if not content:
        return "[File exists but is empty]"
    return _truncate(content)


async def _list_dir(path: str = ".", limit: int = 500) -> str:
    try:
        p = _resolve_path(path)
    except PermissionError as e:
        return _err("permission_denied", str(e), "Use a path inside the workspace.")
    if not p.is_dir():
        return _err("param_error", f"not a directory: {path}", "Use read_file for files.")
    try:
        entries = sorted(p.iterdir(), key=lambda x: x.name.lower())
    except OSError as e:
        return _err("param_error", f"list failed: {e}", "Check the path.")
    lines = [(e.name + ("/" if e.is_dir() else "")) for e in entries[:limit]]
    out = "\n".join(lines) if lines else "(empty directory)"
    if len(entries) > limit:
        out += f"\n[{limit} entries limit reached — {len(entries)} total]"
    return _truncate(out)


async def _get_time() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# ast 白名单求值：只放行算术运算符，禁裸 eval、禁属性/下标/调用。
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_ast(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_ast(node.operand))
    raise ValueError("disallowed expression")


async def _calculator(expr: str) -> str:
    try:
        result = _eval_ast(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return _err("param_error", f"invalid expression: {e}",
                    "Use only numbers and + - * / // % ** operators.")
    if isinstance(result, float):
        result = round(result, 10)
    return str(result)


async def _write_file(path: str, content: str) -> str:
    try:
        p = _resolve_path(path)
    except PermissionError as e:
        return _err("permission_denied", str(e), "Use a path inside the workspace.")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return _err("param_error", f"write failed: {e}", "Check the path and permissions.")
    return _ok({"path": str(p), "bytes_written": len(content.encode("utf-8"))})


async def _fetch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return _err("param_error", f"unsupported scheme: {parsed.scheme}",
                    "Use http:// or https://.")
    if not parsed.hostname:
        return _err("param_error", "missing host in URL", "Provide a full URL.")
    if _is_private_host(parsed.hostname):
        return _err("param_error", f"blocked host (internal/private): {parsed.hostname}",
                    "Only public hosts are reachable.")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
    except httpx.TimeoutException:
        return _err("transient", "fetch timed out", "Retry once.")
    except httpx.HTTPStatusError as e:
        return _err("param_error", f"HTTP {e.response.status_code}", "Check the URL.")
    except httpx.HTTPError as e:
        return _err("transient", f"fetch failed: {e}", "Retry once.")
    return _truncate(r.text, _MAX_FETCH_BYTES)


# ---- ToolDef 描述 ----

GENERIC_TOOLDEFS = [
    ToolDef(
        name="read_file",
        description=(
            "Read a text file inside the workspace. Returns the content, "
            "truncated if large. Relative paths resolve against the project root."
        ),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path (relative or absolute)."}},
            "required": ["path"],
        },
        source="generic",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="list_dir",
        description=(
            "List files and subdirectories in a directory. Directories are "
            "suffixed with '/'. Returns entries sorted alphabetically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list.", "default": "."},
                "limit": {"type": "integer", "description": "Max entries to return.", "default": 500},
            },
        },
        source="generic",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="get_time",
        description="Return the current local date and time.",
        parameters={"type": "object", "properties": {}},
        source="generic",
        annotations={"readOnlyHint": True, "idempotentHint": False},
    ),
    ToolDef(
        name="calculator",
        description=(
            "Safely evaluate an arithmetic expression. Supports numbers and "
            "+ - * / // % ** and parentheses. No variables, calls, or attribute access."
        ),
        parameters={
            "type": "object",
            "properties": {"expr": {"type": "string", "description": "Arithmetic expression, e.g. '2 * (3 + 4) ** 2'."}},
            "required": ["expr"],
        },
        source="generic",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="write_file",
        description=(
            "Write text content to a file, overwriting if it exists. Creates "
            "parent directories automatically. Destructive — requires authorization."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write (relative or absolute)."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
        source="generic",
        annotations={"readOnlyHint": False, "idempotentHint": True},
    ),
    ToolDef(
        name="fetch_url",
        description=(
            "Fetch the content of a public web URL over http/https. Returns the "
            "raw text, truncated if large. Internal/private hosts are blocked."
        ),
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Public URL to fetch."}},
            "required": ["url"],
        },
        source="generic",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
]

GENERIC_FUNCS = {
    "read_file": _read_file,
    "list_dir": _list_dir,
    "get_time": _get_time,
    "calculator": _calculator,
    "write_file": _write_file,
    "fetch_url": _fetch_url,
}


# 暴露白名单：研究 agent 只用文件 explorer 三件套。
# get_time / calculator / fetch_url 代码保留（通用能力、将来通用 agent 复用），
# 但不注册进工具面——对研究 agent 是死重。
_EXPOSED_TOOLS = {"read_file", "list_dir", "write_file"}


class GenericProvider(ToolProvider):
    """通用工具提供层 — 文件/网络/系统横切能力。

    只暴露 _EXPOSED_TOOLS 里的文件三件套；其余代码保留、不注册。
    """

    name = "generic"

    def __init__(self):
        self._tool_map = {td.name: td for td in GENERIC_TOOLDEFS}

    async def list_tools(self) -> list[ToolDef]:
        return [td for td in self._tool_map.values() if td.name in _EXPOSED_TOOLS]

    async def call_tool(self, name: str, arguments: dict):
        if name not in GENERIC_FUNCS:
            raise KeyError(f"Generic tool '{name}' not found")
        td = self._tool_map.get(name)
        if td is not None and not tool_allowed(td.annotations):
            return _err("permission_denied",
                        f"Action '{name}' is not authorized for the current role.",
                        "Tell the user this action requires higher authorization. Do not retry.")
        return await GENERIC_FUNCS[name](**arguments)
