"""
tools.py — agent tool public API.

两级注册表：
  _BASE_TOOLS  — 完整工具注册表（builtin + generic + mcp + skill），
                 父 agent 从中挑选全部只读工具，subagent 按 SubagentSpec 挑受限子集。
  _ALL_TOOLS   — 父 agent 工具面：本地只读全套（文件 + 库检索/阅读）+ skills
                 + subagents（仅 arxiv / ingest 两个写/外网操作）。

用法：
    from agent.tools import ALL_TOOLS     # 向后兼容：指向 _ALL_TOOLS
    await ensure_tools()                  # 启动时初始化（连接 MCP、扫描 skills）
    tools = get_cached_tools()            # 父 agent 工具面
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

# 从 builtin_provider 重新导出核心工具函数（向后兼容）
from .providers.builtin_provider import search_papers, fetch_content  # noqa: F401

# ---- 工具缓存 ----

_ALL_TOOLS: list[BaseTool] = []     # 父 agent 工具面（见模块 docstring）
_BASE_TOOLS: list[BaseTool] = []    # 完整注册表（供 subagent 挑选）
_build_lock = asyncio.Lock()
_built = False


def get_cached_tools() -> list[BaseTool]:
    """返回已缓存的工具列表。需先 await ensure_tools()。"""
    return _ALL_TOOLS


def get_base_tools() -> list[BaseTool]:
    """返回完整工具注册表（builtin + generic + mcp + skill）。

    与 get_cached_tools() 的区别：后者返回父 agent 工具面（_ALL_TOOLS，
    不含 search_papers/fetch_content 等叶子工具）；前者返回全部叶子工具
    （_BASE_TOOLS），供 subagent 按名称挑选受限子集。
    """
    return _BASE_TOOLS


async def ensure_tools(mcp_config: str | Path | None = None,
                       skills_dir: str = "skills") -> list[BaseTool]:
    """初始化工具列表。幂等，首次调用连接 MCP server + 扫描 skills。

    Args:
        mcp_config: .mcp.json 路径，默认自动查找
        skills_dir: skills/ 目录路径
    """
    global _ALL_TOOLS, _BASE_TOOLS, _built
    if _built:
        return _ALL_TOOLS

    async with _build_lock:
        if _built:
            return _ALL_TOOLS

        from .providers import CompositeToolProvider
        from .providers.builtin_provider import BuiltinProvider
        from .providers.generic_provider import GenericProvider
        from .providers.mcp_provider import MCPProvider, load_mcp_config
        from .providers.skill_provider import SkillProvider
        from .domains.creation import CreationProvider
        from .domains.coding import CodingProvider

        providers = [
            BuiltinProvider(),
            CreationProvider(),   # 创作域 doc 工具（creator subagent 可见）
            CodingProvider(),     # 编码域实验/git/delegate/study 工具（coder subagent 可见）
            GenericProvider(),
            MCPProvider(load_mcp_config(mcp_config)),
            SkillProvider(skills_dir),
        ]

        composite = CompositeToolProvider(providers)
        tooldefs = await composite.list_all()

        # 统一调度器：超时 + 重试 + 审计，覆盖所有 provider 的工具
        from .dispatcher import ToolDispatcher
        dispatcher = ToolDispatcher(composite.call_tool, tooldefs)

        # 完整注册表（builtin + generic + mcp + skill），供 subagent 挑选子集。
        _BASE_TOOLS = [_to_langchain_tool(td, dispatcher.call) for td in tooldefs]

        # subagent 工具：从完整注册表按 SubagentSpec 配置注入各自权限工具。
        from .subagents import build_subagents
        base_map = {t.name: t for t in _BASE_TOOLS}
        subagent_tools = build_subagents(base_map)

        # 父 agent 工具面 = 文件 explorer 三件套 + 本地三态检测 + 后台任务查询
        # + 库只读工具 + skills + subagents（Claude Code 模式：全能父 agent 持有
        # 全部只读工具）。search_papers/fetch_content 直接由父 agent 调用（列库/
        # 读论文内容），避免"列目录/对比"类任务被迫经 subagent 中转。
        # 仅写/外网操作隔离在 subagent：ingest（下载/入库，原子性）、arxiv（外网）。
        # check_paper 是入库决策梯的确定性第一步（Redis→fs 本地快检），
        # check_task_status 是后台任务栈查询（用户随时可能问"入库好了吗"），
        # 都必须由父 agent 直接调用，故显式列入。
        PARENT_NAMES = {"read_file", "list_dir", "write_file", "check_paper",
                        "check_task_status", "search_papers", "fetch_content",
                        "skill__list", "skill__load"}
        parent_tools = [t for t in _BASE_TOOLS if t.name in PARENT_NAMES]
        parent_tools += subagent_tools

        # 原地清空/扩展，保持 ALL_TOOLS 与 _ALL_TOOLS 别名一致。
        _ALL_TOOLS.clear()
        _ALL_TOOLS.extend(parent_tools)
        _built = True
        return _ALL_TOOLS


# ---- 向后兼容导出 ----

# ponytail: ALL_TOOLS 和 _ALL_TOOLS 指向同一列表，ensure_tools() 惰性填充。
ALL_TOOLS: list[BaseTool] = _ALL_TOOLS


# ---- 内部: ToolDef → LangChain StructuredTool ----

def _to_langchain_tool(td, call_fn) -> BaseTool:
    """将 ToolDef 转换为 LangChain StructuredTool。

    动态生成 args_schema Pydantic 模型。对于无参工具（如 skill__list），
    使用空模型。
    """
    from pydantic import BaseModel, ConfigDict, Field, create_model

    parameters = td.parameters or {}
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    if not props:
        # 无参工具
        class _EmptyArgs(BaseModel):
            pass

        async def _call_empty() -> str:
            return str(await call_fn(td.name, {}))

        return StructuredTool(
            name=td.name,
            description=td.description,
            args_schema=_EmptyArgs,
            coroutine=_call_empty,
        )

    # 动态生成 Pydantic 模型
    fields: dict = {}
    for prop_name, prop_schema in props.items():
        prop_type = prop_schema.get("type", "string")
        is_required = prop_name in required
        # 必填 → 无默认；选填 → 用 schema default（如 search_papers.query=""），
        # 否则 None。尊重 JSON Schema 的 default，保证 builtin 工具的默认参数不丢失。
        default = ... if is_required else prop_schema.get("default")

        py_type = {
            "string": str, "integer": int, "number": float,
            "boolean": bool, "array": list, "object": dict,
        }.get(prop_type, str)

        desc = prop_schema.get("description", "")
        fields[prop_name] = (py_type, Field(default, description=desc))

    # 严格校验：未知参数（如 LLM 误传 task）直接报错，而非被 Pydantic 静默
    # 丢弃后落到默认值（例如 search_papers 的 query 默认 "" → 变成列出全部论文）。
    args_model = create_model(
        f"{td.name}_args",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )

    async def _call(**kwargs) -> str:
        return str(await call_fn(td.name, kwargs))

    return StructuredTool(
        name=td.name,
        description=td.description,
        args_schema=args_model,
        coroutine=_call,
    )
