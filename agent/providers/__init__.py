"""
agent/providers/ — 统一工具提供层

ToolProvider ABC → 所有工具源遵循同一接口。
CompositeToolProvider → 聚合并去重。

三个实现：
- BuiltinProvider: 论文核心能力（search_papers, fetch_content）
- MCPProvider: 社区 MCP server 工具（读 .mcp.json）
- SkillProvider: SKILL.md 开放标准技能（渐进式三阶段加载）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDef:
    """统一工具描述，兼容 MCP 和 Skills 开放标准。

    MCP Tool.annotations 映射到 annotations 字段：
    - readOnlyHint / destructiveHint / idempotentHint / openWorldHint
    """

    name: str
    description: str
    parameters: dict  # JSON Schema
    source: str  # "mcp" | "skill" | "builtin"
    annotations: dict = field(default_factory=dict)


class ToolProvider(ABC):
    """所有工具源的标准接口。

    渐进式发现三阶段：
    1. list_tools() → 返回 name + description（轻量，用于模型选择工具）
    2. （隐式）模型选中工具后，call_tool() 直接调用
    3. 对 SkillProvider：load_skill() 返回完整指示，read_resource() 按需加载
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def list_tools(self) -> list[ToolDef]: ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict) -> Any: ...

    async def close(self) -> None:
        """可选：释放资源（MCP 连接等）"""
        pass


class CompositeToolProvider:
    """聚合多个 provider，按优先级去重。

    去重规则：同名工具保留第一个 provider 的版本。
    provider 列表顺序即优先级：builtin > mcp > skill。
    """

    def __init__(self, providers: list[ToolProvider]):
        self._providers = providers
        self._tool_map: dict[str, tuple[ToolDef, ToolProvider]] = {}

    async def _build_index(self) -> None:
        """构建 tool_name → (ToolDef, provider) 索引。去重按优先级。"""
        if self._tool_map:
            return
        seen: set[str] = set()
        for p in self._providers:
            try:
                tools = await p.list_tools()
            except Exception:
                continue  # provider 挂了不影响其他
            for td in tools:
                if td.name not in seen:
                    seen.add(td.name)
                    self._tool_map[td.name] = (td, p)

    async def list_all(self) -> list[ToolDef]:
        """返回去重后的全部工具描述（仅 name + description + source）。"""
        await self._build_index()
        return [td for td, _ in self._tool_map.values()]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """调用工具，路由到对应的 provider。"""
        await self._build_index()
        if name not in self._tool_map:
            raise KeyError(f"Tool '{name}' not found")
        _, provider = self._tool_map[name]
        return await provider.call_tool(name, arguments)

    async def close(self) -> None:
        """释放所有 provider 资源。"""
        for p in self._providers:
            try:
                await p.close()
            except Exception:
                pass
        self._tool_map.clear()
