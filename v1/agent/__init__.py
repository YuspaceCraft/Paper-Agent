"""
agent — Agent 核心层
=====
接受用户提问 → 意图路由 → create_agent → LoopResult + 记忆管理。

架构（v2.1 — 全结构化信息交换）：
  agent_query() → 意图路由(决定工具集) → System Prompt 构建
  → LangChain create_agent → 提取答案 → LoopResult

  信息交换路径（全部结构化）：
    工具 → LLM:     Pydantic model → JSON __str__() → ToolMessage
    LLM → 模块:     ```json_strict { answer, sources, confidence } ```
    模块 → 模块:    Intent / RewriteResult / ToolResultSummary dataclass

本层职责：
  - Agent 入口（意图路由 + create_agent 薄封装）
  - 工具封装（文献检索、工具过滤、scope 管理）
  - 对话记忆管理（短期窗口 + 长期持久化）
  - RAG 组件（检索、嵌入、生成、文档处理）

快速开始:
  from agent import agent_query, ingest, create_memory
  memory = create_memory("hybrid")
  result = agent_query("对比DenseNet和ResNet", memory)
  print(result.answer)
"""

from agent.agent import (
    UnifiedAgentLoop,
    agent_query,
    react_agent_query,
    plan_execute_query,
    ConfirmedPlan,
    ExecutionResult,
    AgentLoopContext,
    LoopResult,
)
from agent.ingest import ingest
from agent.memory import (
    create_memory,
    BaseMemory,
    BufferMemory,
    WindowMemory,
    HybridMemory,
    Message,
)
from agent.conversation import ConversationStore

__all__ = [
    # Agent
    "UnifiedAgentLoop",
    "agent_query",
    "react_agent_query",
    "plan_execute_query",
    "ConfirmedPlan",
    "ExecutionResult",
    "AgentLoopContext",
    "LoopResult",
    # Ingest
    "ingest",
    # Memory
    "create_memory",
    "BaseMemory",
    "BufferMemory",
    "WindowMemory",
    "HybridMemory",
    "Message",
    # Conversation
    "ConversationStore",
]
