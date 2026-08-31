"""
state.py — AgentState + structured output models for the LangGraph agent.

v3: 3-way intent classification (literature_search / general_chat / needs_clarify)
    + confidence gate for clarify routing.
"""

import os

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class UnderstandResult(BaseModel):
    """Structured output from the understand (router) node.

    v3: reduced from 4 intents to 3. Added confidence for clarify gating.
    """
    intent: str = Field(
        description="One of: literature_search, general_chat, needs_clarify"
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Key concepts, methods, or paper name terms (max 5)",
    )
    focus_papers: list[str] = Field(
        default_factory=list,
        description="Specific paper name terms the user wants to read",
    )
    confidence: float = Field(
        default=1.0,
        description="0.0-1.0 confidence in the intent classification. "
                    "Below 0.5 triggers clarification instead of search.",
    )


class AgentState(MessagesState):
    """Agent state extending LangGraph's MessagesState.

    Layered design:
      - User layer: intent, entities, focus_papers (what the user said)
      - Router layer: confidence (how sure the classification is)
      - Memory layer: context_snapshot, summary_cache (structured conversation context)
      - Resolution layer: resolved (what the system matched, with confidence)
      - Execution layer: iteration, failures (what happened)
    """
    intent: str = ""
    confidence: float = 1.0
    entities: list[str] = []
    focus_papers: list[str] = []
    # Memory layer — added by memory_node
    # Compact snapshot for downstream nodes. Built by MemoryManager.build_snapshot().
    context_snapshot: str = ""
    # Cached LLM summary of messages older than MemoryManager.BUFFER_SIZE.
    summary_cache: str = ""
    # How many older messages have been summarized (index into messages list).
    summary_through_seq: int = 0
    # Resolution layer — added by resolve_node
    # {papers: [{query, match, confidence, level, match_type}], section: {ordinal, text, confidence} | None}
    resolved: dict = {}
    iteration: int = 0
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    consecutive_failures: int = 0
    # ---- Plan-and-Execute (Phase 7) ----
    # mode: "react" (single ReAct, default) | "plan" (plan_node → executor → synthesize)
    mode: str = "react"
    # plan: ordered steps from plan_node. [{id, description, target, args, depends_on}]
    plan: list[dict] = []
    # plan_progress: number of plan steps completed by executor_node
    plan_progress: int = 0
    # subagent_results: executor output. [{step_id, ok, output, error}]
    subagent_results: list[dict] = []
    # ---- Subagent runtime (Phase 8) ----
    # subagent_system: non-empty → agent_node uses this instead of AGENT_SYSTEM
    subagent_system: str = ""
    # bound_tools: non-empty → agent_node binds only these tool names (subagent restricted set)
    bound_tools: list[str] = []
    # ---- governance budget ----
    # tokens_used accumulates _estimate_tokens() across agent_node LLM calls
    # within one turn. When it reaches token_budget, agent_node forces a
    # final answer (no tools) to guarantee termination.
    token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "20000"))
    tokens_used: int = 0
