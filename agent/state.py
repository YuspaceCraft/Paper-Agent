"""
state.py — AgentState + structured output models for the LangGraph agent.

v3: 3-way intent classification (literature_search / general_chat / needs_clarify)
    + confidence gate for clarify routing.
"""

import os

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from .config import get_limits


class UnderstandResult(BaseModel):
    """Structured output from the understand (router) node.

    v3: reduced from 4 intents to 3. Added confidence for clarify gating.
    v10: added domain for 领域级路由 (paper / creation / coding).
    """
    intent: str = Field(
        description="One of: literature_search, general_chat, needs_clarify, "
                    "task_query"
    )
    domain: str = Field(
        default="paper",
        description="Working domain: paper (research Q&A) | creation (writing tasks) "
                    "| coding (experiment/code tasks). Default 'paper' if unclear.",
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
    # 执行粒度双上限（v11）：
    #   max_steps — 单 turn 内最多执行多少轮 agent↔tools 往返（step 粒度）。
    #              只做安全网，不设常态限制：复杂工具编排（逐篇验证 20~30 步）
    #              是合法需求，撞上限那步也执行完再收尾，绝不中途丢弃已发请求。
    #              优先级 env AGENT_MAX_STEPS / AGENT_MAX_ITERATIONS > agent/config.yaml。
    #   max_turns — 会话级轮次上限（turn 粒度）：超过后 agent 不再调用工具，
    #              强制基于已有信息给出 final answer（配合 memory 摘要控制会话膨胀）。
    #              优先级 env AGENT_MAX_TURNS > agent/config.yaml。
    max_steps: int = int(
        os.getenv("AGENT_MAX_STEPS")
        or os.getenv("AGENT_MAX_ITERATIONS")
        or get_limits().max_steps
    )
    # turn_count 由 understand_node 每个新回合 +1（跨回合经 checkpointer 持久化）。
    turn_count: int = 0
    max_turns: int = int(os.getenv("AGENT_MAX_TURNS", str(get_limits().max_turns)))
    consecutive_failures: int = 0
    # ---- 工具调用去重缓存 (v15) ----
    # {f"{name}|{canonical_args}": {"content": str, "count": int}}。
    # build_tools_node 对单轮内「相同(工具,参数)」的重复调用复用上次结果,不再
    # 重新执行副作用(重搜同一 query / 重复入库等)。每次 node 更新整表——单轮
    # 工具调用数有限,规模可控。LangGraph 丢弃未声明的 key,必须在此声明。
    tool_result_cache: dict = {}
    # ---- Plan-and-Execute (Phase 7) ----
    # requested_mode: 客户端显式模式覆盖（随每次 input 透传，checkpoint 持久化）。
    # "auto"(默认, decide_mode 启发式) | "react" | "plan"。客户端每回合重发，
    # 无陈旧状态问题；graph.run() 等旧调用不传 → 默认 auto。
    requested_mode: str = "auto"
    # mode: 本回合实际采用的执行模式 "react" (single ReAct, default) | "plan"
    # (plan_node → executor → verify → synthesize)。decide_mode 填充。
    mode: str = "react"
    # plan: ordered steps from plan_node. [{id, description, target, args, depends_on}]
    # executor_node 回填每个 dict 的 status: pending/running/done/failed/skipped
    plan: list[dict] = []
    # plan_progress: number of plan steps completed by executor_node
    plan_progress: int = 0
    # subagent_results: executor output. [{step_id, ok, output, error}]
    subagent_results: list[dict] = []
    # verification: verify_node 输出（计划完成验证报告式结论）。
    # {status: satisfied|partial|failed|no_evidence, done, total,
    #  outstanding: [{id, description, reason}]}；synthesize 消费。
    verification: dict = {}
    # ---- Subagent runtime (Phase 8) ----
    # subagent_system: non-empty → agent_node uses this instead of AGENT_SYSTEM
    subagent_system: str = ""
    # bound_tools: non-empty → agent_node binds only these tool names (subagent restricted set)
    bound_tools: list[str] = []
    # ---- domain routing (v10) ----
    # Working domain: "paper" (research Q&A) | "creation" (writing) | "coding"
    # (experiments). understand_node labels it; domain_node rule-overrides it.
    # LangGraph schema discards updates for keys NOT declared here — missing
    # this field made the graph always fall back to "paper" (v10 断裂点).
    domain: str = "paper"
    # Active writing document (set by plan_node when the creation plan creates one).
    doc_id: str | None = None
    # ---- leader-department supervision (领导-部门制) ----
    # active_tasks: 本会话派发的子 agent 任务句柄缓存（{task_id, role, title}，
    # 跨 turn 累积，task_node 回写去重）。确定性解析「那个任务/刚才派发的」。
    active_tasks: list[dict] = []

    # ---- governance budget ----
    # tokens_used = 当前一次 agent_node LLM 调用的实际输入规模（system +
    # 全量 messages + 当轮反馈）。语义是「本次喂给模型的上下文有没有超过上限」：
    # 达到 token_budget 时 agent_node 强制产出 final answer（不带工具），
    # 保证终止且不把模型输入打到真实上下文窗口外。
    # 注意：不能把每轮输入累加——那会让多轮工具任务（逐篇 fetch_content
    # 验证）在 3~4 轮后超线性撞线，正常任务被提前截断（TROUBLESHOOTING
    # 「agent / 多工具任务被 token 预算提前截断」）。
    token_budget: int = int(os.getenv("AGENT_TOKEN_BUDGET", "60000"))
    tokens_used: int = 0
    # ---- conversation context (对话中心化重构) ----
    # context_node 每回合确定性重建，提供对话级工作区绑定与记忆：
    #   active_doc_id     — 本会话正在写的文档
    #   active_project    — 本会话正在进行的实验项目
    #   study_topic       — 本会话涉及的研究主题（study KB 键）
    #   recent_experiments— 本会话最近运行的实验 exp_id 列表（新→旧，cap 5）
    # 跨 turn 经 checkpointer 持久化；子 agent 保持零状态，记忆由父注入。
    context: dict = {}
