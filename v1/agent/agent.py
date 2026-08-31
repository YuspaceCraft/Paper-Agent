"""agent.py — Agent 核心入口
========
基于 LangChain create_agent 的薄封装。

  agent_query()           — 主入口（意图路由 + create_agent）
  react_agent_query()     — ReAct 模式（委托给 agent_query）
  plan_execute_query()    — Plan-Execute 模式（委托给 agent_query）

架构：
  用户提问 → 意图路由(决定工具集) → 构建 System Prompt
  → create_agent(model, tools, system_prompt) → 提取答案 → LoopResult

  create_agent 原生处理 Thought→Action→Observation 循环，
  包括工具选择、结果解析、信息不足时的补充检索、最终答案生成。

使用方式：
  from agent.agent import agent_query
  result = agent_query("对比 DenseNet 和 Change-Agent 的方法", memory)
  print(result.answer)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable

from config import (
    AGENT_MAX_ITERATIONS,
    AGENT_VERBOSE,
    AGENT_LLM_TEMPERATURE,
    LLM_MODEL,
    OPENAI_API_KEY,
    DASHSCOPE_BASE_URL,
    DEBUG_LOG_ENABLED,
    DEBUG_LOG_DIR,
    DEBUG_LOG_MAX_FILES,
)

from agent.memory import BaseMemory


# ================================================================
#  ConfirmedPlan — PLANNING 阶段输出
# ================================================================

@dataclass
class ConfirmedPlan:
    """问题确认阶段的输出。"""
    original_question: str = ""
    rewritten_question: str = ""      # 指代消解后的查询
    query_type: str = "general"       # fact / review / compare / general
    intent_type: str = "knowledge_retrieval"
    retrieval_scope: str = "local"    # local / online / hybrid
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification_hint: str = ""
    sub_steps: list[str] = field(default_factory=list)  # 子任务拆解
    reasoning: str = ""
    active_skill: str = ""  # Skills 系统激活的技能名称


# ================================================================
#  ExecutionResult — EXECUTING 阶段输出（向后兼容）
# ================================================================

@dataclass
class ExecutionResult:
    """任务执行阶段的输出。"""
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)
    context: str = ""  # 检索上下文（用于反思阶段检查忠实度）


# ================================================================
#  LoopResult — Agent 循环最终输出（结构化，供 web/CLI 使用）
# ================================================================

@dataclass
class LoopResult:
    """Agent 循环的最终结构化结果。

    替代旧的纯字符串返回值，让调用方可以区分：
      - 正常回答 (status="answer")
      - 需要澄清 (status="clarify_scope" / "clarify_general")
      - 超出领域 (status="out_of_domain")
      - 闲聊 (status="general_chat")
    """
    status: str = "answer"  # "answer" | "clarify_scope" | "clarify_general" | "out_of_domain" | "general_chat"
    answer: str = ""
    clarification_message: str = ""
    retrieved_docs: list = field(default_factory=list)
    rewritten_query: str = ""
    query_type: str = "general"
    debug_log_path: str = ""  # 调试日志文件路径
    pending_plan: dict | None = None  # 澄清挂起时保存的 ConfirmedPlan（序列化），供 resume 使用


# ================================================================
#  AgentLoopContext — Loop 运行时可变状态
# ================================================================

@dataclass
class AgentLoopContext:
    """
    Loop 内可变状态，与不变的 system_prompt 分离。

    消息序列严格遵循角色规范:
      SystemMessage (system_prompt)
      HumanMessage (user_question)
      AIMessage (plan / intermediate reasoning)
      ToolMessage (tool result summaries — 不是原始输出!)
      ...
      AIMessage (final answer)
    """
    messages: list = field(default_factory=list)         # 严格角色消息序列
    step_results: list[dict] = field(default_factory=list)  # [{step, description, tool, summary, raw_result}]
    error_history: list[str] = field(default_factory=list)   # 连续错误指纹
    retry_count: dict[str, int] = field(default_factory=dict)  # step_id → 重试次数
    completed_steps: set = field(default_factory=set)       # 已成功完成的步骤编号（跨迭代）
    iteration: int = 0
    phase: str = "idle"  # "planning"|"executing"|"reflecting"|"responding"|"waiting_for_input"
    supplement_count: int = 0  # 补充检索触发次数
    phase_data: dict = field(default_factory=dict)  # Phase 3: 类型化的阶段产出（不序列化为 text 注入 messages）


# ================================================================
#  TaskProgress — 任务执行状态机（跨迭代、跨轮次追踪）
# ================================================================

@dataclass
class TaskProgress:
    """
    追踪任务执行的实际状态，解决 agent 在反思/修正阶段因缺少操作记录而编造结果的问题。

    现在是 UnifiedSTM 的轻量视图 — 所有查询方法从 stm.task_stack 派生，
    不再维护独立的操作列表，消除双写导致的数据不一致。

    使用方式:
        tp = TaskProgress(stm)  # stm 为 UnifiedSTM 实例
    """
    intent_type: str = "knowledge_retrieval"
    _stm: object = None  # UnifiedSTM 引用

    def __post_init__(self):
        # 支持旧构造方式（无 stm 参数）的向后兼容
        pass

    @property
    def operations(self) -> list[dict]:
        """从 STM 的任务栈动态生成操作记录视图。"""
        if self._stm is None:
            return []
        return self._stm.task_stack.get_execution_records()

    def record(self, tool_name: str, args: dict, result_summary: str, success: bool = True):
        """
        记录一次工具操作。现在为空操作 — STM 已通过 observe/update_step
        记录了所有执行信息，TaskProgress 不再维护独立存储。
        """
        pass

    def get_file_snapshot(self) -> str:
        """获取最近的文件系统快照（来自 list_directory / list_paper_categories）。"""
        if self._stm is None:
            return "（尚未获取文件系统快照）"
        snapshot_ops = [
            op for op in self.operations
            if op["tool"] in ("list_directory", "list_paper_categories", "search_files")
            and op["success"]
        ]
        if not snapshot_ops:
            return "（尚未获取文件系统快照）"
        latest = snapshot_ops[-1]
        return latest["result_summary"]

    def get_completed_actions(self) -> str:
        """获取已完成的操作摘要（供反思/修正使用）。"""
        ops = self.operations
        if not ops:
            return "（尚无已完成的操作）"
        lines = []
        for i, op in enumerate(ops, 1):
            status = "✅" if op["success"] else "❌"
            lines.append(
                f"  {i}. [{op['tool']}] {status} {op['result_summary'][:150]}"
            )
        return "\n".join(lines)

    def has_operation(self, tool_name: str) -> bool:
        """检查是否已执行过某类操作。"""
        return any(op["tool"] == tool_name for op in self.operations)

    @staticmethod
    def _summarize_args(args: dict) -> str:
        if not args:
            return ""
        items = []
        for k, v in args.items():
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            items.append(f"{k}={v_str}")
        return ", ".join(items[:4])


# ================================================================
#  系统提示模板
# ================================================================

# ═══════════════════════════════════════════════════════════════
# System Prompt（拆分为不可变核心 + 动态上下文）
# ═══════════════════════════════════════════════════════════════
# Phase 2: 不可变核心规则始终在 SystemMessage 中（~200 tokens），
# 动态上下文（scope + tools + 文件管理/skill）在每次执行推理时注入。

IMMUTABLE_SYSTEM_PROMPT = """\
你是专业的科研文献分析助手 Agent。你拥有两大核心能力：文献检索 + 文件管理。

**核心规则（必须始终遵守）：**
1. **先检索再回答**: 必须调用工具获取信息后才能回答，禁止凭空编造
2. **诚实面对失败**: 检索无结果 → 如实告知用户，绝不虚构内容
3. **标注来源**: 所有事实陈述必须标注出处（文件名/作者/年份）
4. **工具失败不重试**: 同一工具+同一参数失败后，换工具或换策略，不无脑重试
5. **完成回答后不再调用工具**
6. **结构化输出**: 对比类问题使用表格，复杂回答使用标题和列表
7. **路径判断先于行动**: 先判断用户意图（学术/文件管理/无关/模糊），再选择执行路径
8. **工具输出均为 JSON**: 所有工具返回的都是结构化 JSON，请直接读取字段值（如 hit_count、sources、status、item_count 等），不要试图解析纯文本格式

**工具输出格式说明：**
工具返回的是紧凑 JSON 对象，固定包含 "type" 字段标识类型。关键字段：
- `SearchResult` → hit_count(整数), sources(文件名列表), paper_titles, quality_warning, summary
- `FileListResult` → item_count, directories(列表), files(列表), summary
- `FileOperationResult` → operation, target, status("ok"|"error"|"exists"), details, stats, summary
- `MemoryResult` → count, items(列表), keywords, source, memory_id, summary
- `SystemStatusResult` → vector_count, embedding_model, llm_model, ltm_count, summary
- `QueryRewriteResult` → original, rewritten, query_type, needs_rewrite(布尔), summary
- `summary` 字段包含人类可读的详细摘要，可用于提取具体的论文内容、文件详情等

**最终回答格式要求：**
完成所有工具调用后，你的最终回答必须使用以下格式（用 ```json_strict 包裹）：

```json_strict
{
  "answer": "你的完整回答（Markdown 格式，可含标题/列表/表格）",
  "sources": ["来源1", "来源2"],
  "confidence": "high|medium|low",
  "retrieval_status": "success|partial|empty|not_needed"
}
```

其中：
- answer: 给用户的完整回答，支持 Markdown
- sources: 所有引用来源的列表（文件名/作者），无来源时为空列表
- confidence: high=充分证据，medium=部分证据，low=证据不足
- retrieval_status: success=检索成功，partial=部分成功，empty=无结果，not_needed=无需检索（闲聊/文件操作等）

你是严谨、专业的科研助手，始终基于证据回答问题。"""

# 保留完整的 SYSTEM_PROMPT_TEMPLATE 供 _get_dynamic_context 和向后兼容
# （不再直接用作 SystemMessage，而是拆分为 immutable + dynamic）
_SYSTEM_PROMPT_TEMPLATE = """\
你是专业的科研文献分析助手 Agent。你拥有两大核心能力：文献检索 + 文件管理。

**工作流程：先判断意图，再选择执行路径。**

收到用户问题后，你必须按以下优先级处理：

### 第一步：分析问题意图
先判断用户问题属于哪一类：
- **A. 学术科研问题**：涉及论文方法、实验、模型、数据集、文献对比、综述等
- **B. 论文文件管理**：整理、分类、移动、搜索论文 PDF 文件
- **C. 明显无关问题**：天气、股票、娱乐八卦、生活常识等与学术研究无关的闲聊
- **D. 边界模糊问题**：可能相关但措辞含糊（如过于宽泛的"讲讲机器学习"）

### 第二步：按意图选择执行路径

**路径 A — 学术科研问题：**
1. 先调用检索工具（search_literature / search_papers / get_paper_detail 等）获取文献证据
2. 基于检索结果回答问题，绝不凭空编造
3. 所有事实陈述必须标注来源（文件名或作者年份）
4. 检索结果不足以回答时，明确说"现有文献未提及该细节"

**路径 B — 论文文件管理：**
1. 先调用 list_directory 或 list_paper_categories 了解现状
2. 制定操作计划并告知用户
3. 获得确认后执行（move_file / organize_paper / batch_classify 等）
4. 绝不猜测文件路径或文件名

**路径 C — 明显无关问题：**
直接诚实告知用户：「抱歉，我是科研文献助手，专注于已索引论文文献的检索与分析。」同时简要说明你能提供的帮助（检索论文、对比分析、整理论文等）。**无需调用任何检索工具。**

**路径 D — 边界模糊问题：**
简短追问 1-2 句澄清用户真实意图（如"您是想了解某篇具体论文中的方法，还是想了解该领域的整体现状？"）。**无需调用检索工具。** 澄清意图后再走对应路径。

### 检索与停止规则（仅适用于路径 A）

- **首次检索**：选择最合适的工具，使用精准的查询词
- **检索失败处理**：
  - 返回"未找到"或"[ERR] 向量数据库为空" → 诚实告知用户本地知识库未覆盖该内容，给出建议
  - 返回"[WARN] 检索质量警告" → 说明索引内容不完整
  - **检索尝试应适度**：首次检索无结果时尝试换关键词或换工具重试。若多次（通常 2-3 次）仍无有效结果，则停止并诚实告知用户当前知识库未覆盖该内容，给出建议（上传 PDF / 切换到在线检索模式）。对于需要多维度检索的复杂问题（如对比多篇论文），可按子任务需要分别检索，不在此限
- **截断内容不可用**：检索到的文档块仅有文件名/标题而无实质段落文本（< 100 字符），视为检索失败
- **不要重复相同查询**：首次检索无结果时，第二次尝试应使用不同关键词或换用其他检索工具；两次均失败则停止

### 通用规则

- **引用标注**：所有事实陈述必须标注来源（文件名或作者年份）
- **结构化输出**：对比类问题使用表格，复杂回答使用标题和列表
- **善用工具**：
  - 模糊指代（"这篇"、"该方法"）→ 调用 rewrite_query 优化查询
  - 需要历史讨论 → 调用 get_conversation_context
  - 对比多篇论文 → 使用 compare_papers
  - 需要特定论文详情 → 使用 get_paper_detail
- **主动记忆**：当用户明确表达偏好（如"我喜欢简洁的回答"、"以后都用中文"）、重要上下文（如当前研究方向、关注领域）或从检索中获得的值得记住的关键结论时，调用 add_to_memory 保存到长期记忆。对话轮次 >= 3 时，调用 search_long_term_memory 检查是否有相关历史偏好
- **工具失败处理**：摘要显示工具执行失败时，分析失败原因后采取不同策略——配置/权限类错误直接告知用户；查询无结果可尝试换关键词重试 1 次；工具未找到或参数错误则换用替代工具。不要无脑重复调用同一工具
- 完成回答后直接结束，不再调用工具

{scope_instruction}

{tools_description}

你是严谨、专业的科研助手，始终基于证据回答问题。"""


_SCOPE_INSTRUCTIONS = {
    "local": """**当前检索范围：📂 本地知识库**
你有以下本地检索工具可用：
- search_literature — 在已上传/已索引的 PDF 论文中检索内容
- get_paper_detail — 按标题/作者精确查找本地论文
- compare_papers — 对比分析本地多篇论文

策略：
1. 对于学术科研问题，调用 search_literature 检索本地知识库——即使问题中提到 arXiv/在线/最新等词，本地知识库可能也有相关论文
2. 基于检索结果回答。如果确实未找到，诚实告知用户本地未找到
3. 如果用户明显需要在线检索（如 arXiv），在本地检索后建议用户切换到在线检索模式
4. 对于明显与学术无关的问题，直接告知用户你的能力范围，无需调用检索工具""",

    "online": """**当前检索范围：🌐 在线检索 (arXiv)**
你有以下在线检索工具可用：
- search_papers — 在 arXiv 上按关键词/作者/分类搜索论文
- get_paper_data — 获取 arXiv 论文的详细元数据
- get_full_paper_text — 下载并阅读 arXiv 论文全文
策略：使用在线工具从 arXiv 检索最新论文。本地知识库不参与本次查询。""",

    "hybrid": """**当前检索范围：🔀 本地 + 在线 (Hybrid)**
你同时拥有本地和在线检索工具：
- 本地：search_literature, get_paper_detail, compare_papers
- 在线：search_papers, get_paper_data, get_full_paper_text
策略：1. 先用 search_literature 查本地 2. 不足再用 search_papers 在线补充 3. 整合标注来源""",
}


# ---- 文件管理专用规则（覆盖通用 prompt 中的检索优先规则）----
_FILE_MANAGEMENT_RULES = """

**当前任务：📁 论文文件管理**
你正在执行文件管理任务。你的可用工具仅限于文件系统操作工具。

**1. 先观察文件结构，再行动**
- 首先调用 list_directory 或 list_paper_categories 了解当前有哪些论文和目录
- 使用 search_files 查找 PDF 文件的位置

**2. 论文分类方法（按主题归类时）**
- ✅ 直接根据 **PDF 文件名中的关键词** 推断分类主题
  文件名通常包含完整论文标题，足以判断主题归属
- ✅ 使用 batch_classify 工具按文件名模式批量归类
- 如果文件名无法推断 → 归入 uncategorized/ 目录，在报告中标注

**3. 批量操作流程**
- 先调用 list_directory 或 list_paper_categories 了解文件结构
- 基于实际文件数据，自主规划后续操作（创建目录、批量归类、验证结果等）
- 你可以在初始计划中直接安排「观察 → 分类 → 验证」的完整流程，
  只要观察步骤排在操作步骤之前即可——先看有什么文件，再决定怎么分类

**4. 🚫 禁止行为**
- 不要编造文件名或目录名 — 基于实际观察结果操作
- 不要在不知道文件列表时生成具体的分类步骤 — 先观察，后分类
- 空结果绝不编造：如果 list_directory 返回空，如实说"目录为空"

**5. 回答格式**
最终回答只包含：
- 已创建的分类目录列表
- 已移动的论文及目标目录
- 当前分类统计
- 如有操作失败，明确标注并说明原因"""



# ================================================================
#  创建 Agent LLM
# ================================================================

def _create_agent_llm(temperature: float | None = None):
    """创建 Agent 决策用的 LLM 实例。"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        model=LLM_MODEL,
        temperature=temperature if temperature is not None else AGENT_LLM_TEMPERATURE,
    )


# ================================================================
#  结构化答案解析
# ================================================================

def _parse_structured_answer(raw_text: str) -> dict:
    """从 LLM 输出中提取结构化 JSON 答案。

    支持两种格式：
      1. ```json_strict { ... } ``` 代码块（推荐）
      2. 纯 JSON 文本

    返回 dict 含字段: answer, sources, confidence, retrieval_status。
    解析失败时返回 {"answer": raw_text} 作为回退。
    """
    if not raw_text or not raw_text.strip():
        return {"answer": raw_text or ""}

    text = raw_text.strip()

    # 格式 1: 提取 ```json_strict ... ``` 代码块
    block_match = re.search(
        r'```(?:json_strict|json)\s*\n?(.*?)\n?```',
        text, re.DOTALL | re.IGNORECASE,
    )
    if block_match:
        try:
            data = json.loads(block_match.group(1).strip())
            return {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "confidence": data.get("confidence", "medium"),
                "retrieval_status": data.get("retrieval_status", "unknown"),
            }
        except (json.JSONDecodeError, AttributeError):
            pass

    # 格式 2: 整个文本就是 JSON
    # 尝试提取首尾的 {...}
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group())
            if "answer" in data:
                return {
                    "answer": data.get("answer", ""),
                    "sources": data.get("sources", []),
                    "confidence": data.get("confidence", "medium"),
                    "retrieval_status": data.get("retrieval_status", "unknown"),
                }
        except (json.JSONDecodeError, AttributeError):
            pass

    # 回退：返回原始文本作为 answer
    return {"answer": raw_text}


# ================================================================
#  LangSmith 辅助：运行时元数据注入
# ================================================================

def _set_langsmith_tags(
    question: str,
    scope: str | None,
    intent_type: str | None,
    phase: str = "new",
) -> None:
    """向当前 LangSmith Run 注入运行时标签和元数据。

    调用时机：意图路由完成后（intent_type + scope 已知），
    在 create_agent 执行前将上下文信息写入 LangSmith Trace，
    便于在 LangSmith 面板中按意图/scope/问题过滤和分组。

    Args:
        question:   用户原始提问（截取前 200 字符作为标签）
        scope:      检索范围（local/online/hybrid）
        intent_type: 意图类型（knowledge_retrieval/file_management/etc.）
        phase:      阶段标记（new=首次执行 / resume=澄清恢复）
    """
    try:
        from langsmith.run_helpers import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree is None:
            return  # 不在 trace 上下文中（如 LangSmith 未启用）

        # 设置 tags（可在 LangSmith 面板中按 tag 过滤）
        tags = ["agent-v2"]
        if intent_type:
            tags.append(f"intent:{intent_type}")
        if scope:
            tags.append(f"scope:{scope}")
        tags.append(f"phase:{phase}")
        run_tree.tags = run_tree.tags + tuple(tags)

        # 设置 metadata（可在 LangSmith 面板中查看详情）
        metadata_updates = {
            "question_preview": question[:200],
            "scope": scope or "unknown",
            "intent_type": intent_type or "unknown",
            "entry_phase": phase,
        }
        if run_tree.metadata:
            run_tree.metadata.update(metadata_updates)
        else:
            run_tree.metadata = metadata_updates

    except Exception:
        pass  # LangSmith 不可用时静默降级，不影响主流程


# ================================================================
#  UnifiedAgentLoop — 基于 create_agent 的薄封装
# ================================================================

class UnifiedAgentLoop:
    """
    统一 Agent 循环（基于 LangChain create_agent 的薄封装）。

    架构:
      用户提问 → 意图路由(决定工具集) → 构建 System Prompt
      → create_agent(model, tools, system_prompt) → 提取答案 → LoopResult

      create_agent 原生处理 Thought→Action→Observation 循环，
      包括工具选择、结果解析、信息不足时的补充检索、最终答案生成。

    使用方式:
        loop = UnifiedAgentLoop()
        result = loop.run("对比 DenseNet 和 ResNet", memory,
                          progress_callback=my_callback)
    """

    def __init__(self):
        self._max_iterations = AGENT_MAX_ITERATIONS
        self._verbose = AGENT_VERBOSE
        # System Prompt 缓存: scope → 构建好的完整 prompt（初始化一次，循环中只读）
        self._system_prompts: dict[str, str] = {}

    # ================================================================
    #  System Prompt 管理（缓存，构建一次）
    # ================================================================

    def _get_system_prompt(self, retrieval_scope: str, intent_type: str = "knowledge_retrieval",
                          active_skill: str = "") -> str:
        """获取缓存的 System Prompt（每个 (scope, intent, skill) 组合构建一次，后续只读）。"""
        cache_key = f"{retrieval_scope}:{intent_type}:{active_skill}"
        if cache_key not in self._system_prompts:
            self._system_prompts[cache_key] = self._build_system_prompt(retrieval_scope, intent_type, active_skill)
        return self._system_prompts[cache_key]

    def _build_system_prompt(self, retrieval_scope: str, intent_type: str = "knowledge_retrieval",
                            active_skill: str = "") -> str:
        """构建 System Prompt（Phase 2: 仅返回不可变核心规则，~200 tokens）。

        动态上下文（scope + tools + 文件管理/skill）移至 _get_dynamic_context()，
        在执行推理时注入到 compact_observation 中，不再占用 SystemMessage 空间。"""
        return IMMUTABLE_SYSTEM_PROMPT

    def _get_dynamic_context(self, retrieval_scope: str, intent_type: str = "knowledge_retrieval",
                             active_skill: str = "") -> str:
        """构建动态上下文规则（scope + tools + 文件管理/skill）。

        在执行推理时注入到 compact_observation 末尾，
        与不可变 System Prompt 分离，实现「规则分层」:
          - SystemMessage: 永远不变的核心约束（~200 tokens）
          - 动态上下文:   随 scope/intent/skill 变化的操作指令（~200-400 chars）
        """
        parts = []

        # 1. Scope instruction（精简版，取前 300 chars）
        scope_instruction = _SCOPE_INSTRUCTIONS.get(retrieval_scope, _SCOPE_INSTRUCTIONS["local"])
        # 只保留策略部分，截断冗长的工具列表（tools_description 已单独列出）
        scope_short = scope_instruction[:300]
        if len(scope_instruction) > 300:
            scope_short = scope_short[:scope_short.rfind("\n")] + "\n..."
        parts.append(scope_short)

        # 2. Tools description
        tools_description = self._describe_tools(retrieval_scope, intent_type,
                                                  active_skill=active_skill)
        if tools_description:
            parts.append(tools_description[:400])

        # 3. 文件管理意图：追加专用规则
        if intent_type == "file_management":
            parts.append(_FILE_MANAGEMENT_RULES[:500])

        # 4. Skills 系统：追加激活技能的专用指令
        if active_skill:
            try:
                from skills import skill_registry
                from config import SKILLS_MAX_INSTRUCTIONS_CHARS
                skill_instructions = skill_registry.get_skill_instructions(
                    active_skill, max_chars=SKILLS_MAX_INSTRUCTIONS_CHARS,
                )
                if skill_instructions:
                    parts.append(
                        f"\n**🎯 当前激活技能：{active_skill}**\n"
                        f"{skill_instructions[:500]}"
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"无法加载技能 '{active_skill}' 的指令: {e}"
                )

        return "\n\n".join(parts)

    def _describe_tools(self, retrieval_scope: str, intent_type: str = "knowledge_retrieval",
                       active_skill: str = "") -> str:
        """生成工具列表描述（按意图过滤，简洁版，附在 System Prompt 末尾）。"""
        try:
            from agent.tools import get_tools_for_intent_and_scope
            tools = get_tools_for_intent_and_scope(intent_type, retrieval_scope,
                                                   active_skill=active_skill)
            if not tools:
                return ""
            lines = ["**可用工具列表：**"]
            for t in tools:
                desc = (t.description or "")[:100]
                lines.append(f"- `{t.name}`: {desc}")
            return "\n".join(lines)
        except Exception:
            return ""

    # ================================================================
    #  run — 主入口（同步）
    # ================================================================

    @traceable(
        run_type="chain",
        name="agent_run",
        metadata={
            "framework": "langchain_create_agent",
            "agent_loop_version": "2.0",
        },
    )
    def run(
        self,
        question: str,
        memory: BaseMemory | None = None,
        agent_type: str | None = None,
        retrieval_scope: str | None = None,
        force_type: bool = False,
        verbose: bool | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
        resume_plan: dict | None = None,
    ) -> LoopResult:
        """执行 Agent 查询（委托给 LangChain create_agent）。

        agent_type 和 force_type 保留用于向后兼容，已不再区分执行类型——
        所有路径统一使用 create_agent，由 LLM 自主完成 Thought→Action→Observation 循环。

        LangSmith: 此方法作为顶层 Trace（run_type="chain", name="agent_run"），
        create_agent 内部的 LLM/Tool 调用自动作为子 Span 嵌套其下。
        运行时元数据（intent_type、scope、question）通过 _set_langsmith_tags() 注入。
        """
        if verbose is None:
            verbose = self._verbose

        # ---- 初始化调试日志 ----
        debug_logger = None
        if DEBUG_LOG_ENABLED:
            from agent.debug_logger import DebugLogger
            _resume_log = resume_plan.get("debug_log_path") if resume_plan else None
            if _resume_log:
                debug_logger = DebugLogger(question, log_dir=DEBUG_LOG_DIR,
                                          resume_path=_resume_log)
            else:
                debug_logger = DebugLogger(question, log_dir=DEBUG_LOG_DIR)
                DebugLogger.cleanup_old_logs(DEBUG_LOG_DIR, DEBUG_LOG_MAX_FILES)

        # ---- 设置工具上下文 ----
        from agent.tools import set_agent_memory, set_agent_scope
        if memory is not None:
            set_agent_memory(memory)

        # ---- 意图路由 + Scope 解析 ----
        if resume_plan:
            # 澄清恢复：使用存储的 scope/intent，跳过意图检测
            final_scope = resume_plan.get("retrieval_scope", retrieval_scope or "local")
            intent_type = resume_plan.get("intent_type", "knowledge_retrieval")
            rewritten_question = resume_plan.get("rewritten_question", question)
            active_skill = resume_plan.get("active_skill", "")
            _set_langsmith_tags(question, final_scope, intent_type, "resume")
            if verbose:
                print(f"[AGENT] 从澄清恢复 (scope={final_scope}, intent={intent_type})")
            if progress_callback:
                progress_callback("planning", f"从澄清恢复: scope={final_scope}")
            if debug_logger:
                debug_logger.log("clarification_resumed", phase="planning", data={
                    "resumed_scope": final_scope,
                    "resumed_intent": intent_type,
                })
        else:
            if progress_callback:
                progress_callback("planning", "正在分析问题意图...")
            if verbose:
                print("[AGENT] 意图分析")

            from agent.intent_router import route_intent
            intent = route_intent(question, memory)

            intent_type = intent.intent_type
            active_skill = intent.active_skill

            # Scope 解析：外部显式指定 > 意图判定 > 默认值
            if retrieval_scope is not None:
                final_scope = retrieval_scope
            else:
                final_scope = intent.retrieval_scope

            _set_langsmith_tags(question, final_scope, intent_type, "new")

            if debug_logger:
                debug_logger.log("intent_detected", phase="planning", data={
                    "intent_type": intent_type,
                    "confidence": intent.confidence,
                    "retrieval_scope": final_scope,
                })

            if progress_callback:
                scope_emoji = {"local": "📂", "online": "🌐", "hybrid": "🔀"}.get(
                    final_scope, "📂")
                progress_callback("planning",
                    f"意图: {intent_type} | {scope_emoji} {final_scope}")

            # ---- 快速退出：非检索/操作意图 ----
            if intent_type == "out_of_domain":
                if debug_logger:
                    debug_logger.log("loop_result", phase="responding",
                                    data={"status": "out_of_domain"})
                    debug_logger.close()
                return LoopResult(
                    status="out_of_domain",
                    answer=(
                        "抱歉，我的知识库仅限于已索引的科研论文文献，"
                        "无法回答与学术研究无关的问题。\n\n"
                        "**您可以尝试：**\n"
                        "1. 上传相关的 PDF 论文文献\n"
                        "2. 询问与已索引文献相关的学术问题（如方法、实验、结论等）\n"
                        "3. 让我帮您对比或分析已上传的论文"
                    ),
                    debug_log_path=debug_logger.get_log_path() if debug_logger else "",
                )

            if intent_type == "general_chat":
                if debug_logger:
                    debug_logger.log("loop_result", phase="responding",
                                    data={"status": "general_chat"})
                    debug_logger.close()
                return LoopResult(
                    status="general_chat",
                    answer=(
                        "你好！我是科研文献助手，可以帮你：\n\n"
                        "- 📚 **检索论文内容**：询问已上传论文的方法、实验、结论等\n"
                        "- 🌐 **在线搜索论文**：从 arXiv 等平台搜索最新文献\n"
                        "- 🔬 **对比分析**：多篇论文的方法和性能差异\n"
                        "- 📁 **整理论文**：将 PDF 按主题归类到不同文件夹\n\n"
                        "上传 PDF 或直接提出学术问题即可！"
                    ),
                    debug_log_path=debug_logger.get_log_path() if debug_logger else "",
                )

            # ---- 澄清判断：scope 模糊且无外部指定 ----
            if intent.needs_clarification and intent.retrieval_scope == "ambiguous":
                if retrieval_scope is not None and retrieval_scope in (
                    "local", "online", "hybrid"
                ):
                    final_scope = retrieval_scope
                else:
                    if progress_callback:
                        progress_callback("waiting", "需要用户澄清...")
                    if debug_logger:
                        debug_logger.log("clarification_sent", phase="waiting", data={
                            "status": "clarify_scope",
                            "clarification_hint": intent.clarification_hint[:200],
                        })
                        debug_logger.log("loop_result", phase="waiting",
                                        data={"status": "clarify_scope"})
                        debug_logger.close()
                    _pending = {
                        "original_question": question,
                        "rewritten_question": question,
                        "intent_type": intent_type,
                        "retrieval_scope": intent.retrieval_scope,
                        "active_skill": active_skill,
                        "debug_log_path": debug_logger.get_log_path() if debug_logger else "",
                    }
                    return LoopResult(
                        status="clarify_scope",
                        clarification_message=intent.clarification_hint,
                        debug_log_path=debug_logger.get_log_path() if debug_logger else "",
                        pending_plan=_pending,
                    )

            # ---- 指代消解（有对话历史时）----
            rewritten_question = question
            has_history = memory is not None and memory.message_count() > 0
            if has_history:
                try:
                    from agent.query_rewriter import get_query_rewriter
                    from agent.generator import create_llm as _create_gen_llm
                    rewriter = get_query_rewriter()
                    llm_rewrite = _create_gen_llm()
                    rewrite_result = rewriter.rewrite(question, memory, llm_rewrite)
                    if rewrite_result.needs_rewrite:
                        rewritten_question = rewrite_result.rewritten
                        if verbose:
                            print(
                                f'[AGENT] 指代消解: "{question[:40]}..."'
                                f' → "{rewrite_result.rewritten[:60]}..."'
                            )
                        if progress_callback:
                            progress_callback("planning",
                                f"指代消解: {rewrite_result.rewritten[:60]}...")
                        if debug_logger:
                            debug_logger.log("query_rewritten", phase="planning", data={
                                "original": question[:100],
                                "rewritten": rewrite_result.rewritten[:200],
                            })
                except Exception as e:
                    if verbose:
                        print(f"[AGENT] 指代消解跳过: {e}")

        # ---- 应用检索范围到工具上下文 ----
        set_agent_scope(final_scope)

        # ---- 构建 System Prompt ----
        system_prompt = IMMUTABLE_SYSTEM_PROMPT
        dynamic_ctx = self._get_dynamic_context(final_scope, intent_type,
                                                active_skill=active_skill)
        full_system_prompt = system_prompt + "\n\n" + dynamic_ctx

        if verbose:
            print(f"[AGENT] System prompt 已构建"
                  f" (scope={final_scope}, intent={intent_type})")

        # ---- 获取过滤后的工具列表 ----
        from agent.tools import get_tools_for_intent_and_scope
        tools = get_tools_for_intent_and_scope(intent_type, final_scope,
                                                active_skill=active_skill)

        if debug_logger:
            debug_logger.log("agent_start", phase="executing", data={
                "intent_type": intent_type,
                "scope": final_scope,
                "tools_count": len(tools),
                "tool_names": [t.name for t in tools],
            })

        # ---- 创建 Agent 并执行 ----
        llm = _create_agent_llm()

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=full_system_prompt,
        )

        # 构建消息序列
        messages = []
        if memory:
            messages.extend(self._build_chat_history(memory))
        messages.append(HumanMessage(content=rewritten_question))

        if progress_callback:
            progress_callback("executing", "Agent 正在分析和执行...")

        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": AGENT_MAX_ITERATIONS + 5},
            )
        except Exception as e:
            error_msg = str(e)
            if debug_logger:
                debug_logger.log_error("agent_execution_error", error_msg,
                                      phase="executing", exc_info=True)
            if "recursion" in error_msg.lower():
                answer = (
                    "抱歉，任务执行步骤过多，已自动停止。\n\n"
                    "**建议**：请尝试简化问题后重新提问。"
                )
            elif "rate" in error_msg.lower():
                answer = "抱歉，API 请求频率过高，请稍后再试。"
            else:
                answer = (
                    "抱歉，执行过程中出现错误。\n\n"
                    f"**错误信息**：{error_msg[:200]}\n\n"
                    "**建议**：请重试或简化问题。"
                )
            if debug_logger:
                debug_logger.log("loop_result", phase="responding", data={
                    "status": "answer", "answer_length": len(answer),
                })
                debug_logger.close()
            return LoopResult(
                status="answer",
                answer=answer,
                rewritten_query=rewritten_question,
                debug_log_path=debug_logger.get_log_path() if debug_logger else "",
            )

        # ---- 提取最终答案（结构化解析）----
        output_messages = result.get("messages", [])
        raw_answer = ""
        # 优先取最后一条不带 tool_calls 的 AIMessage（最终回答）
        for msg in reversed(output_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                raw_answer = msg.content
                break
        # 回退：取最后一条 AIMessage
        if not raw_answer:
            for msg in reversed(output_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    raw_answer = msg.content
                    break
        # 最终回退
        if not raw_answer:
            raw_answer = (
                "抱歉，未能生成有效的回答。请重新描述您的问题。\n\n"
                "**建议**：尝试更具体地描述您需要检索或操作的内容。"
            )

        # 结构化解析 LLM 输出
        structured = _parse_structured_answer(raw_answer)
        answer = structured.get("answer", raw_answer)
        sources = structured.get("sources", [])
        confidence = structured.get("confidence", "medium")
        retrieval_status = structured.get("retrieval_status", "unknown")

        # ---- 记录工具调用到调试日志 ----
        if debug_logger:
            for msg in output_messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        debug_logger.log("tool_invoked", phase="executing", data={
                            "tool": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                        })
            debug_logger.log("loop_result", phase="responding", data={
                "status": "answer",
                "answer_length": len(answer),
                "structured": {
                    "confidence": confidence,
                    "retrieval_status": retrieval_status,
                    "sources_count": len(sources),
                },
            })
            debug_logger.close()

        if progress_callback:
            progress_callback("responding", "生成最终回复")

        if verbose:
            print(f"[AGENT] 完成 (answer={len(answer)} chars, "
                  f"confidence={confidence}, sources={len(sources)})")

        return LoopResult(
            status="answer",
            answer=answer,
            rewritten_query=rewritten_question,
            debug_log_path=debug_logger.get_log_path() if debug_logger else "",
            # 附加结构化元数据到检索文档列表
            retrieved_docs=[{"sources": sources, "confidence": confidence,
                           "retrieval_status": retrieval_status}],
        )


    # ================================================================
    #  静态辅助方法
    # ================================================================

    @staticmethod
    def _build_chat_history(memory: BaseMemory) -> list:
        """将 BaseMemory 中的消息转换为 LangChain 消息格式。"""
        messages = []
        for msg in memory.get_messages()[-20:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        return messages

    @staticmethod
    def _trim_messages_by_tokens(messages: list, max_tokens: int = 6000) -> list:
        """按 token 预算修剪消息列表。

        策略：SystemMessage + 首条 HumanMessage 始终保留，从最新消息向前累积。
        """
        CHARS_PER_TOKEN = 2.5

        def _estimate_tokens(msg) -> int:
            content = getattr(msg, "content", "") or ""
            extra = 0
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    extra += len(str(tc.get("args", ""))) // 3
            return max(1, int(len(content) / CHARS_PER_TOKEN) + extra)

        max_chars_budget = int(max_tokens * CHARS_PER_TOKEN)
        if not messages:
            return messages

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        system_chars = sum(len(getattr(m, "content", "") or "") for m in system_msgs)

        first_human_idx = None
        for i, m in enumerate(other_msgs):
            if isinstance(m, HumanMessage):
                first_human_idx = i
                break

        first_human = []
        remaining_budget = max_chars_budget - system_chars
        if first_human_idx is not None:
            first_human = [other_msgs[first_human_idx]]
            remaining_budget -= len(getattr(other_msgs[first_human_idx], "content", "") or "")

        tail_msgs = other_msgs[(first_human_idx + 1):] if first_human_idx is not None else other_msgs
        kept = []
        char_count = 0
        min_keep = 5

        for m in reversed(tail_msgs):
            msg_chars = len(getattr(m, "content", "") or "")
            if char_count + msg_chars <= remaining_budget or len(kept) < min_keep:
                kept.insert(0, m)
                char_count += msg_chars
            else:
                break

        return system_msgs + first_human + kept


# ================================================================
#  全局单例
# ================================================================

_unified_loop: UnifiedAgentLoop | None = None


def _get_unified_loop() -> UnifiedAgentLoop:
    """获取全局 UnifiedAgentLoop 单例。"""
    global _unified_loop
    if _unified_loop is None:
        _unified_loop = UnifiedAgentLoop()
    return _unified_loop


# ================================================================
#  公开 API（向后兼容）
# ================================================================

def react_agent_query(
    question: str,
    memory: BaseMemory | None = None,
    retrieval_scope: str | None = None,
    verbose: bool | None = None,
    resume_plan: dict | None = None,
) -> LoopResult:
    """ReAct 模式查询（委托给 agent_query）。"""
    return agent_query(
        question=question,
        memory=memory,
        retrieval_scope=retrieval_scope,
        verbose=verbose,
        resume_plan=resume_plan,
    )


def plan_execute_query(
    question: str,
    memory: BaseMemory | None = None,
    retrieval_scope: str | None = None,
    verbose: bool | None = None,
    resume_plan: dict | None = None,
) -> LoopResult:
    """Plan-Execute 模式查询（委托给 agent_query）。"""
    return agent_query(
        question=question,
        memory=memory,
        retrieval_scope=retrieval_scope,
        verbose=verbose,
        resume_plan=resume_plan,
    )


def agent_query(
    question: str,
    memory: BaseMemory | None = None,
    agent_type: str | None = None,
    retrieval_scope: str | None = None,
    force_type: bool = False,
    verbose: bool | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    resume_plan: dict | None = None,
) -> LoopResult:
    """
    Agent 模式查询入口（调度器）。

    委托给 UnifiedAgentLoop，自动选择 Agent 类型。

    返回:
        LoopResult: 结构化结果
          .status in ("answer", "clarify_scope", "out_of_domain", "general_chat")
          .answer — 最终回复文本
          .clarification_message — 需要澄清时的提示
          .pending_plan — 澄清挂起时保存的已确认计划，供 resume 使用
    """
    if verbose is None:
        verbose = AGENT_VERBOSE

    loop = _get_unified_loop()
    return loop.run(
        question=question,
        memory=memory,
        agent_type=agent_type,
        retrieval_scope=retrieval_scope,
        force_type=force_type,
        verbose=verbose,
        progress_callback=progress_callback,
        resume_plan=resume_plan,
    )
