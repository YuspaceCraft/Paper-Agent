"""
unified_stm.py — 统一短期记忆 (Unified Short-Term Memory)
==============
集中式短期记忆系统，是 agent 单次对话内 **唯一** 的状态存储。

设计原则:
  1. 单一写入路径 — 所有状态变更都通过 UnifiedSTM 的方法
  2. 单一读取路径 — LLM 上下文始终从 build_context_header() 构建
  3. 固定头部注入 — 任务栈 + 观察缓冲区 + 约束卡 + 反思日志，顺序固定
  4. 不与 ctx.messages 双写 — ctx.messages 退化为 UnifiedSTM 的序列化输出

四大组件:
  TaskStack         — 「我现在应该做什么」目标分解 + 步骤状态机
  ObservationBuffer — 「最近发生了什么」N 步工具结果 + 自动摘要
  Constraints       — 「我必须遵守什么」核心规则压缩卡
  ReflectionLog     — 「之前的坑在哪里」失败轨迹 + 可检索的经验教训

生命周期: 单次 agent_query() → run() 调用内。每次新问题创建新实例。

使用方式:
  from agent.unified_stm import UnifiedSTM, TaskStack, ObservationBuffer, Constraints, ReflectionLog

  stm = UnifiedSTM(question="对比 DenseNet 和 ResNet", scope="local", intent="knowledge_retrieval")
  stm.task_stack.set_plan(["检索 DenseNet 论文", "检索 ResNet 论文", "对比分析"])
  stm.observe("search_literature", {"query": "DenseNet"}, "找到 3 篇", success=True)
  header = stm.build_context_header()  # 每次 LLM 推理前调用
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

MAX_OBSERVATIONS = 8        # 观察缓冲区容量
MAX_OBSERVATION_CHARS = 250 # 单条观察最大字符
OBSERVATION_COMPACT_AT = 6  # 超过此数量时压缩旧观察
MAX_REFLECTION_LOG = 12     # 反思日志最大条数
MAX_CONTEXT_HEADER_CHARS = 2500  # 上下文头部预算


# ═══════════════════════════════════════════════════════════════
# 约束卡模板
# ═══════════════════════════════════════════════════════════════

_CONSTRAINTS_TEMPLATE = """\
1. **先检索再回答**: 必须调用工具获取信息后才能回答，禁止凭空编造
2. **诚实面对失败**: 检索无结果 → 如实告知用户，绝不虚构内容
3. **标注来源**: 所有事实陈述必须标注出处（文件名/作者/年份）
4. **当前范围**: {scope_emoji} {scope_name}
5. **当前意图**: {intent_name}
6. **工具失败不重试**: 同一工具+同一参数失败后，换工具或换策略，不无脑重试"""

_SCOPE_INFO = {
    "local":  ("📂", "本地知识库 — 使用 search_literature 检索已索引论文"),
    "online": ("🌐", "arXiv 在线检索 — 使用 search_papers 检索最新论文"),
    "hybrid": ("🔀", "本地 + arXiv 混合 — 先用 search_literature 再 supplement 在线"),
}

_INTENT_INFO = {
    "knowledge_retrieval": "学术文献检索与分析",
    "file_management":     "论文文件整理与分类",
    "general_chat":        "闲聊/能力说明",
    "out_of_domain":       "超出领域的问题",
}


# ═══════════════════════════════════════════════════════════════
# TaskNode — 任务栈节点
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskNode:
    """任务栈中的一个步骤节点。"""
    step_id: int
    description: str
    status: str = "pending"       # pending | active | completed | failed | skipped
    tool_used: str = ""
    result_summary: str = ""
    raw_result: str = ""          # 保留检索原文供反思验证（最多 3000 chars）
    success: bool = False
    retry_count: int = 0
    max_retries: int = 3

    @property
    def icon(self) -> str:
        return {
            "pending": "⬜", "active": "🔄", "completed": "✅",
            "failed": "❌", "skipped": "⏭️",
        }.get(self.status, "⬜")

    def mark_active(self):
        self.status = "active"

    def mark_completed(self, tool: str = "", summary: str = "",
                       raw: str = "", success: bool = True):
        self.status = "completed" if success else "failed"
        self.tool_used = tool
        self.result_summary = summary[:MAX_OBSERVATION_CHARS]
        self.raw_result = raw[:3000]
        self.success = success

    def mark_skipped(self, reason: str = ""):
        self.status = "skipped"
        self.result_summary = reason

    def increment_retry(self) -> bool:
        """返回 True 如果还可以重试。"""
        self.retry_count += 1
        return self.retry_count <= self.max_retries


# ═══════════════════════════════════════════════════════════════
# TaskStack — 目标分解 + 步骤状态机
# ═══════════════════════════════════════════════════════════════

@dataclass
class TaskStack:
    """
    任务栈：维护当前目标、子目标分解及完成状态。

    每次推理前，agent 首先确认此栈，回答「我现在应该做什么」而非「我上一句说了什么」。
    """
    question: str = ""
    intent_type: str = "knowledge_retrieval"
    steps: list[TaskNode] = field(default_factory=list)

    # -- 写入 --

    def set_goal(self, question: str, intent_type: str = "knowledge_retrieval"):
        """设定总目标。"""
        self.question = question
        self.intent_type = intent_type

    def set_plan(self, step_descriptions: list[str]):
        """从步骤描述列表批量创建 TaskNode。"""
        self.steps = [
            TaskNode(step_id=i, description=desc)
            for i, desc in enumerate(step_descriptions, 1)
        ]

    def set_plan_nodes(self, nodes: list[TaskNode]):
        """直接设置步骤节点列表（用于从 ConfirmedPlan.sub_steps 恢复）。"""
        self.steps = nodes

    def activate_step(self, step_id: int):
        """标记某步骤为执行中，同时取消其他 active 步骤。"""
        for s in self.steps:
            if s.status == "active":
                s.status = "pending"
        for s in self.steps:
            if s.step_id == step_id:
                s.mark_active()
                return

    def complete_step(self, step_id: int, tool: str = "",
                      summary: str = "", raw: str = "", success: bool = True):
        """标记步骤完成。"""
        for s in self.steps:
            if s.step_id == step_id:
                s.mark_completed(tool, summary, raw, success)
                return

    def skip_step(self, step_id: int, reason: str = ""):
        """跳过步骤。"""
        for s in self.steps:
            if s.step_id == step_id:
                s.mark_skipped(reason)
                return

    def can_retry(self, step_id: int) -> bool:
        """检查步骤是否可以重试。"""
        for s in self.steps:
            if s.step_id == step_id:
                return s.increment_retry()
        return False

    def append_step(self, description: str) -> int:
        """追加补充步骤，返回新步骤 ID。"""
        new_id = len(self.steps) + 1
        self.steps.append(TaskNode(step_id=new_id, description=description))
        return new_id

    # -- 查询 --

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    @property
    def total_count(self) -> int:
        return len(self.steps)

    @property
    def current_step(self) -> TaskNode | None:
        """当前正在执行的步骤。"""
        for s in self.steps:
            if s.status == "active":
                return s
        return None

    @property
    def next_pending(self) -> TaskNode | None:
        """下一个待执行的步骤。"""
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    @property
    def progress_summary(self) -> str:
        """简短进度摘要。"""
        return f"{self.completed_count}/{self.total_count} 步骤完成" + (
            f", {self.failed_count} 失败" if self.failed_count > 0 else ""
        )

    def get_progress_line(self) -> str:
        """一行进度摘要（~40-80 chars），供紧凑观测使用。

        格式: "[2/5] ✅ search → next: compare papers"
        不含详细步骤描述，只给 LLM 一个即时方向感。
        """
        done = self.completed_count
        failed = self.failed_count
        total = self.total_count
        base = f"[{done}/{total}]"
        if failed:
            base += f" ({failed} failed)"

        # 查找下一个待执行步骤
        current = self.current_step
        if current:
            base += f" → 🔄 {current.description[:60]}"
        else:
            next_step = self.next_pending
            if next_step:
                base += f" → next: {next_step.description[:60]}"
            elif done >= total and total > 0:
                base += " → 全部完成"
        return base

    def has_any_success(self) -> bool:
        return any(s.status == "completed" and s.success for s in self.steps)

    def has_any_execution(self) -> bool:
        return any(s.status in ("completed", "failed") for s in self.steps)

    def get_completed_steps(self) -> list[TaskNode]:
        return [s for s in self.steps if s.status == "completed"]

    def get_execution_records(self) -> list[dict]:
        """获取执行记录视图（供 TaskProgress 兼容）。"""
        return [
            {
                "tool": s.tool_used,
                "args_summary": s.description[:80],
                "result_summary": s.result_summary[:200],
                "success": s.success,
                "timestamp": "",
            }
            for s in self.steps if s.status in ("completed", "failed")
        ]

    # -- 渲染 --

    def render(self) -> str:
        """渲染任务栈为结构化文本。"""
        lines = [f"🎯 总目标: {self.question[:120]}"]
        if not self.steps:
            lines.append("（无预设步骤 — ReAct 自由探索模式）")
            return "\n".join(lines)

        lines.append(f"📋 进度: {self.progress_summary}")
        for s in self.steps:
            detail = s.result_summary[:80] if s.result_summary else s.description[:80]
            lines.append(f"  {s.icon} 步骤{s.step_id}: {detail}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "question": self.question[:100],
            "intent": self.intent_type,
            "progress": self.progress_summary,
            "steps": [
                {"id": s.step_id, "desc": s.description[:60],
                 "status": s.status, "tool": s.tool_used}
                for s in self.steps
            ],
        }


# ═══════════════════════════════════════════════════════════════
# Observation — 单条观察
# ═══════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """单条工具执行观察。"""
    tool: str
    args_summary: str
    result_summary: str
    success: bool
    iteration: int = 1
    timestamp: str = ""
    raw_result: str = ""   # 保留原始结果供反思验证
    step_id: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")

    def one_line(self) -> str:
        icon = "✅" if self.success else "❌"
        return f"[{self.timestamp}] {icon} {self.tool}: {self.result_summary[:120]}"

    def compact_line(self) -> str:
        """超压缩一行（用于旧观察折叠）。"""
        icon = "✅" if self.success else "❌"
        return f"{icon} {self.tool} → {self.result_summary[:80]}"


# ═══════════════════════════════════════════════════════════════
# ObservationBuffer — 观察缓冲区
# ═══════════════════════════════════════════════════════════════

@dataclass
class ObservationBuffer:
    """
    观察缓冲区：最近 N 步的工具返回结果和环境反馈。

    - 新观察追加到队尾
    - 超过 COMPACT_AT 时，最旧的观察被压缩为摘要行
    - 检索类工具的 raw_result 始终保留（供反思验证）
    - 非检索工具只保留摘要
    """

    observations: deque = field(default_factory=lambda: deque(maxlen=MAX_OBSERVATIONS))
    compacted: list[str] = field(default_factory=list)  # 已被压缩的旧观察摘要
    _retrieval_tools: set = field(default_factory=lambda: {
        "search_literature", "search_papers", "get_paper_detail",
        "get_paper_data", "compare_papers",
    })

    # -- 写入 --

    def add(self, tool: str, args: dict | None, result_summary: str,
            success: bool = True, iteration: int = 1, raw_result: str = "",
            step_id: int = 0):
        """添加一条观察。如果缓冲区满，压缩最旧的观察。"""
        args_summary = ""
        if args:
            items = []
            for k, v in args.items():
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:47] + "..."
                items.append(f"{k}={v_str}")
            args_summary = ", ".join(items[:4])

        # 检索工具保留 raw_result，非检索工具不保留（节省内存）
        keep_raw = raw_result[:3000] if (tool in self._retrieval_tools and raw_result) else ""

        obs = Observation(
            tool=tool,
            args_summary=args_summary,
            result_summary=result_summary[:MAX_OBSERVATION_CHARS],
            success=success,
            iteration=iteration,
            raw_result=keep_raw,
            step_id=step_id,
        )

        # 如果缓冲区已满，压缩最旧的一条
        if len(self.observations) >= self.observations.maxlen:
            old = self.observations.popleft()
            self.compacted.append(old.compact_line())
            # 限制压缩历史长度
            if len(self.compacted) > 20:
                self.compacted = self.compacted[-15:]

        self.observations.append(obs)

    # -- 查询 --

    def last(self) -> Observation | None:
        """最近一条观察。"""
        return self.observations[-1] if self.observations else None

    def last_n(self, n: int = 3) -> list[Observation]:
        """最近 N 条观察。"""
        items = list(self.observations)
        return items[-n:]

    def has_any_success(self) -> bool:
        return any(o.success for o in self.observations)

    def success_count(self) -> int:
        return sum(1 for o in self.observations if o.success)

    def failure_count(self) -> int:
        return sum(1 for o in self.observations if not o.success)

    def get_raw_results(self) -> str:
        """获取最近检索工具返回的原始内容（供反思验证）。"""
        retrieval_obs = [
            o for o in self.observations
            if o.tool in self._retrieval_tools and o.raw_result
        ]
        if not retrieval_obs:
            return ""

        # 只取最近一轮迭代的原文
        latest_iter = max(o.iteration for o in retrieval_obs)
        latest = [o for o in retrieval_obs if o.iteration == latest_iter]

        lines = ["[工具原始输出 — 用于验证答案忠实度]"]
        for o in latest:
            lines.append(f"\n--- [{o.tool}] ---\n{o.raw_result[:3000]}")
        return "\n".join(lines)

    # -- 渲染 --

    def render(self, max_items: int = 8) -> str:
        """渲染观察缓冲区为结构化文本。"""
        lines = []

        if self.compacted:
            lines.append(f"[更早的记录] {' | '.join(self.compacted[-5:])}")
            lines.append("")

        all_obs = list(self.observations)
        show = all_obs[-max_items:] if len(all_obs) > max_items else all_obs

        if len(all_obs) > max_items:
            lines.append(f"... 省略 {len(all_obs) - max_items} 条 ...")

        for i, o in enumerate(show):
            idx = len(all_obs) - len(show) + i + 1
            lines.append(f"[{idx}] {o.one_line()}")

        if not lines:
            return "（尚无工具执行记录）"

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.observations) + len(self.compacted)


# ═══════════════════════════════════════════════════════════════
# ReflectionEntry — 反思条目
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReflectionEntry:
    """一条反思/纠错记录。"""
    iteration: int
    issue: str            # 具体问题描述
    lesson: str           # 抽象化的经验教训（可跨场景复用）
    suggestions: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")

    def one_line(self) -> str:
        return f"⚠️ 第{self.iteration}轮: {self.issue[:100]} → 教训: {self.lesson[:100]}"


# ═══════════════════════════════════════════════════════════════
# ReflectionLog — 反思/纠错日志
# ═══════════════════════════════════════════════════════════════

@dataclass
class ReflectionLog:
    """
    反思日志：记录失败尝试和自我修正结论。

    - 新条目追加时，自动去重（与已有条目相似度过高则跳过）
    - 渲染时优先显示最近的和最频繁出现的教训类型
    - 支持简单的关键词检索（当遇到类似困境时优先检索）
    """

    entries: list[ReflectionEntry] = field(default_factory=list)

    # -- 写入 --

    def add(self, iteration: int, issue: str, lesson: str = "",
            suggestions: list[str] | None = None,
            scores: dict[str, float] | None = None):
        """添加反思条目。自动去重相似条目。"""
        if not lesson:
            lesson = self._derive_lesson(issue)

        entry = ReflectionEntry(
            iteration=iteration,
            issue=issue[:200],
            lesson=lesson[:200],
            suggestions=suggestions or [],
            scores=scores or {},
        )

        # 去重：如果与最近条目高度相似，跳过
        if self._is_duplicate(entry):
            return

        self.entries.append(entry)

        # 限制容量
        if len(self.entries) > MAX_REFLECTION_LOG:
            self.entries = self.entries[-MAX_REFLECTION_LOG:]

    def add_from_verdict(self, iteration: int, issues: list[str],
                         suggestions: list[str], scores: dict[str, float]):
        """从反思 verdict 批量添加条目。"""
        for issue in issues:
            self.add(
                iteration=iteration,
                issue=issue,
                suggestions=suggestions,
                scores=scores,
            )

    # -- 查询 --

    def search(self, query: str, top_k: int = 3) -> list[ReflectionEntry]:
        """
        简单关键词检索：查找与当前困境相似的历史反思。

        使用 token 重叠 + lesson 关键词匹配。
        """
        query_tokens = set(query.lower().split())
        if not query_tokens or not self.entries:
            return []

        scored = []
        for entry in self.entries:
            entry_text = (entry.issue + " " + entry.lesson).lower()
            entry_tokens = set(entry_text.split())
            overlap = len(query_tokens & entry_tokens)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def recent_lessons(self, n: int = 5) -> list[str]:
        """最近的经验教训摘要。"""
        return [e.lesson[:120] for e in self.entries[-n:]]

    # -- 渲染 --

    def render(self, max_entries: int = 5) -> str:
        """渲染反思日志为结构化文本。"""
        if not self.entries:
            return "（尚无反思记录）"

        recent = self.entries[-max_entries:]

        # 按迭代分组，最近的在前
        lines = []
        for e in reversed(recent):
            lines.append(e.one_line())

        if len(self.entries) > max_entries:
            lines.append(f"... 及其他 {len(self.entries) - max_entries} 条历史反思")

        return "\n".join(lines)

    # -- 内部 --

    def _is_duplicate(self, new_entry: ReflectionEntry) -> bool:
        """检查是否与已有条目重复（基于 issue 文本重叠比例）。"""
        if not self.entries:
            return False
        # 只检查最近 3 条
        for recent in self.entries[-3:]:
            overlap = self._char_overlap(new_entry.issue, recent.issue)
            if overlap > 0.6:
                return True
        return False

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        """两个字符串的字符级重叠比例。"""
        a_set = set(a.lower())
        b_set = set(b.lower())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / min(len(a_set), len(b_set))

    @staticmethod
    def _derive_lesson(issue: str) -> str:
        """从具体问题描述中生成简要教训。"""
        # 基于关键词的启发式映射（兜底，LLM 反思时会生成更精确的 lesson）
        mappings = {
            "检索": "执行检索前需确认知识库状态",
            "编造": "只能基于工具返回的实际结果回答",
            "文件": "操作前必须先 list_directory 确认文件存在",
            "空": "工具返回空结果时应诚实告知而非编造",
            "失败": "分析失败原因，换策略而非重复相同调用",
            "未执行": "必须在回答前完成计划中的工具调用",
            "矛盾": "答案中的事实必须与执行记录一致",
        }
        for keyword, lesson in mappings.items():
            if keyword in issue:
                return lesson
        return f"注意: {issue[:80]}"


# ═══════════════════════════════════════════════════════════════
# Constraints — 全局约束卡
# ═══════════════════════════════════════════════════════════════

@dataclass
class Constraints:
    """
    全局约束/规则区：System Prompt 核心限制的压缩版。

    每次推理前强制注入上下文头部，对抗长对话中的「指令遗忘」。
    """
    scope: str = "local"
    intent_type: str = "knowledge_retrieval"
    extra_rules: list[str] = field(default_factory=list)

    def update(self, scope: str = "", intent_type: str = ""):
        if scope:
            self.scope = scope
        if intent_type:
            self.intent_type = intent_type

    def add_rule(self, rule: str):
        """追加临时约束（如「当前必须调用 search_literature」）。"""
        self.extra_rules.append(rule)
        if len(self.extra_rules) > 5:
            self.extra_rules = self.extra_rules[-5:]

    def clear_extra_rules(self):
        self.extra_rules.clear()

    def render(self) -> str:
        scope_emoji, scope_name = _SCOPE_INFO.get(
            self.scope, _SCOPE_INFO["local"])
        intent_name = _INTENT_INFO.get(
            self.intent_type, "学术文献检索与分析")

        base = _CONSTRAINTS_TEMPLATE.format(
            scope_emoji=scope_emoji,
            scope_name=scope_name,
            intent_name=intent_name,
        )

        if self.extra_rules:
            extra = "\n".join(f"  ⚠️ {r}" for r in self.extra_rules)
            base += f"\n**临时约束:**\n{extra}"

        return base


# ═══════════════════════════════════════════════════════════════
# UnifiedSTM — 统一短期记忆（顶层编排）
# ═══════════════════════════════════════════════════════════════

@dataclass
class UnifiedSTM:
    """
    统一短期记忆 — agent 单次对话内的**唯一**状态存储。

    替代旧架构中的 WorkingMemory + AgentLoopContext + TaskProgress 三件套。
    所有状态变更通过 UnifiedSTM 方法写入，所有 LLM 上下文从 build_context_header() 读取。
    """

    # 四大组件
    task_stack: TaskStack = field(default_factory=TaskStack)
    observations: ObservationBuffer = field(default_factory=ObservationBuffer)
    constraints: Constraints = field(default_factory=Constraints)
    reflection_log: ReflectionLog = field(default_factory=ReflectionLog)

    # 附加状态
    current_iteration: int = 1
    supplement_count: int = 0
    last_answer: str = ""
    conversation_history: str = ""
    errors: list[dict] = field(default_factory=list)

    @classmethod
    def create(cls, question: str, scope: str = "local",
               intent_type: str = "knowledge_retrieval") -> UnifiedSTM:
        """工厂方法：创建并初始化 UnifiedSTM。"""
        stm = cls()
        stm.task_stack.set_goal(question, intent_type)
        stm.constraints.update(scope=scope, intent_type=intent_type)
        return stm

    # ═══════════════════════════════════════════════════
    # 便捷写入（统一入口）
    # ═══════════════════════════════════════════════════

    def set_plan(self, steps: list[str], intent_type: str = ""):
        """写入任务计划。"""
        self.task_stack.set_plan(steps)
        if intent_type:
            self.task_stack.intent_type = intent_type
            self.constraints.update(intent_type=intent_type)

    def set_scope(self, scope: str):
        self.constraints.update(scope=scope)

    def set_intent(self, intent_type: str):
        self.task_stack.intent_type = intent_type
        self.constraints.update(intent_type=intent_type)

    def set_iteration(self, iteration: int):
        self.current_iteration = iteration

    def set_history(self, text: str):
        """写入对话历史（来自 BaseMemory）。"""
        if text and len(text) > 2000:
            text = "...(更早的对话已省略)\n" + text[-1950:]
        self.conversation_history = text or ""

    def observe(self, tool: str, args: dict | None, result_summary: str,
                success: bool = True, raw_result: str = "", step_id: int = 0):
        """记录一次工具执行观察。同时更新任务栈对应步骤。"""
        self.observations.add(
            tool=tool, args=args, result_summary=result_summary,
            success=success, iteration=self.current_iteration,
            raw_result=raw_result, step_id=step_id,
        )
        # 同步更新任务栈
        if step_id and step_id <= self.task_stack.total_count:
            self.task_stack.complete_step(
                step_id, tool, result_summary, raw_result, success)

    def update_step(self, step_id: int, tool: str, result_summary: str,
                    success: bool = True, raw_result: str = ""):
        """更新步骤状态（Plan-Execute 路径便捷方法）。"""
        self.observe(tool=tool, args=None, result_summary=result_summary,
                     success=success, raw_result=raw_result, step_id=step_id)

    def record_reflection(self, issues: list[str], suggestions: list[str],
                          scores: dict[str, float]):
        """记录反思结果。"""
        self.reflection_log.add_from_verdict(
            iteration=self.current_iteration,
            issues=issues,
            suggestions=suggestions,
            scores=scores,
        )

    def record_answer(self, answer: str):
        """记录本轮最终答案。"""
        self.last_answer = answer[:2000]

    def add_error(self, error_type: str, message: str):
        self.errors.append({
            "type": error_type,
            "message": message[:300],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        if len(self.errors) > 20:
            self.errors = self.errors[-20:]

    def add_constraint(self, rule: str):
        """追加临时约束（如补充检索提示）。"""
        self.constraints.add_rule(rule)

    def clear_constraints(self):
        self.constraints.clear_extra_rules()

    # ═══════════════════════════════════════════════════════
    # 查询
    # ═══════════════════════════════════════════════════════

    def has_any_execution(self) -> bool:
        return self.task_stack.has_any_execution()

    def has_any_success(self) -> bool:
        return self.observations.has_any_success()

    @property
    def scope(self) -> str:
        return self.constraints.scope

    @property
    def intent_type(self) -> str:
        return self.task_stack.intent_type

    # ═══════════════════════════════════════════════════════
    # 核心方法：构建 LLM 上下文头部
    # ═══════════════════════════════════════════════════════

    def build_context_header(self, max_chars: int = MAX_CONTEXT_HEADER_CHARS) -> str:
        """
        构建每次 LLM 推理前必须注入的固定上下文头部。

        顺序固定（不可变）：
          1. 核心约束卡 — 「我必须遵守什么」
          2. 任务栈 — 「我现在应该做什么」
          3. 观察记录 — 「最近发生了什么」
          4. 反思日志 — 「之前的坑在哪里」

        这是 STM 对外唯一的上下文输出接口。
        反思/修正/合成阶段都通过此方法获取统一的上下文视图。
        """
        sections = [
            ("【核心约束 — 每次推理前必读】", self.constraints.render()),
            ("【任务栈 — 当前目标与进度】", self.task_stack.render()),
            ("【观察记录 — 最近执行结果】", self.observations.render()),
        ]

        reflection_text = self.reflection_log.render()
        if reflection_text and reflection_text != "（尚无反思记录）":
            sections.append(("【反思日志 — 经验教训】", reflection_text))

        # 对话历史（如有，简要）
        if self.conversation_history:
            history_short = self.conversation_history[:400]
            sections.append(("[对话历史]", history_short))

        # 组装
        parts = []
        total = 0
        for title, content in sections:
            block = f"{title}\n{content}"
            if total + len(block) > max_chars:
                remaining = max_chars - total - 50
                if remaining > 100:
                    block = f"{title}\n{content[:remaining]}\n...(已截断)"
                else:
                    break
            parts.append(block)
            total += len(block)

        header = "\n\n".join(parts)

        # 用分隔线包裹
        return (
            "══════════════════════════════════════════\n"
            + header
            + "\n══════════════════════════════════════════"
        )

    def build_compact_observation(self, dynamic_context: str = "",
                                  max_tool_results: int = 3) -> str:
        """为执行推理构建精简观测（~80-200 tokens）。

        与 build_context_header() 的区别：
          - build_context_header(): 完整上下文（~2500 chars），反思/修正阶段使用
          - build_compact_observation(): 紧凑观测（~400 chars），执行推理使用

        不包含：完整任务栈渲染、反思历史、对话历史、约束卡
        （约束卡已在 System Prompt 中，反思历史仅在必要时注入一行）。

        包含：当前步骤进度 + 最近 N 条工具结果 + 动态上下文规则。
        """
        parts = []

        # 1. 当前进度（一行，~40-80 chars）
        progress = self.task_stack.get_progress_line()
        parts.append(f"📋 {progress}")

        # 2. 最近 N 条观察（~80-150 chars）
        recent = list(self.observations.observations)[-max_tool_results:]
        if recent:
            for obs in recent:
                status = "✅" if obs.success else "❌"
                parts.append(f"{status} [{obs.tool}] {obs.result_summary[:120]}")
        elif self.observations.compacted:
            # 所有观察已被压缩，显示最近一条压缩记录
            parts.append(f"📝 {self.observations.compacted[-1][:120]}")

        # 3. 最近反思要点（仅当有关键问题时，~40 chars）
        if self.reflection_log.entries:
            last = self.reflection_log.entries[-1]
            if last.issue:
                parts.append(f"⚠️ 上次问题: {last.issue[:100]}")

        # 4. 动态上下文规则（scope + tools + 文件管理/skill，由调用方传入）
        if dynamic_context:
            parts.append(dynamic_context)

        return "\n".join(parts)

    def build_execution_summary(self) -> str:
        """构建执行结果摘要（供合成阶段使用）。"""
        lines = [f"执行完成: {self.observations.success_count()} 成功, "
                 f"{self.observations.failure_count()} 失败"]
        for i, o in enumerate(self.observations.observations, 1):
            lines.append(f"  {o.one_line()}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # 兼容方法（供旧代码平滑迁移）
    # ═══════════════════════════════════════════════════════

    def get_reflection_context(self) -> str:
        """获取反思阶段上下文（兼容旧 API）。"""
        # 反思需要详细上下文 + 检索原文
        header = self.build_context_header()
        raw = self.observations.get_raw_results()
        if raw:
            header += "\n\n" + raw
        return header

    def get_correction_context(self) -> str:
        """获取修正阶段上下文（兼容旧 API）。"""
        ctx = self.get_reflection_context()
        ctx += (
            "\n\n⚠️ 修正规则: 只删除与上述执行记录明确矛盾的内容。"
            "摘要中不可见的细节 ≠ 编造。有疑问时保留。"
        )
        return ctx

    def get_completed_step_count(self) -> int:
        return self.task_stack.completed_count

    def snapshot(self) -> dict:
        """返回可读的状态快照（供调试）。"""
        return {
            "question": self.task_stack.question[:100],
            "intent": self.task_stack.intent_type,
            "scope": self.constraints.scope,
            "iteration": self.current_iteration,
            "plan": self.task_stack.progress_summary,
            "observations": len(self.observations),
            "reflections": len(self.reflection_log.entries),
            "errors": len(self.errors),
            "header_chars": len(self.build_context_header()),
        }
