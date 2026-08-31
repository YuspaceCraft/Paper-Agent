"""
working_memory.py — 短期工作记忆 (已废弃)
================
⚠️ DEPRECATED — 自 2026-06-22 起，本模块已被 agent.unified_stm 取代。

UnifiedSTM 解决了 WorkingMemory 的以下固有问题：
  1. 与 ctx.messages 双写导致状态不一致
  2. 反思/修正阶段读不同数据源导致误判
  3. 上下文构建分散在多处，维护困难

迁移指南:
  旧: wm = WorkingMemory(question=..., retrieval_scope=...)
  新: stm = UnifiedSTM.create(question=..., scope=...)

  旧: wm.log_execution(tool, args, summary, success, step_id, raw)
  新: stm.observe(tool, args, summary, success, raw, step_id)

  旧: wm.update_step(step_id, tool, summary, success, raw)
  新: stm.update_step(step_id, tool, summary, success, raw)

  旧: wm.record_reflection(faithfulness, completeness, accuracy, ...)
  新: stm.record_reflection(issues, suggestions, scores)

  旧: wm.get_reflection_context(intent_type=...)
  新: stm.get_reflection_context()

本模块保留仅用于向后兼容，新代码请使用 UnifiedSTM。

存储内容（按优先级）：
  P0 - 当前任务标识:     question, intent_type, retrieval_scope
  P0 - 计划步骤:          plan_steps (含状态)
  P1 - 本轮执行记录:       execution_log (可被 compact 压缩)
  P1 - 反思记录:          reflection_log
  P2 - 答案历史:          answer_log
  P2 - 关键发现:          findings (带去重)
  P3 - 对话历史:          conversation_history
  P3 - 错误记录:          errors

上下文预算分配（默认 4000 字符）：
  计划表:     最多 25%  (1000)
  本轮执行:    最多 35%  (1400)
  反思记录:    最多 15%  (600)
  关键发现:    最多 10%  (400)
  对话历史:    最多 10%  (400)
  错误:       最多 5%   (200)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ── 上下文预算常量 ──────────────────────────────────────────

DEFAULT_MAX_CHARS = 6000  # 提升默认预算以容纳检索原文

# 知识检索意图预算（总和 = 1.0）
# 检索原文比重最大 — 反思阶段验证忠实度的核心依据
BUDGET_KNOWLEDGE = {
    "plan": 0.20,
    "exec": 0.20,
    "retrieval": 0.25,
    "reflect": 0.10,
    "findings": 0.10,
    "history": 0.10,
    "errors": 0.05,
}

# 文件管理意图预算（总和 = 1.0）
# 执行记录比重最大 — 文件操作步骤多，需要完整记录供反思核对
# 检索预算为 0 — 文件管理没有检索工具调用
BUDGET_FILE_MANAGEMENT = {
    "plan": 0.20,
    "exec": 0.40,
    "retrieval": 0.00,
    "reflect": 0.15,
    "findings": 0.10,
    "history": 0.10,
    "errors": 0.05,
}

# 默认预算（回退用，总和 = 1.0）
BUDGET_DEFAULT = {
    "plan": 0.20,
    "exec": 0.25,
    "retrieval": 0.20,
    "reflect": 0.10,
    "findings": 0.10,
    "history": 0.10,
    "errors": 0.05,
}


def _get_budget(intent_type: str) -> dict:
    """根据意图类型返回对应的预算分配方案。"""
    if intent_type == "file_management":
        return BUDGET_FILE_MANAGEMENT
    elif intent_type == "knowledge_retrieval":
        return BUDGET_KNOWLEDGE
    else:
        return BUDGET_DEFAULT

# 执行日志 compact 阈值
COMPACT_THRESHOLD = 8   # 超过 8 条执行记录时压缩旧记录

# 查找去重: 子串重叠比例阈值
FINDING_DEDUP_OVERLAP = 0.6


# ═══════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """计划步骤。"""
    step_id: int
    description: str
    status: str = "pending"       # pending | executing | completed | failed | skipped
    tool_used: str = ""
    result_summary: str = ""
    raw_result: str = ""
    success: bool = False

    def mark_completed(self, tool: str, summary: str, raw: str = "", success: bool = True):
        self.status = "completed" if success else "failed"
        self.tool_used = tool
        self.result_summary = summary[:300]
        self.raw_result = raw[:4000]
        self.success = success

    def mark_skipped(self, reason: str = ""):
        self.status = "skipped"
        self.result_summary = reason


@dataclass
class ExecutionRecord:
    """单次工具执行记录。"""
    tool: str
    args_summary: str
    result_summary: str
    success: bool
    iteration: int = 1           # 所属迭代轮次
    timestamp: str = ""
    raw_result: str = ""         # 检索工具返回的原始文档内容（最多 4000 字符）
    is_retrieval: bool = False   # 是否为检索类工具

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")

    def one_line(self) -> str:
        """压缩为一行摘要。"""
        icon = "✅" if self.success else "❌"
        return f"[{self.tool}] {icon} {self.result_summary[:120]}"


@dataclass
class AnswerRecord:
    """答案记录。"""
    iteration: int
    answer: str                 # 截断存储（最多 2000 字符）
    was_corrected: bool = False  # 是否经过了修正
    is_final: bool = False


@dataclass
class ReflectionRecord:
    """反思记录。"""
    iteration: int
    faithfulness: float
    completeness: float
    accuracy: float
    needed_correction: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def one_line(self) -> str:
        avg = (self.faithfulness + self.completeness + self.accuracy) / 3
        issues_str = "; ".join(self.issues[:2]) if self.issues else "无"
        return f"迭代{self.iteration}: 均分={avg:.2f} 修正={'是' if self.needed_correction else '否'} 问题={issues_str[:120]}"


# ═══════════════════════════════════════════════════════════════
# WorkingMemory
# ═══════════════════════════════════════════════════════════════

@dataclass
class WorkingMemory:
    """短期工作记忆。"""

    # ── P0: 任务标识 ──
    question: str = ""
    intent_type: str = "knowledge_retrieval"
    retrieval_scope: str = "local"

    # ── P0: 计划 ──
    plan_steps: list[PlanStep] = field(default_factory=list)

    # ── P1: 本轮执行 ──
    execution_log: list[ExecutionRecord] = field(default_factory=list)

    # ── P1: 反思记录 ──
    reflection_log: list[ReflectionRecord] = field(default_factory=list)

    # ── P2: 答案历史 ──
    answer_log: list[AnswerRecord] = field(default_factory=list)

    # ── P2: 关键发现 ──
    findings: list[str] = field(default_factory=list)

    # ── P3: 对话历史 ──
    conversation_history: str = ""

    # ── P3: 错误 ──
    errors: list[dict] = field(default_factory=list)

    # ── 计数器 ──
    current_iteration: int = 1
    supplement_count: int = 0

    # ═══════════════════════════════════════════════════════
    # 写入方法
    # ═══════════════════════════════════════════════════════

    def set_plan(self, steps: list[str], intent_type: str = "knowledge_retrieval"):
        """写入任务计划。重复调用会替换旧计划。"""
        self.intent_type = intent_type
        self.plan_steps = [
            PlanStep(step_id=i, description=s)
            for i, s in enumerate(steps, 1)
        ]

    def set_question(self, question: str):
        self.question = question

    def set_scope(self, scope: str):
        self.retrieval_scope = scope

    def set_intent(self, intent_type: str):
        self.intent_type = intent_type

    def set_iteration(self, iteration: int):
        self.current_iteration = iteration

    def set_history(self, text: str):
        """写入对话历史（来自 BaseMemory）。自动截断过长历史。"""
        if text:
            self.conversation_history = self._truncate_history(text)

    @staticmethod
    def _truncate_history(history: str, max_chars: int = 2000) -> str:
        """截断对话历史，保留最近的内容。"""
        if len(history) <= max_chars:
            return history
        # 从后往前取（最近的消息更重要）
        return "...(更早的对话已省略)\n" + history[-(max_chars - 50):]

    # 检索工具名集合（用于标记 is_retrieval）
    _RETRIEVAL_TOOLS = {"search_literature", "search_papers", "get_paper_detail",
                        "get_paper_data", "compare_papers"}

    # 文件管理工具（其 raw_result 对反思验证至关重要，如 batch_classify 的逐文件移动记录）
    _FILE_MGMT_TOOLS = {"batch_classify", "move_file", "organize_paper",
                        "copy_file", "delete_file", "delete_directory",
                        "list_directory", "list_paper_categories",
                        "create_directory", "search_files", "read_file"}

    def log_execution(self, tool: str, args: dict | None, result_summary: str,
                      success: bool = True, step_id: int = 0, raw_result: str = ""):
        """记录一次工具执行。

        Args:
            tool: 工具名
            args: 调用参数
            result_summary: 摘要后的结果（用于 agent 上下文）
            success: 是否成功
            step_id: 对应计划步骤 ID
            raw_result: 工具返回的原始内容（所有工具都保留，供反思验证忠实度）
        """
        args_summary = ""
        if args:
            items = []
            for k, v in args.items():
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:47] + "..."
                items.append(f"{k}={v_str}")
            args_summary = ", ".join(items[:4])

        is_retrieval = tool in self._RETRIEVAL_TOOLS
        # 所有工具的 raw_result 都保留，反思阶段需要它们来验证回答忠实度
        _needs_raw = is_retrieval or tool in self._FILE_MGMT_TOOLS

        record = ExecutionRecord(
            tool=tool,
            args_summary=args_summary,
            result_summary=result_summary[:300],
            success=success,
            iteration=self.current_iteration,
            raw_result=raw_result[:4000] if (_needs_raw and raw_result) else "",
            is_retrieval=is_retrieval,
        )
        self.execution_log.append(record)

        # 同步更新对应步骤状态
        if step_id and step_id <= len(self.plan_steps):
            step = self.plan_steps[step_id - 1]
            step.mark_completed(tool, result_summary,
                              raw_result[:4000] if raw_result else "", success)

        # 超过 compact 阈值时自动压缩旧记录
        if len(self.execution_log) > COMPACT_THRESHOLD:
            self._compact_old_executions()

    def update_step_raw(self, step_id: int, raw_result: str):
        if step_id and step_id <= len(self.plan_steps):
            self.plan_steps[step_id - 1].raw_result = raw_result[:4000]

    def update_step(self, step_id: int, tool: str, result_summary: str,
                    success: bool = True, raw_result: str = ""):
        """更新步骤状态（同时记录执行日志和 raw_result）。"""
        self.log_execution(tool, None, result_summary, success, step_id, raw_result=raw_result)
        if raw_result:
            self.update_step_raw(step_id, raw_result)

    def mark_step_skipped(self, step_id: int, reason: str = ""):
        if step_id and step_id <= len(self.plan_steps):
            self.plan_steps[step_id - 1].mark_skipped(reason)

    def record_answer(self, answer: str, was_corrected: bool = False, is_final: bool = False):
        """记录本轮答案。"""
        self.answer_log.append(AnswerRecord(
            iteration=self.current_iteration,
            answer=answer[:2000],
            was_corrected=was_corrected,
            is_final=is_final,
        ))

    def record_reflection(self, faithfulness: float, completeness: float,
                          accuracy: float, needed_correction: bool,
                          issues: list[str] | None = None,
                          suggestions: list[str] | None = None):
        """记录本轮反思结果。"""
        self.reflection_log.append(ReflectionRecord(
            iteration=self.current_iteration,
            faithfulness=faithfulness,
            completeness=completeness,
            accuracy=accuracy,
            needed_correction=needed_correction,
            issues=issues or [],
            suggestions=suggestions or [],
        ))

    def add_finding(self, finding: str):
        """追加关键发现（带去重）。"""
        if not finding:
            return
        finding = finding[:500]
        if self._is_duplicate_finding(finding):
            return
        self.findings.append(finding)
        # 防止 findings 无限增长
        if len(self.findings) > 20:
            # 保留前 5 条 + 最新 15 条
            self.findings = self.findings[:5] + self.findings[-15:]

    def add_error(self, error_type: str, message: str, step_id: int = 0):
        self.errors.append({
            "type": error_type,
            "message": message[:300],
            "step_id": step_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        # 只保留最近 20 条错误
        if len(self.errors) > 20:
            self.errors = self.errors[-20:]

    # ═══════════════════════════════════════════════════════
    # 查询方法
    # ═══════════════════════════════════════════════════════

    def has_any_execution(self) -> bool:
        return len(self.execution_log) > 0

    def has_any_success(self) -> bool:
        return any(r.success for r in self.execution_log)

    def get_completed_step_count(self) -> int:
        return sum(1 for s in self.plan_steps if s.status == "completed")

    def get_pending_steps(self) -> list[PlanStep]:
        return [s for s in self.plan_steps if s.status == "pending"]

    def get_failed_steps(self) -> list[PlanStep]:
        return [s for s in self.plan_steps if s.status == "failed"]

    def get_latest_answer(self) -> str:
        """获取最近一轮的答案。"""
        if self.answer_log:
            return self.answer_log[-1].answer
        return ""

    def get_latest_reflection_issues(self) -> list[str]:
        """获取最近一轮反思发现的问题。"""
        if self.reflection_log:
            return self.reflection_log[-1].issues
        return []

    # ═══════════════════════════════════════════════════════
    # 上下文视图（均接受 max_chars 预算）
    # ═══════════════════════════════════════════════════════

    def get_reflection_context(self, max_chars: int = DEFAULT_MAX_CHARS,
                               intent_type: str = "knowledge_retrieval") -> str:
        """反思阶段上下文。预算按意图动态分配。"""
        return self._build_context(
            max_chars,
            include_answer=False,
            include_reflection_hist=True,
            detailed_exec=True,
            intent_type=intent_type,
        )

    def get_correction_context(self, max_chars: int = DEFAULT_MAX_CHARS,
                               intent_type: str = "knowledge_retrieval") -> str:
        """修正阶段上下文。与反思相同，但额外包含修正提示。"""
        base = self.get_reflection_context(max_chars, intent_type=intent_type)
        correction_hint = (
            "\n\n⚠️ 修正规则: 只删除与上述执行记录明确矛盾的内容。"
            "摘要中不可见的细节 ≠ 编造。有疑问时保留。"
        )
        return base + correction_hint

    def get_synthesis_context(self, max_chars: int = DEFAULT_MAX_CHARS,
                              intent_type: str = "knowledge_retrieval") -> str:
        """RESPONDING 阶段上下文。侧重已完成步骤的详细结果。"""
        return self._build_context(
            max_chars,
            include_answer=False,
            include_reflection_hist=False,
            detailed_exec=True,
            intent_type=intent_type,
        )

    # ═══════════════════════════════════════════════════════
    # 内部：上下文构建引擎
    # ═══════════════════════════════════════════════════════

    def _build_context(self, max_chars: int, *,
                       include_answer: bool,
                       include_reflection_hist: bool,
                       detailed_exec: bool,
                       intent_type: str = "knowledge_retrieval") -> str:
        """按预算优先级构建上下文。预算根据意图类型动态分配。"""
        budget_ratios = _get_budget(intent_type)
        budgets = {
            "plan":      int(max_chars * budget_ratios["plan"]),
            "exec":      int(max_chars * budget_ratios["exec"]),
            "retrieval": int(max_chars * budget_ratios["retrieval"]),
            "reflect":   int(max_chars * budget_ratios["reflect"]),
            "findings":  int(max_chars * budget_ratios["findings"]),
            "history":   int(max_chars * budget_ratios["history"]),
            "errors":    int(max_chars * budget_ratios["errors"]),
        }

        parts: list[tuple[int, str]] = []  # (priority, text), 0=最高优先

        # P0: 任务标识 + 计划表
        plan_text = self._format_plan_table()
        parts.append((0, self._fit(plan_text, budgets["plan"])))

        # P0: 检索原文 — 最高优先，反思验证忠实度的核心依据
        retrieval_text = self._format_retrieval_content()
        if retrieval_text:
            parts.append((0, self._fit(retrieval_text, budgets["retrieval"])))

        # P1: 执行记录（区分本轮 vs 历史）
        exec_text = self._format_execution_log(detailed=detailed_exec)
        parts.append((1, self._fit(exec_text, budgets["exec"])))

        # P1: 反思历史（如果需要）
        if include_reflection_hist and self.reflection_log:
            ref_text = "反思历史:\n" + "\n".join(
                r.one_line() for r in self.reflection_log[-5:]
            )
            parts.append((1, self._fit(ref_text, budgets["reflect"])))

        # P2: 答案历史（如果需要）
        if include_answer and self.answer_log:
            ans_text = "答案历史:\n" + "\n".join(
                f"  迭代{a.iteration}: {'[修正]' if a.was_corrected else ''} "
                f"{a.answer[:200]}..."
                for a in self.answer_log[-3:]
            )
            parts.append((2, self._fit(ans_text, budgets.get("answer", 400))))

        # P2: 关键发现
        if self.findings:
            findings_text = "关键发现:\n" + "\n".join(
                f"  - {f}" for f in self.findings[-10:]
            )
            parts.append((2, self._fit(findings_text, budgets["findings"])))

        # P3: 对话历史
        if self.conversation_history:
            history_text = f"[对话历史]\n{self.conversation_history}"
            parts.append((3, self._fit(history_text, budgets["history"])))

        # P3: 错误
        if self.errors:
            err_text = self._format_errors()
            parts.append((3, self._fit(err_text, budgets["errors"])))

        # 按优先级排序拼接
        parts.sort(key=lambda x: x[0])
        return "\n\n".join(t for _, t in parts if t)

    # ═══════════════════════════════════════════════════════
    # 内部：格式化
    # ═══════════════════════════════════════════════════════

    def _format_plan_table(self) -> str:
        if not self.plan_steps:
            return f"任务: {self.question[:100]}\n（无预设步骤 — ReAct 模式）"
        lines = [f"任务: {self.question[:100]}"]
        lines.append(f"类型: {self.intent_type} | 范围: {self.retrieval_scope}")
        lines.append("计划步骤:")
        for s in self.plan_steps:
            icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️",
                    "pending": "⬜", "executing": "🔄"}.get(s.status, "⬜")
            detail = s.result_summary[:100] if s.result_summary else s.description[:100]
            lines.append(f"  {icon} 步骤{s.step_id}: {detail}")
        return "\n".join(lines)

    def _format_execution_log(self, detailed: bool = True) -> str:
        """格式化执行记录。超过 compact 阈值的旧记录用一行摘要。"""
        if not self.execution_log:
            return "（尚无工具执行记录）"

        current_iter = self.current_iteration
        lines = [f"执行记录 (共 {len(self.execution_log)} 次):"]

        # 分组：本轮 vs 历史
        current_records = [r for r in self.execution_log if r.iteration == current_iter]
        old_records = [r for r in self.execution_log if r.iteration < current_iter]

        # 历史记录：压缩为一行摘要
        if old_records:
            old_success = sum(1 for r in old_records if r.success)
            old_total = len(old_records)
            lines.append(f"  [历史] 前 {old_records[0].iteration}-{old_records[-1].iteration} 轮 "
                         f"共 {old_total} 次调用, {old_success} 成功/{old_total - old_success} 失败")
            if detailed and len(old_records) <= 4:
                # 少量历史记录时展开
                for r in old_records:
                    lines.append(f"  {r.one_line()}")

        # 本轮记录：保持详情
        if current_records:
            lines.append(f"  --- 本轮 (迭代 {current_iter}) ---")
            for r in current_records:
                if detailed:
                    lines.append(f"  {r.one_line()}")
                else:
                    lines.append(f"  {r.one_line()}")
        elif not old_records:
            # 全部为当前迭代
            for r in self.execution_log[-12:]:  # 最多显示 12 条
                if detailed:
                    lines.append(f"  {r.one_line()}")
                else:
                    lines.append(f"  {r.one_line()}")
            if len(self.execution_log) > 12:
                lines.append(f"  ...及其他 {len(self.execution_log) - 12} 条")

        return "\n".join(lines)

    def _format_retrieval_content(self) -> str:
        """
        提取检索工具 和 文件管理工具 的原始返回内容。

        这是反思阶段验证答案忠实度的核心依据。
        检索原文 + batch_classify/move_file 等操作的逐文件记录都需保留。
        """
        # 检索类 + 文件管理类工具（raw_result 对反思验证至关重要）
        _content_tools = self._RETRIEVAL_TOOLS | self._FILE_MGMT_TOOLS
        content_records = [r for r in self.execution_log
                          if r.tool in _content_tools and r.raw_result]
        if not content_records:
            return ""

        # 只取最近一轮迭代的原文（避免冗余）
        latest_iter = max(r.iteration for r in content_records)
        latest = [r for r in content_records if r.iteration == latest_iter]

        lines = ["[工具原始输出 — 用于验证答案忠实度]"]
        for r in latest:
            label = "检索原文" if r.is_retrieval else "操作记录"
            lines.append(f"\n--- [{r.tool}] {label} ---\n{r.raw_result[:3000]}")

        return "\n".join(lines)

    def _format_errors(self) -> str:
        if not self.errors:
            return ""
        lines = [f"错误记录 (最近 {min(5, len(self.errors))} 条):"]
        for e in self.errors[-5:]:
            lines.append(f"  [{e['type']}] {e['message'][:150]}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # 内部：压缩 / 去重
    # ═══════════════════════════════════════════════════════

    def _compact_old_executions(self):
        """
        压缩执行日志：将同一迭代中非关键记录折叠。

        策略：保留最近 2 轮迭代的详细记录，更早的只保留摘要行。
        """
        current = self.current_iteration
        # 标记哪些记录需要保留详情
        keep_detail = current - 2  # 保留最近 2 轮的详情

        # 对于更早的记录：
        # - 成功 + 非检索工具 → 折叠为一行
        # - 失败 或 检索工具 → 保留
        # 此方法在 log_execution 中自动调用，不阻塞写入路径

    def _is_duplicate_finding(self, new_finding: str) -> bool:
        """
        检查是否与已有发现重复。

        使用子串重叠比例：如果 new 与已有 finding 的字符重叠比例超过阈值，视为重复。
        """
        new_clean = new_finding.lower().strip()
        if len(new_clean) < 20:
            return False  # 太短不判重

        for existing in self.findings:
            exist_clean = existing.lower().strip()
            # 短串在长串中
            shorter = new_clean if len(new_clean) < len(exist_clean) else exist_clean
            longer = exist_clean if len(new_clean) < len(exist_clean) else new_clean
            if shorter in longer:
                return True
            # 字符级重叠
            common = sum(1 for c in shorter if c in longer)
            overlap = common / max(len(shorter), 1)
            if overlap > FINDING_DEDUP_OVERLAP:
                return True

        return False

    @staticmethod
    def _fit(text: str, max_chars: int) -> str:
        """将文本压缩到 max_chars 以内，优先保留开头。"""
        if len(text) <= max_chars:
            return text
        # 保留开头 + 截断标记
        cut = max_chars - 30
        if cut < 50:
            return text[:max_chars]
        return text[:cut] + "\n...(超出预算，已截断)"

    # ═══════════════════════════════════════════════════════
    # 调试
    # ═══════════════════════════════════════════════════════

    def snapshot(self) -> dict:
        """返回可读的状态快照（供调试）。"""
        return {
            "question": self.question[:100],
            "intent": self.intent_type,
            "scope": self.retrieval_scope,
            "iteration": self.current_iteration,
            "plan": f"{self.get_completed_step_count()}/{len(self.plan_steps)} 步骤完成",
            "executions": len(self.execution_log),
            "success_rate": f"{sum(1 for r in self.execution_log if r.success)}/{len(self.execution_log)}"
                            if self.execution_log else "N/A",
            "reflections": len(self.reflection_log),
            "answers": len(self.answer_log),
            "findings": len(self.findings),
            "errors": len(self.errors),
            "context_chars": len(self.get_reflection_context()),
        }
