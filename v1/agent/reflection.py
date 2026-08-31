"""
reflection.py — 反思 & 自我修正模块
=============
在 Agent 生成答案后，对其进行批判性反思，必要时触发修正或补充检索。

核心类：
  ReflectionModule  — 反思 + 修正 + 补充检索判断
  reflective_correct — 便捷函数：生成 → 反思 → 修正 → 输出

反思维度：
  1. 忠实度 (Faithfulness):  答案是否基于检索上下文（不是编造的）
  2. 完整性 (Completeness):  答案是否覆盖了问题的所有方面
  3. 准确性 (Accuracy):      引用标注是否正确，事实是否准确

使用方式：
  from agent.reflection import reflective_correct

  # 方式 1: 对已有答案进行反思修正
  answer = reflective_correct(question, raw_answer, memory)

  # 方式 2: 在 Agent 循环中使用
  ref = ReflectionModule()
  verdict = ref.reflect(answer, context, question)
  if verdict.needs_correction:
      answer = ref.self_correct(answer, verdict, context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config import (
    REFLECTION_MAX_ROUNDS,
    REFLECTION_TEMPERATURE,
    LLM_MODEL,
    OPENAI_API_KEY,
    DASHSCOPE_BASE_URL,
)

if TYPE_CHECKING:
    from agent.memory import BaseMemory


# ================================================================
#  ReflectionVerdict — 反思结论 (Pydantic, Phase 3)
# ================================================================

class ReflectionVerdict(BaseModel):
    """反思结论 — 由 LLM with_structured_output 直接返回。

    Field descriptions 会内嵌到 JSON Schema 中传递给 LLM，
    替代旧版 prompt 中手写的 JSON 格式说明。
    """
    needs_correction: bool = Field(
        default=False,
        description="是否需要修正答案。忠实度/完整性/准确性任一低于0.7时设为true",
    )
    needs_more_retrieval: bool = Field(
        default=False,
        description="是否需要补充检索更多信息",
    )
    faithfulness_score: float = Field(
        default=1.0,
        description="忠实度 0.0~1.0: 答案是否基于检索上下文而非编造。1.0=完全忠实，0.0=严重编造",
    )
    completeness_score: float = Field(
        default=1.0,
        description="完整性 0.0~1.0: 答案是否覆盖了问题的所有方面。1.0=完全覆盖",
    )
    accuracy_score: float = Field(
        default=1.0,
        description="准确性 0.0~1.0: 引用标注是否正确，事实是否准确。1.0=完全准确",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="发现的具体问题列表（A/B/C/D类）。无问题时为空列表",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="具体的修正方向建议。无建议时为空列表",
    )


# ================================================================
#  CorrectionOutput — 修正输出 (Pydantic, Phase 3)
# ================================================================

class CorrectionOutput(BaseModel):
    """修正模块的结构化输出。"""
    corrected_answer: str = Field(
        default="",
        description="修正后的完整答案文本",
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="实际做出的修改列表",
    )
    confidence: float = Field(
        default=1.0,
        description="修正后答案的置信度 0.0~1.0",
    )


# ================================================================
#  ReflectionModule
# ================================================================

class ReflectionModule:
    """
    反思模块：对 Agent 生成的答案进行批判性审查。

    使用较高的 temperature 和批判性 prompt，
    让 LLM 扮演"审稿人"角色，找出答案中的问题。
    """

    REFLECTION_PROMPT = """\
检查科研文献助手回复的质量。只关注实际存在的问题，不要吹毛求疵。

**上下文说明：**
  上下文包含两部分：
  1. **工具执行记录** — 每行格式：[工具名] ✅/❌ 结果摘要
     这是助手实际调用的工具、参数和返回结果，是你判断忠实度的唯一依据。
  2. **[检索到的文献原文]** — 检索工具返回的文档内容（如有），用于验证引用准确性。

**检查项（只标记以下三类问题）：**

A. 与执行记录矛盾 — 回复声称的事实与工具实际返回的结果冲突
   ✅ 可接受: "当前知识库中未找到相关文献"（工具未执行或返回空时，这是诚实的概括）
   ✅ 可接受: "知识库"、"文献"、"索引" 等通用概念（不需要在执行记录中逐字定义）
   ❌ 需标记: 声称 "已将 paper.pdf 移至目录 X" 但执行记录无对应操作
   ❌ 需标记: 声称 "检索到 5 篇论文" 但执行记录显示检索失败或为 0
   ❌ 需标记: 引用了执行记录中完全不存在的文件名/论文标题/作者
   ❌ 需标记: 声称的操作数与执行记录中的 ✅ 操作数不符

B. 任务偏离 — 回复未响应用户问题的核心诉求
   ✅ 可接受: 无法完成任务时诚实说明原因并提供替代建议
   ❌ 需标记: 用户要求整理文件，回复却在讨论文献检索结果
   ❌ 需标记: 遗漏了用户明确要求的子任务
   ❌ 需标记: 用户要求分类/移动/整理文件，但回复中没有文件被实际操作的证据

C. 空话/废话 — 回复包含大段与任务无关的通用文本
   ❌ 需标记: 长篇介绍性文字代替实际执行结果
   ❌ 需标记: 编造了用户未要求的规范性建议（如命名规范、兼容性说明等）

D. 任务未完成 — 回复声称完成了任务但执行记录显示核心操作失败或未执行
   ❌ 需标记: 用户要求分类文件，执行记录显示仅列出了目录但没有创建文件夹或移动文件
   ❌ 需标记: 执行记录中核心操作全部失败（❌），但回复却描述了一个"已完成"的结果
   ❌ 需标记: 回复说"已完成分类"但实际上 organize_paper/move_file/create_directory 均未成功
   ✅ 可接受: 回复诚实地说"当前无法完成分类，因为..."（即使任务未完成）

**不需要标记的情况：**
  - 回复使用了 "知识库"、"系统" 等概括性词汇 → 不视为编造
  - 回复对已有记录做了合理归纳（如从多个相似条目中总结规律）→ 不视为矛盾
  - 执行记录中 ❌ 标记的操作 → 如果回复如实说明了失败，则不算矛盾
  - 执行记录为空 → 如果回复表示"无法获取信息"或"未找到"，则不算编造

**需要标记的实质编造（即使原文被摘要截断，以下信息在执行记录中也是可见的）：**
  - 具体文件名、论文标题、作者名称 — 如果执行记录中不存在，视为编造
  - 具体的文件数量/统计数据 — 必须与执行记录一致
  - 声称完成的文件操作（创建/移动/删除）— 执行记录必须有对应 ✅ 操作
  - 引用的检索结果 — 执行记录中必须有对应的文档块或来源

**判断原则：**
  - 如果回复声称的具体事实（文件名、数字、操作）在执行记录中完全不存在 → 标记为 A 类问题
  - 如果回复中某个方向的细节比执行记录更丰富但核心事实与记录一致 → 不视为矛盾
  - 不确定时 → 标记为问题（宁可多报也不漏报编造）

**评分：**
  1.0: 无疑点，回复与执行记录一致且切题
  0.7-0.9: 小瑕疵（如个别统计偏差），不影响整体质量
  0.4-0.6: 存在明显矛盾或偏离，需要修正
  0.0-0.3: 严重编造或完全未响应用户任务

用户问题：{question}

工具执行上下文：{context}

助手回复：{answer}"""

    CORRECTION_PROMPT = """\
修正科研文献助手的回复，解决质量检查发现的问题。

**修正目标：让回复与工具实际执行结果一致，并切合用户任务。**

**修正方法：**
1. 矛盾修正：将回复中与执行记录冲突的内容替换为实际结果
   - 如执行记录显示 3 个文件 → 回复却说了 10 个 → 改为 3 个
   - 如执行记录显示操作失败 → 回复却说成功 → 如实说明失败
2. 偏离修正：将回复焦点拉回用户任务
   - 如用户要求整理文件 → 删除对文献检索结果的长篇讨论
   - 如任务无法完成 → 诚实说明 + 给出下一步建议
3. 空话删除：移除用户未请求的规范性断言和无关内容
4. 补充遗漏：如果检查发现遗漏了用户要求的子任务，补充相关信息

**⚠️ 关键：不修正的内容：**
  - 概括性表述（"知识库"、"文献"、"系统"等）→ 保留
  - 执行记录摘要中不可见的细节 → 保留（除非与记录明确矛盾）
  - 诚实告知失败/未找到的回复 → 保留
  - 有疑问时 → 保留而非删除

用户问题：{question}

工具执行上下文：{context}

当前回复：{answer}

需要修正的问题：{issues}

改进方向：{suggestions}

评分参考：忠实度={faithfulness_score}/1.0, 完整性={completeness_score}/1.0, 准确性={accuracy_score}/1.0

输出修正后的回复（不要解释修改了哪些地方）："""

    def _create_llm(self, temperature: float = REFLECTION_TEMPERATURE):
        """创建反思用的 LLM（比生成用的 temperature 略高，增加批判性）。"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            model=LLM_MODEL,
            temperature=temperature,
        )

    def reflect(
        self,
        answer: str,
        context: str,
        question: str,
        debug_logger=None,
    ) -> ReflectionVerdict:
        """
        对答案进行反思审查（Phase 3: 使用 with_structured_output）。

        参数:
            answer:   Agent 生成的答案
            context:  检索上下文（用于检查忠实度）
            question: 原始用户问题
            debug_logger: 可选的调试日志器

        返回:
            ReflectionVerdict: 包含分数、问题列表和建议（SDK 直接解析为 Pydantic model）
        """
        if not answer:
            return ReflectionVerdict(
                needs_correction=True,
                faithfulness_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                issues=["答案为空"],
                suggestions=["重新生成答案"],
            )

        if not context:
            print("[REFLECT] [WARN] 上下文为空，无法验证答案忠实度，返回保守评分")
            return ReflectionVerdict(
                needs_correction=True,
                faithfulness_score=0.0,
                completeness_score=0.3,
                accuracy_score=0.0,
                issues=["上下文为空，无法验证答案是否基于真实检索结果"],
                suggestions=["补充检索或诚实告知用户当前无法回答"],
            )

        llm = self._create_llm()

        import time as _time
        _start = _time.time()

        prompt = self.REFLECTION_PROMPT.format(
            question=question,
            context=context[:6000],
            answer=answer[:3000],
        )

        try:
            # Phase 3: 用 with_structured_output 替代手动 JSON 解析
            structured_llm = llm.with_structured_output(ReflectionVerdict)
            verdict = structured_llm.invoke(prompt)
        except Exception as e:
            print(f"[REFLECT] [WARN] 反思调用失败: {e}")
            if debug_logger:
                debug_logger.log_error("reflection_llm_error", str(e), phase="reflecting")
            return ReflectionVerdict(
                needs_correction=True,
                faithfulness_score=0.0,
                completeness_score=0.0,
                accuracy_score=0.0,
                issues=[f"反思调用失败: {str(e)[:150]}"],
                suggestions=["重新生成答案"],
            )

        _elapsed_ms = int((_time.time() - _start) * 1000)
        if debug_logger:
            debug_logger.log("reflection_llm_done", phase="reflecting", data={
                "llm_call_duration_ms": _elapsed_ms,
                "context_chars": len(context),
                "answer_chars": len(answer),
            })

        # 安全检查：如果任一分数低于 0.7，强制标记需要修正
        if (verdict.faithfulness_score < 0.7 or
            verdict.completeness_score < 0.7 or
            verdict.accuracy_score < 0.7):
            verdict.needs_correction = True

        avg_score = (verdict.faithfulness_score + verdict.completeness_score + verdict.accuracy_score) / 3
        print(f"[REFLECT] 审查完成: 忠实度={verdict.faithfulness_score:.2f}, "
              f"完整性={verdict.completeness_score:.2f}, 准确性={verdict.accuracy_score:.2f}, "
              f"均分={avg_score:.2f} | "
              f"需要修正={'是' if verdict.needs_correction else '否'}, "
              f"需要补充检索={'是' if verdict.needs_more_retrieval else '否'}")

        if verdict.issues:
            for issue in verdict.issues[:3]:
                print(f"[REFLECT] [ISSUE] {issue}")

        return verdict

    def self_correct(
        self,
        answer: str,
        verdict: ReflectionVerdict,
        context: str,
        question: str = "",
        debug_logger=None,
    ) -> CorrectionOutput:
        """
        根据反思结论修正答案（Phase 3: 返回结构化 CorrectionOutput）。

        参数:
            answer:   原始答案
            verdict:  反思结论
            context:  检索上下文
            question: 原始问题
            debug_logger: 可选的调试日志器

        返回:
            CorrectionOutput: 包含修正后答案、修改列表和置信度
        """
        if not verdict.needs_correction:
            return CorrectionOutput(
                corrected_answer=answer,
                changes_made=[],
                confidence=1.0,
            )

        llm = self._create_llm(temperature=0)  # 修正时用低 temperature 保持稳定

        import time as _time
        _start = _time.time()

        prompt = self.CORRECTION_PROMPT.format(
            question=question,
            context=context[:6000],
            answer=answer[:3000],
            issues="; ".join(verdict.issues) if verdict.issues else "无具体问题",
            suggestions="; ".join(verdict.suggestions) if verdict.suggestions else "无具体建议",
            faithfulness_score=verdict.faithfulness_score,
            completeness_score=verdict.completeness_score,
            accuracy_score=verdict.accuracy_score,
        )

        try:
            # Phase 3: 修正也使用结构化输出
            structured_llm = llm.with_structured_output(CorrectionOutput)
            result = structured_llm.invoke(prompt)
            _elapsed_ms = int((_time.time() - _start) * 1000)
            if debug_logger:
                debug_logger.log("correction_applied", phase="reflecting", data={
                    "llm_call_duration_ms": _elapsed_ms,
                    "original_chars": len(answer),
                    "corrected_chars": len(result.corrected_answer),
                    "changes_count": len(result.changes_made),
                    "confidence": result.confidence,
                })
            print(f"[REFLECT] [CORRECT] 已生成修正答案 (置信度={result.confidence:.2f}, "
                  f"修改={len(result.changes_made)}处)")
            return result
        except Exception as e:
            print(f"[REFLECT] [WARN] 修正失败: {e}")
            if debug_logger:
                debug_logger.log_error("correction_llm_error", str(e), phase="reflecting")
            return CorrectionOutput(
                corrected_answer=answer,
                changes_made=[],
                confidence=0.0,
            )

    def should_retrieve_more(self, verdict: ReflectionVerdict) -> bool:
        """
        判断是否需要触发补充检索。

        条件：
          - 反思结果明确标记 needs_more_retrieval
          - 或忠实度低于阈值（说明缺少上下文）
        """
        return verdict.needs_more_retrieval or verdict.faithfulness_score < 0.5



# ================================================================
#  reflective_correct — 便捷函数
# ================================================================

def reflective_correct(
    question: str,
    answer: str,
    memory: BaseMemory | None = None,
    max_rounds: int | None = None,
    verbose: bool = True,
) -> str:
    """
    反射修正循环：生成答案 → 反思 → 修正 → 再反思 → ...

    参数:
        question:   用户问题
        answer:     原始答案（Agent 已生成的）
        memory:     对话记忆对象
        max_rounds: 最大反思轮数（None 则使用配置值）
        verbose:    是否输出日志

    返回:
        str: 修正后的最终答案
    """
    if max_rounds is None:
        max_rounds = REFLECTION_MAX_ROUNDS

    if max_rounds < 1:
        return answer

    # 获取上下文（用于检查忠实度）
    context = ""
    if memory:
        from agent.memory import HybridMemory
        if hasattr(memory, 'get_context_for_query'):
            ctx = memory.get_context_for_query(question)
            context = "\n".join([
                ctx.get("history", ""),
                ctx.get("long_term_memory", ""),
            ])

    module = ReflectionModule()
    current_answer = answer

    for round_num in range(1, max_rounds + 1):
        if verbose:
            print(f"\n[REFLECT] === 反思轮次 {round_num}/{max_rounds} ===")

        # 反思
        verdict = module.reflect(current_answer, context, question)

        # 不需要修正或补充检索 → 退出循环
        if not verdict.needs_correction and not verdict.needs_more_retrieval:
            if verbose:
                print(f"[REFLECT] 答案质量合格，无需修正")
            break

        # 需要补充检索
        if module.should_retrieve_more(verdict) and round_num < max_rounds:
            if verbose:
                print(f"[REFLECT] 触发补充检索...")
            extra_context = _do_supplemental_retrieval(question, verdict)
            if extra_context:
                context = context + "\n\n[补充检索]\n" + extra_context

        # 修正
        current_answer = module.self_correct(current_answer, verdict, context, question)

    return current_answer


def _do_supplemental_retrieval(question: str, verdict: ReflectionVerdict) -> str:
    """
    基于反思结果进行补充检索。

    根据反思指出的问题，构造更精确的检索查询。
    """
    # 从 issues 和 suggestions 中提取检索线索
    search_queries = []

    if verdict.issues:
        # 将问题转化为检索查询
        for issue in verdict.issues[:2]:
            search_queries.append(issue)
    if verdict.suggestions:
        for sug in verdict.suggestions[:2]:
            search_queries.append(sug)

    if not search_queries:
        search_queries = [question]

    from agent.tools import search_literature

    all_results = []
    for q in search_queries[:3]:
        try:
            result = search_literature.invoke({"query": q, "top_k": 3})
            all_results.append(f"查询: {q}\n{result}")
        except Exception as e:
            all_results.append(f"查询: {q}\n[ERR] {e}")

    return "\n\n".join(all_results)


# 全局单例
_reflection_module: ReflectionModule | None = None


def get_reflection_module() -> ReflectionModule:
    """获取全局反思模块单例。"""
    global _reflection_module
    if _reflection_module is None:
        _reflection_module = ReflectionModule()
    return _reflection_module
