"""
query_rewriter.py — 查询重写与类型分类
================
在检索前对用户查询进行预处理，解决指代消解和查询类型判断。

核心能力：
  - 指代消解：将"这篇论文"、"该方法"等模糊指代替换为具体实体
  - 查询分类：fact（事实查询）/ review（综述）/ compare（对比）/ general（通用）
  - 返回重写后的查询 + 类型标签

使用方式：
  rewriter = QueryRewriter()
  result = rewriter.rewrite("它用的什么损失函数？", memory, llm)
  # result.rewritten → "DenseNet (Huang et al., 2017) 用的什么损失函数？"
  # result.query_type → "fact"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from agent.memory import BaseMemory


# ============================================================
#  RewriteResult — 重写结果
# ============================================================

@dataclass
class RewriteResult:
    """查询重写的结果。"""
    original: str         # 原始查询
    rewritten: str        # 重写后的查询（如果无需重写则等于 original）
    query_type: str       # "fact" | "review" | "compare" | "general"
    needs_rewrite: bool   # 是否实际进行了重写
    explanation: str      # 重写/分类的理由


# ============================================================
#  QueryRewriter — 查询重写器
# ============================================================

class QueryRewriter:
    """
    查询重写器。

    使用 LLM 进行指代消解和查询类型分类。
    当查询包含"这篇论文"、"该方法"、"上述"等模糊指代时，
    结合对话历史将其重写为包含具体实体的完整查询。
    """

    # 模糊指代模式（中文）
    AMBIGUOUS_PATTERNS = [
        r"这[个篇种]",
        r"该[方法论文实验研究]",
        r"上述",
        r"前面",
        r"之前",
        r"它[们的]?",
        r"此[方法论文]",
        r"上[述面文]",
    ]

    REWRITE_PROMPT = (
        "你是一个查询优化器。根据对话历史和用户当前输入，完成以下任务：\n\n"
        "1. 指代消解：如果用户使用了模糊指代（如\"这篇论文\"、\"该方法\"、\"上述实验\"、\"它\"），"
        "请结合对话历史将其替换为具体的实体名称。如果输入已经很清晰，原样返回。\n\n"
        "2. 查询类型分类，从以下类型中选择一个：\n"
        "   - fact: 事实查询（询问具体数据、结果、方法名）\n"
        "   - review: 综述/对比（要求总结、比较多个方案、检索广泛信息）\n"
        "   - compare: 具体对比分析（对比两个或多个具体方法/论文）\n"
        "   - general: 通用问题（无法归入以上类别）\n\n"
        "请严格输出以下 JSON 格式（不要输出其他内容）：\n"
        '{{"rewritten": "重写后的查询", "query_type": "fact|review|compare|general",'
        ' "explanation": "简短的重写/分类理由"}}\n\n'
        "对话历史：\n{history}\n\n"
        "用户输入：{query}"
    )

    def rewrite(
        self,
        query: str,
        memory: BaseMemory,
        llm: ChatOpenAI,
    ) -> RewriteResult:
        """
        重写查询并进行类型分类。

        参数:
            query:  原始用户查询
            memory: 对话记忆（用于获取历史上下文）
            llm:    LLM 实例

        返回:
            RewriteResult: 包含重写查询和类型标签
        """
        query = query.strip()
        if not query:
            return RewriteResult(
                original=query,
                rewritten=query,
                query_type="general",
                needs_rewrite=False,
                explanation="空查询",
            )

        # 快速检查：是否包含模糊指代
        has_ambiguous = self._detect_ambiguous(query)

        # 获取对话历史（最近几轮用于指代消解）
        history_text = self._get_recent_history(memory, turns=3)

        if not has_ambiguous and not history_text:
            # 无需重写，只用规则判断类型
            return RewriteResult(
                original=query,
                rewritten=query,
                query_type=self._rule_based_classify(query),
                needs_rewrite=False,
                explanation="查询清晰，无需重写",
            )

        # 调用 LLM 进行重写和分类
        try:
            result = self._llm_rewrite(query, history_text, llm)
            return result
        except Exception:
            # LLM 调用失败，回退到原始查询
            return RewriteResult(
                original=query,
                rewritten=query,
                query_type=self._rule_based_classify(query),
                needs_rewrite=False,
                explanation="重写失败，回退到原始查询",
            )

    def _detect_ambiguous(self, query: str) -> bool:
        """检测查询是否包含模糊指代。"""
        for pattern in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _get_recent_history(self, memory: BaseMemory, turns: int = 3) -> str:
        """获取最近 N 轮对话历史（文本格式）。"""
        messages = memory.get_messages()
        if not messages:
            return ""

        # 取最后 N 轮（2 * turns 条消息）
        recent = messages[-(2 * turns):]
        lines = []
        for msg in recent:
            role = "用户" if msg.role == "human" else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _llm_rewrite(
        self,
        query: str,
        history: str,
        llm: ChatOpenAI,
    ) -> RewriteResult:
        """调用 LLM 进行重写。"""
        prompt = self.REWRITE_PROMPT.format(query=query, history=history or "（无历史）")

        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)

        # 解析 JSON 响应
        data = self._parse_json(text)
        rewritten = data.get("rewritten", query)
        query_type = data.get("query_type", "general")
        explanation = data.get("explanation", "")

        # 验证 query_type
        if query_type not in ("fact", "review", "compare", "general"):
            query_type = self._rule_based_classify(rewritten)

        needs_rewrite = rewritten != query

        if needs_rewrite:
            print(f"[REWRITE] 指代消解: \"{query[:40]}...\" → \"{rewritten[:60]}...\"")
        print(f"[REWRITE] 查询类型: {query_type}")

        return RewriteResult(
            original=query,
            rewritten=rewritten,
            query_type=query_type,
            needs_rewrite=needs_rewrite,
            explanation=explanation,
        )

    def _rule_based_classify(self, query: str) -> str:
        """基于规则快速判断查询类型（备选方案）。"""
        q = query.lower()

        # fact 信号词
        fact_signals = [
            "准确率", "精度", "得分", "多少", "几个", "什么方法",
            "损失函数", "数据集", "性能", "accuracy", "score", "多少层",
            "参数", "结果", "数值",
        ]
        # compare 信号词
        compare_signals = [
            "对比", "比较", "区别", "差异", "优缺点", "vs", "相比",
            "哪个更好", "有何不同", "分别",
        ]
        # review 信号词
        review_signals = [
            "综述", "总结", "概述", "有哪些", "现状", "进展",
            "发展", "有哪些方法", "所有", "概览",
        ]

        for s in compare_signals:
            if s in q:
                return "compare"
        for s in fact_signals:
            if s in q:
                return "fact"
        for s in review_signals:
            if s in q:
                return "review"
        return "general"

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 响应中提取 JSON 对象。"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 尝试提取 {...} 部分
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


# 全局单例
_query_rewriter: QueryRewriter | None = None


def get_query_rewriter() -> QueryRewriter:
    """获取全局查询重写器单例。"""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter
