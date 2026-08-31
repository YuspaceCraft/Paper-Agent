"""
clarifier.py — 主动追问 & 澄清模块
===========
当用户提问模糊时，主动生成澄清问题；支持用户回答后继续处理。

核心类：
  Clarifier         — 模糊检测 + 追问生成 + 澄清解析
  ClarificationResult — 澄清单次交互的结果

使用方式：
  # CLI 模式
  clarifier = Clarifier()
  result = clarifier.check("这篇论文怎么样？", memory)
  if result.needs_clarification:
      print(result.clarification_message)
      # 用户回答后...
      resolved = clarifier.resolve(result, user_answer)

  # Streamlit 模式
  result = clarifier.check(question, memory)
  if result.needs_clarification:
      # 显示追问选项按钮
      for opt in result.options:
          st.button(opt)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from config import (
    CLARIFY_ENABLED,
    LLM_MODEL,
    OPENAI_API_KEY,
    DASHSCOPE_BASE_URL,
)

if TYPE_CHECKING:
    from agent.memory import BaseMemory


# ================================================================
#  ClarificationResult
# ================================================================

@dataclass
class ClarificationResult:
    """单次澄清检查的结果。"""
    needs_clarification: bool = False
    original_question: str = ""
    clarification_message: str = ""       # 追问文本（自然语言）
    options: list[str] = field(default_factory=list)  # 选项（如有）
    vagueness_type: str = ""              # "ambiguous_reference" | "missing_param" | "too_broad" | "none"
    explanation: str = ""


# ================================================================
#  Clarifier
# ================================================================

class Clarifier:
    """
    主动追问 & 澄清模块。

    检测用户提问的模糊性，并在必要时生成具体的澄清问题。
    支持两种模糊类型：
      1. 指代模糊：用户使用了"这篇"、"该方法"等指代，但无历史上下文
      2. 范围过于宽泛：问题太开放，需要用户缩小范围
      3. 缺少必要参数：如"帮我分析论文"但没有说哪篇
    """

    # 模糊信号模式（中文）
    AMBIGUOUS_REFERENCE_PATTERNS = [
        r"这[个篇种]",
        r"该[方法论文实验研究策略]",
        r"上述",
        r"前面[的]?",
        r"之前[的]?",
        r"它[们的]?",
        r"此[方法论文]",
        r"上[述面文]",
        r"那[个篇种]",
    ]

    # 过于宽泛的信号
    TOO_BROAD_PATTERNS = [
        r"怎么样$", r"如何$", r"好不好$", r"行不行$",
        r"有什么$", r"有哪些$",
        r"全部", r"所有[的]?论文",
        r"帮我分析", r"帮我看看",
    ]

    # 缺少参数信号
    MISSING_PARAM_PATTERNS = [
        r"分析一下$", r"帮我分析$", r"讲一下$", r"说一下$",
    ]

    CLARIFY_PROMPT = """\
你是一个善于沟通的科研助手。用户提出了一个可能模糊的问题，你需要生成 1-3 个具体的澄清追问。

规则：
1. 如果是模糊指代（"这篇"、"该方法"），询问具体指哪篇论文/方法
2. 如果范围太宽（"有什么论文"），询问具体关注的方向/主题
3. 如果缺少参数，询问必要的信息
4. 追问应该是选择题或具体问题，不要开放式"请详细说明"
5. 尽量结合对话历史中已讨论的内容，给出参考选项
6. 如果问题本身已经很清晰，输出 needs_clarification: false

请严格输出以下 JSON 格式：
{{
  "needs_clarification": true/false,
  "vagueness_type": "ambiguous_reference|too_broad|missing_param|none",
  "clarification_message": "追问文本（自然语言）",
  "options": ["选项1", "选项2", "选项3"],
  "explanation": "简短说明为什么需要澄清"
}}

对话历史：{history}

用户问题：{question}

JSON 输出："""

    RESOLVE_PROMPT = """\
你是一个查询优化器。用户原始问题比较模糊，经过追问后得到了补充信息。
请将原始问题和用户的澄清回答合并为一个清晰的查询语句。

原始问题：{original_question}

追问：{clarification_message}

用户回答：{user_response}

请输出一个清晰、完整的查询语句（直接输出查询文本，不要解释）："""

    def _create_llm(self):
        """创建澄清用的 LLM 实例。"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            model=LLM_MODEL,
            temperature=0,
        )

    def check(
        self,
        question: str,
        memory: BaseMemory | None = None,
    ) -> ClarificationResult:
        """
        检查问题是否需要澄清。

        先做快速规则检测，如果命中再用 LLM 精判。
        如果 CLARIFY_ENABLED 为 False，直接返回不需要澄清。

        参数:
            question: 用户问题
            memory:   对话记忆（用于检测是否有历史上下文做指代消解）

        返回:
            ClarificationResult
        """
        if not CLARIFY_ENABLED:
            return ClarificationResult(
                needs_clarification=False,
                original_question=question,
                vagueness_type="none",
            )

        question = question.strip()
        if not question:
            return ClarificationResult(
                needs_clarification=True,
                original_question=question,
                vagueness_type="missing_param",
                clarification_message="请输入你的问题。",
            )

        # 快速规则检测
        has_ambiguous_ref = any(
            re.search(p, question) for p in self.AMBIGUOUS_REFERENCE_PATTERNS
        )
        is_too_broad = any(
            re.search(p, question) for p in self.TOO_BROAD_PATTERNS
        )
        has_missing_param = any(
            re.search(p, question) for p in self.MISSING_PARAM_PATTERNS
        )

        # 如果是指代模糊但有历史上下文，可能不需要澄清
        if has_ambiguous_ref:
            if memory and memory.message_count() > 0:
                # 有历史上下文，但可能仍需要澄清
                # 让 LLM 判断（历史中的指代是否足够明确）
                pass
            else:
                # 无历史上下文，肯定需要澄清
                return self._quick_clarify(
                    question, "ambiguous_reference",
                    "你指的是哪篇论文/方法？当前对话没有历史上下文。",
                    memory,
                )

        # 快速标记：不经过 LLM，直接判断
        if is_too_broad or has_missing_param:
            return self._llm_clarify(question, memory)

        # 没有命中任何规则，检查是否是极短问题
        if len(question) < 10:
            return self._llm_clarify(question, memory)

        # 看起来清晰
        return ClarificationResult(
            needs_clarification=False,
            original_question=question,
            vagueness_type="none",
        )

    def _quick_clarify(
        self,
        question: str,
        vagueness_type: str,
        fallback_msg: str,
        memory: BaseMemory | None,
    ) -> ClarificationResult:
        """快速生成澄清结果（规则+LLM辅助）。"""
        # 尝试用 LLM 生成更好的追问
        try:
            return self._llm_clarify(question, memory)
        except Exception:
            return ClarificationResult(
                needs_clarification=True,
                original_question=question,
                vagueness_type=vagueness_type,
                clarification_message=fallback_msg,
                options=[],
            )

    def _llm_clarify(
        self,
        question: str,
        memory: BaseMemory | None,
    ) -> ClarificationResult:
        """使用 LLM 进行精细化澄清判断。"""
        llm = self._create_llm()

        # 获取对话历史
        history_text = ""
        if memory and memory.message_count() > 0:
            history_text = memory.get_history_text()

        prompt = self.CLARIFY_PROMPT.format(
            history=history_text if history_text else "（无历史，这是首轮对话）",
            question=question,
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"[CLARIFY] [WARN] LLM 澄清调用失败: {e}")
            return ClarificationResult(
                needs_clarification=False,
                original_question=question,
                vagueness_type="none",
            )

        data = self._parse_json(text)

        needs = data.get("needs_clarification", False)
        # 如果确实没有模糊指代且问题足够长，不在"too_broad/ambiguous"边界 → 不需要澄清
        if needs and len(question) > 40:
            vagueness_type = data.get("vagueness_type", "none")
            if vagueness_type == "too_broad" and not any(
                kw in question.lower() for kw in ["所有", "全部", "有哪些", "总结", "综述", "概述"]
            ):
                needs = False

        print(f"[CLARIFY] 检查 \"{question[:40]}...\" → 需要澄清: {needs} "
              f"(类型: {data.get('vagueness_type', 'none')})")

        return ClarificationResult(
            needs_clarification=needs,
            original_question=question,
            clarification_message=data.get("clarification_message", ""),
            options=data.get("options", []),
            vagueness_type=data.get("vagueness_type", "none"),
            explanation=data.get("explanation", ""),
        )

    def resolve(
        self,
        result: ClarificationResult,
        user_response: str,
    ) -> str:
        """
        将用户对追问的回答与原始问题合并，生成清晰的查询。

        参数:
            result:        澄清检查结果
            user_response: 用户对追问的回答

        返回:
            str: 合并后的清晰查询
        """
        if not user_response.strip():
            return result.original_question

        # 简单情况：如果追问是"你指的哪篇论文？"，直接使用用户回答
        if result.vagueness_type == "ambiguous_reference" and len(user_response) < 200:
            # 尝试用用户回答替换原始问题中的模糊指代
            return f"{user_response.strip()} (原始问题: {result.original_question})"

        # 复杂情况：用 LLM 合并
        llm = self._create_llm()
        prompt = self.RESOLVE_PROMPT.format(
            original_question=result.original_question,
            clarification_message=result.clarification_message,
            user_response=user_response,
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            resolved = response.content if hasattr(response, "content") else str(response)
            resolved = resolved.strip()
            print(f"[CLARIFY] [RESOLVE] \"{result.original_question[:30]}...\" + "
                  f"\"{user_response[:30]}...\" → \"{resolved[:80]}...\"")
            return resolved
        except Exception:
            # 降级：简单拼接
            return f"{result.original_question}（补充信息：{user_response}）"

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 响应中提取 JSON 对象。"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 {...} 部分
        match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL)
        if not match:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}


# ================================================================
#  clarify_if_needed — 便捷函数
# ================================================================

def clarify_if_needed(
    question: str,
    memory: BaseMemory | None = None,
) -> ClarificationResult:
    """
    检查问题是否需要澄清，如需要则返回追问。

    这是外部调用的主入口。CLI/Streamlit 在收到用户问题后，
    先调用此函数检查，再决定是追问还是直接进入 Agent/管道。

    参数:
        question: 用户问题
        memory:   对话记忆

    返回:
        ClarificationResult: 包含是否需要澄清及追问内容
    """
    clarifier = Clarifier()
    return clarifier.check(question, memory)


def resolve_clarification(
    result: ClarificationResult,
    user_response: str,
) -> str:
    """
    将用户澄清回答与原始问题合并。

    参数:
        result:        clarify_if_needed 返回的结果
        user_response: 用户对追问的回答

    返回:
        str: 合并后的清晰查询
    """
    clarifier = Clarifier()
    return clarifier.resolve(result, user_response)
