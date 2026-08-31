"""
judge.py — LLM-as-Judge
========
使用现有 DashScope LLM 作为自动评测"裁判"，评估生成质量三大维度：

  1. 忠实度 (Faithfulness)  — 答案是否基于提供的上下文（检测幻觉）
  2. 答案相关性 (Relevance) — 答案是否切题
  3. 正确性 (Correctness)   — 答案与参考答案的语义一致性

原理：
  每个维度都有精心设计的提示模板，引导 LLM 输出结构化 JSON。
  使用 temperature=0 确保评测结果稳定、可复现。

重用现有 API：
  - src.generator.create_llm()  — ChatOpenAI 实例（temperature=0）
"""

import json
import re
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from agent.generator import create_llm


# ================================================================
#  提示模板
# ================================================================

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一位严谨的评测专家。你的任务是检查生成的答案是否忠实于提供的上下文（检索到的文档内容）。

评测步骤：
1. 从生成的答案中提取每一个原子事实声明（atomic factual claim）。
2. 对于每个声明，判断它是否能被提供的上下文直接支持。
3. "支持"意味着上下文明确陈述了该事实，或可以通过直接推理得出（无需额外假设）。
4. 如果一个声明部分正确但包含上下文中没有的细节，标记为 false。
5. 计算忠实度分数 = 被支持的声明数 / 总声明数（无声明时为 0.0）。

输出要求：
  只输出合法的 JSON，不要有任何其他文本：
  {{
    "claims": [
      {{"text": "...", "supported": true/false, "evidence": "..."}}
    ],
    "score": 0.XX
  }}"""),
    ("human", """\
问题：
{question}

上下文（检索到的文档）：
{context}

生成的答案：
{answer}

请评测忠实度，输出 JSON。"""),
])


RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一位严谨的评测专家。你的任务是评估生成的答案与用户问题的相关性。

评分标准（1-5 分）：
  1 分 — 完全无关 / 答非所问
  2 分 — 部分相关但严重偏离核心问题
  3 分 — 基本相关，回答了部分问题
  4 分 — 较好地切题，回答了核心问题
  5 分 — 完美切题，全面且直接地回答了问题

输出要求：
  只输出合法的 JSON，不要有任何其他文本：
  {{
    "score": N,
    "reasoning": "..."
  }}"""),
    ("human", """\
问题：
{question}

生成的答案：
{answer}

请评估相关性，输出 JSON。"""),
])


CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一位严谨的评测专家。你的任务是比较生成的答案与参考答案的语义一致性。

评分标准（1-5 分）：
  1 分 — 完全错误 / 与参考答案严重矛盾
  2 分 — 大部分错误，有严重遗漏或事实错误
  3 分 — 部分正确，存在一些错误或遗漏
  4 分 — 基本正确，仅有措辞上的细微差异
  5 分 — 在语义上与参考答案高度一致

注意：
  - 措辞不同但语义相同应得高分
  - 生成的答案包含参考答案没有的额外正确信息不应扣分
  - 生成的答案遗漏了参考答案中的关键信息应酌情扣分

输出要求：
  只输出合法的 JSON，不要有任何其他文本：
  {{
    "score": N,
    "reasoning": "..."
  }}"""),
    ("human", """\
问题：
{question}

生成的答案：
{answer}

参考答案：
{reference}

请比较并评估正确性，输出 JSON。"""),
])


# ================================================================
#  记忆评测提示模板
# ================================================================

MEMORY_EXTRACTION_ACCURACY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一位严谨的记忆质量评测专家。你的任务是评估从对话中提取的长期记忆(LTM)事实的准确性。

评测步骤：
1. 阅读完整对话，理解上下文。
2. 对于每条提取的记忆事实，判断它是否准确反映了对话内容。
3. 评分标准：
   - accurate: 事实完全正确，与对话内容一致
   - partially_accurate: 大体正确但有细微偏差或遗漏
   - inaccurate: 与对话内容矛盾或严重歪曲
4. 额外检查：
   - redundant: 是否与列表中的其他记忆重复
   - worth_remembering: 是否值得长期记住（对后续对话有帮助）

输出要求：
  只输出合法的 JSON：
  {{
    "evaluations": [
      {{"fact": "...", "accuracy": "accurate|partially_accurate|inaccurate", "redundant": false, "worth_remembering": true, "reason": "..."}}
    ],
    "overall_accuracy": 0.XX,
    "missed_important_facts": ["对话中重要但未被提取的事实（如果有）"]
  }}"""),
    ("human", """\
对话记录：
{conversation}

提取的长期记忆事实：
{extracted_facts}

请评估记忆提取质量，输出 JSON。"""),
])


MEMORY_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一位严谨的记忆检索评测专家。你的任务是评估长期记忆(LTM)事实与当前用户查询的相关性。

评分标准（1-5 分）：
  1 分 — 完全无关
  2 分 — 弱相关（话题相近但无法帮助回答）
  3 分 — 部分相关（提供了一些背景但不够直接）
  4 分 — 相关（直接有助于回答查询）
  5 分 — 高度相关（对回答查询至关重要）

输出要求：
  只输出合法的 JSON：
  {{
    "relevance_scores": [
      {{"fact": "...", "score": N, "reason": "..."}}
    ],
    "avg_relevance": 0.0
  }}"""),
    ("human", """\
用户查询：
{query}

长期记忆事实列表：
{ltm_facts}

请评估每条记忆与查询的相关性，输出 JSON。"""),
])


MEMORY_IMPACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一位严谨的记忆效用评测专家。你的任务是评估长期记忆(LTM)对生成答案质量的影响。

你将看到同一个查询的两个答案：
- 答案A：使用了长期记忆生成的答案
- 答案B：未使用长期记忆生成的答案

评测步骤：
1. 比较两个答案的：准确性、完整性、针对性、简洁性
2. 判断长期记忆是否改善了答案质量
3. 判断长期记忆中的信息是否被恰当使用

输出要求：
  只输出合法的 JSON：
  {{
    "impact_score": -1|0|1|2,
    "impact_label": "worse|same|better|much_better",
    "ltm_utilized": true|false,
    "comparison_notes": "...",
    "winner": "A|B|tie"
  }}

impact_score说明：-1=变差, 0=无影响, 1=改善, 2=显著改善"""),
    ("human", """\
用户查询：
{query}

使用的长期记忆：
{ltm_used}

答案A（使用长期记忆）：
{answer_with}

答案B（未使用长期记忆）：
{answer_without}

请评估记忆影响，输出 JSON。"""),
])


# ================================================================
#  JSON 提取工具
# ================================================================

_JSON_PATTERN = re.compile(r"\{[^{}]*\{[^{}]*\}[^{}]*\}|(\{[^{}]*\})")

def _find_json_block(text: str) -> str | None:
    """从 LLM 输出中提取 JSON 块。

    先尝试直接解析，失败后用正则找第一个 {...} 块。
    支持嵌套 JSON（如 claims 数组中有对象）。
    """
    text = text.strip()

    # 尝试直接找最外层 { ... }
    # 用更健壮的方式：找到第一个 { 和对应的 }
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _parse_judge_json(text: str) -> dict:
    """解析 LLM judge 输出的 JSON，带容错。

    抛出: ValueError 如果无法解析
    """
    # 先尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 找 JSON 块
    json_block = _find_json_block(text)
    if json_block:
        try:
            return json.loads(json_block)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从输出中解析 JSON: {text[:200]}...")


# ================================================================
#  LLMJudge
# ================================================================

class LLMJudge:
    """使用现有 LLM 做自动评测裁判。

    用法:
        judge = LLMJudge()
        result = judge.evaluate_faithfulness(question, answer, context)
        print(result)  # {"claims": [...], "score": 0.85}
    """

    def __init__(self, llm: ChatOpenAI | None = None):
        """
        参数:
            llm: ChatOpenAI 实例，None 则自动创建（temperature=0）
        """
        self.llm = llm or create_llm()
        # 确保 temperature=0 以获得确定性评测
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = 0

    # ------------------------------------------------------------------
    #  忠实度评测
    # ------------------------------------------------------------------

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
        retries: int = 2,
    ) -> dict:
        """评估答案是否忠实于上下文。

        参数:
            question: 用户问题
            answer:   LLM 生成的答案
            context:  检索到的上下文（拼接后的文档内容）
            retries:  JSON 解析失败时的重试次数

        返回:
            dict: {"claims": [...], "score": 0.X}
                  解析失败时返回 {"score": 0.0, "error": "parse_failure"}
        """
        chain = FAITHFULNESS_PROMPT | self.llm | StrOutputParser()

        for attempt in range(retries + 1):
            try:
                raw = chain.invoke({
                    "question": question,
                    "context": context,
                    "answer": answer,
                })
                result = _parse_judge_json(raw)

                # 规范化 score
                score = result.get("score", 0.0)
                if isinstance(score, str):
                    score = float(score)
                result["score"] = float(score)

                claims = result.get("claims", [])
                print(f"  [JUDGE] 忠实度: {score:.2f} ({len(claims)} 条声明)")
                return result

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                if attempt < retries:
                    print(f"  [JUDGE] [WARN] 忠实度解析失败 (attempt {attempt + 1}): {e}，重试中...")
                    time.sleep(0.5)
                else:
                    print(f"  [JUDGE] [ERR] 忠实度解析最终失败: {e}")
                    return {"claims": [], "score": 0.0, "error": "parse_failure"}

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str:
                    wait = 2 ** (attempt + 1)  # 1s, 2s, 4s
                    print(f"  [JUDGE] [WARN] API 限流，等待 {wait}s 后重试...")
                    time.sleep(wait)
                    continue
                print(f"  [JUDGE] [ERR] 忠实度评测异常: {e}")
                return {"claims": [], "score": 0.0, "error": str(e)}

        return {"claims": [], "score": 0.0, "error": "max_retries_exceeded"}

    # ------------------------------------------------------------------
    #  答案相关性评测
    # ------------------------------------------------------------------

    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
        retries: int = 2,
    ) -> dict:
        """评估答案与问题的相关性（1-5 分）。

        返回:
            dict: {"score": N, "reasoning": "..."}
        """
        chain = RELEVANCE_PROMPT | self.llm | StrOutputParser()

        for attempt in range(retries + 1):
            try:
                raw = chain.invoke({"question": question, "answer": answer})
                result = _parse_judge_json(raw)

                score = result.get("score", 1)
                if isinstance(score, str):
                    score = int(score)
                result["score"] = int(score)
                result["score_normalized"] = (result["score"] - 1) / 4.0  # 归一化到 [0, 1]

                print(f"  [JUDGE] 相关性: {result['score']}/5")
                return result

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                if attempt < retries:
                    print(f"  [JUDGE] [WARN] 相关性解析失败 (attempt {attempt + 1}): {e}，重试中...")
                    time.sleep(0.5)
                else:
                    print(f"  [JUDGE] [ERR] 相关性解析最终失败: {e}")
                    return {"score": 1, "score_normalized": 0.0, "reasoning": "", "error": "parse_failure"}

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"  [JUDGE] [ERR] 相关性评测异常: {e}")
                return {"score": 1, "score_normalized": 0.0, "reasoning": "", "error": str(e)}

        return {"score": 1, "score_normalized": 0.0, "reasoning": "", "error": "max_retries_exceeded"}

    # ------------------------------------------------------------------
    #  正确性评测
    # ------------------------------------------------------------------

    def evaluate_correctness(
        self,
        question: str,
        answer: str,
        reference: str,
        retries: int = 2,
    ) -> dict:
        """评估答案与参考答案的语义一致性（1-5 分）。

        返回:
            dict: {"score": N, "reasoning": "..."}
        """
        chain = CORRECTNESS_PROMPT | self.llm | StrOutputParser()

        for attempt in range(retries + 1):
            try:
                raw = chain.invoke({
                    "question": question,
                    "answer": answer,
                    "reference": reference,
                })
                result = _parse_judge_json(raw)

                score = result.get("score", 1)
                if isinstance(score, str):
                    score = int(score)
                result["score"] = int(score)
                result["score_normalized"] = (result["score"] - 1) / 4.0  # 归一化到 [0, 1]

                print(f"  [JUDGE] 正确性: {result['score']}/5")
                return result

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                if attempt < retries:
                    print(f"  [JUDGE] [WARN] 正确性解析失败 (attempt {attempt + 1}): {e}，重试中...")
                    time.sleep(0.5)
                else:
                    print(f"  [JUDGE] [ERR] 正确性解析最终失败: {e}")
                    return {"score": 1, "score_normalized": 0.0, "reasoning": "", "error": "parse_failure"}

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str:
                    time.sleep(2 ** (attempt + 1))
                    continue
                print(f"  [JUDGE] [ERR] 正确性评测异常: {e}")
                return {"score": 1, "score_normalized": 0.0, "reasoning": "", "error": str(e)}

        return {"score": 1, "score_normalized": 0.0, "reasoning": "", "error": "max_retries_exceeded"}

    # ------------------------------------------------------------------
    #  记忆提取准确性评测
    # ------------------------------------------------------------------

    def evaluate_memory_extraction(
        self,
        conversation: str,
        extracted_facts: list[str],
        retries: int = 2,
    ) -> dict:
        """评估从对话中提取的 LTM 事实的准确性。"""
        chain = MEMORY_EXTRACTION_ACCURACY_PROMPT | self.llm | StrOutputParser()
        facts_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "（无）"

        for attempt in range(retries + 1):
            try:
                raw = chain.invoke({
                    "conversation": conversation,
                    "extracted_facts": facts_text,
                })
                result = _parse_judge_json(raw)
                accuracy = result.get("overall_accuracy", 0.0)
                if isinstance(accuracy, str):
                    accuracy = float(accuracy)
                result["overall_accuracy"] = float(accuracy)
                evals = result.get("evaluations", [])
                print(f"  [JUDGE] 记忆提取准确性: {accuracy:.2f} ({len(evals)} 条)")
                return result
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                if attempt < retries:
                    print(f"  [JUDGE] [WARN] 记忆提取解析失败 (attempt {attempt + 1}): {e}")
                    time.sleep(0.5)
                else:
                    print(f"  [JUDGE] [ERR] 记忆提取最终失败: {e}")
                    return {"evaluations": [], "overall_accuracy": 0.0, "error": "parse_failure"}
            except Exception as e:
                if "rate" in str(e).lower() or "limit" in str(e).lower():
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"evaluations": [], "overall_accuracy": 0.0, "error": str(e)}
        return {"evaluations": [], "overall_accuracy": 0.0, "error": "max_retries_exceeded"}

    # ------------------------------------------------------------------
    #  记忆相关性评测
    # ------------------------------------------------------------------

    def evaluate_memory_relevance(
        self, query: str, ltm_facts: list[str], retries: int = 2,
    ) -> dict:
        """评估 LTM 事实与查询的相关性。"""
        chain = MEMORY_RELEVANCE_PROMPT | self.llm | StrOutputParser()
        facts_text = "\n".join(f"- {f}" for f in ltm_facts) if ltm_facts else "（无）"
        for attempt in range(retries + 1):
            try:
                raw = chain.invoke({"query": query, "ltm_facts": facts_text})
                result = _parse_judge_json(raw)
                avg = result.get("avg_relevance", 0.0)
                if isinstance(avg, str):
                    avg = float(avg)
                result["avg_relevance"] = float(avg)
                for s in result.get("relevance_scores", []):
                    if "score" in s:
                        s["score_normalized"] = (int(s["score"]) - 1) / 4.0
                print(f"  [JUDGE] 记忆相关性: {result['avg_relevance']:.1f}")
                return result
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                if attempt < retries:
                    print(f"  [JUDGE] [WARN] 记忆相关性解析失败: {e}")
                    time.sleep(0.5)
                else:
                    return {"relevance_scores": [], "avg_relevance": 0.0, "error": "parse_failure"}
            except Exception as e:
                if "rate" in str(e).lower() or "limit" in str(e).lower():
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"relevance_scores": [], "avg_relevance": 0.0, "error": str(e)}
        return {"relevance_scores": [], "avg_relevance": 0.0, "error": "max_retries_exceeded"}

    # ------------------------------------------------------------------
    #  记忆影响评测
    # ------------------------------------------------------------------

    def evaluate_memory_impact(
        self, query: str, answer_with: str, answer_without: str,
        ltm_used: str = "", retries: int = 2,
    ) -> dict:
        """评估长期记忆对答案质量的影响。"""
        chain = MEMORY_IMPACT_PROMPT | self.llm | StrOutputParser()
        for attempt in range(retries + 1):
            try:
                raw = chain.invoke({
                    "query": query, "answer_with": answer_with,
                    "answer_without": answer_without,
                    "ltm_used": ltm_used or "（无）",
                })
                result = _parse_judge_json(raw)
                score = result.get("impact_score", 0)
                if isinstance(score, str):
                    score = int(score)
                result["impact_score"] = int(score)
                result["impact_normalized"] = (int(score) + 1) / 3.0
                print(f"  [JUDGE] 记忆影响: {result.get('impact_label', '?')} (score={score})")
                return result
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                if attempt < retries:
                    print(f"  [JUDGE] [WARN] 记忆影响解析失败: {e}")
                    time.sleep(0.5)
                else:
                    return {"impact_score": 0, "impact_label": "error", "impact_normalized": 0.33, "error": "parse_failure"}
            except Exception as e:
                if "rate" in str(e).lower() or "limit" in str(e).lower():
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"impact_score": 0, "impact_label": "error", "impact_normalized": 0.33, "error": str(e)}
        return {"impact_score": 0, "impact_label": "error", "impact_normalized": 0.33, "error": "max_retries_exceeded"}
