"""
generator.py — RAG 生成链
===========
使用 LangChain Expression Language (LCEL) 构建 RAG 生成链。

RAG 链的工作流程：
  1. 用户提问 → 检索器找到相关文档
  2. 将相关文档格式化为"上下文"
  3. 将上下文+问题填入提示模板
  4. 将完整提示发送给 LLM
  5. LLM 基于上下文生成答案
  6. 解析并返回纯文本答案

LCEL 是什么？
  LangChain Expression Language，用管道符 | 串联处理步骤。
  每一个 | 意味着"把左边产出的数据传给右边处理"。
  类似于 Unix 管道 (ls | grep txt)，但传递的是结构化数据。
"""

from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, OPENAI_API_KEY, DASHSCOPE_BASE_URL
from agent.retriever import format_retrieved_docs


# ============================================================
# RAG 提示模板
# ============================================================
# {context}：检索到的文档内容（由检索器自动填入）
# {question}：用户的问题（由 RunnablePassthrough() 原样转发）
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
你是一个乐于助人的科研文献助手。请使用下面提供的上下文来回答用户的问题。

规则：
1. 只使用上下文中提供的信息来回答问题
2. 如果上下文中没有相关答案，请诚实地说"根据提供的文档，我无法回答这个问题"
3. 回答时尽量引用上下文中的具体信息
4. 保持回答简洁、准确

上下文：
{context}"""),
    ("human", "{question}"),
])


RAG_PROMPT_WITH_HISTORY = ChatPromptTemplate.from_messages([
    ("system", """\
你是一个乐于助人的科研文献助手。请使用下面提供的上下文来回答用户的问题。

规则：
1. 优先使用检索到的上下文来回答问题
2. 如果上下文中没有相关信息，可以参考对话历史中的讨论
3. 回答时引用上下文中的具体信息
4. 考虑对话历史中已讨论过的内容，避免重复回答相同问题
5. 保持回答简洁、准确

检索到的上下文：
{context}

对话历史：
{history}"""),
    ("human", "{question}"),
])


def create_llm() -> ChatOpenAI:
    """
    创建 LLM 实例。

    temperature=0 意味着模型会给出最确定、最稳定的回答，
    不会进行随机创造。这对 RAG 很重要——
    我们希望模型忠实于检索到的文档，而不是自由发挥。
    """
    print(f"[GENERATE] [SETUP] 初始化 LLM: {LLM_MODEL} (temperature=0)")
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )


def create_rag_chain(retriever, llm: ChatOpenAI):
    """
    构建 RAG 生成链（LCEL 管道）。

    LCEL 管道解析（从左到右读）：
      {
        "context": retriever | format_retrieved_docs,  ← 检索+格式化 → 填入 context
        "question": RunnablePassthrough()              ← 原样转发用户问题 → 填入 question
      }
      | RAG_PROMPT     ← 将 context 和 question 填入提示模板
      | llm            ← 将完整的提示发送给 LLM
      | StrOutputParser() ← 从 LLM 的响应中提取纯文本

    参数:
        retriever: 检索器（向量数据库的包装）
        llm:       ChatOpenAI 实例

    返回:
        Runnable: 一个可调用的 RAG 链（用法: chain.invoke("你的问题")）
    """
    chain = (
        {
            "context": retriever | format_retrieved_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    print("[GENERATE] [OK] RAG 生成链已就绪")
    return chain


def create_rag_chain_with_history(retriever, llm: ChatOpenAI):
    """
    构建带对话历史的 RAG 生成链（LCEL 管道）。

    与 create_rag_chain 的区别：
      - 输入是一个 dict: {"question": str, "history": str}
      - 使用 itemgetter 从 dict 中提取 question 字符串送给检索器
      - history 直接注入提示模板的 {history} 变量

    LCEL 管道解析：
      {
        "context":  itemgetter("question") | retriever | format_retrieved_docs,
        "question": itemgetter("question"),
        "history":  itemgetter("history"),
      }
      | RAG_PROMPT_WITH_HISTORY
      | llm
      | StrOutputParser()

    参数:
        retriever: 检索器
        llm:       ChatOpenAI 实例

    返回:
        Runnable: 可调用的带历史 RAG 链
    """
    chain = (
        {
            "context": itemgetter("question") | retriever | format_retrieved_docs,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | RAG_PROMPT_WITH_HISTORY
        | llm
        | StrOutputParser()
    )
    print("[GENERATE] [OK] RAG 生成链（带历史）已就绪")
    return chain


# ============================================================
#  Hybrid RAG — 科研文献助手（长期+短期记忆）
# ============================================================

RAG_PROMPT_HYBRID = ChatPromptTemplate.from_messages([
    ("system", """\
你是专业的科研文献分析助手。你的核心能力是精准理解用户意图，通过检索获取高质量文献证据，并在有限的上下文窗口内维持长程对话的连贯性与准确性。你绝不编造文献内容，所有事实性陈述必须有检索依据。

核心规则：
1. **引用标注**：所有事实陈述必须标注来源，格式为 (Author, Year) 或文件名
2. **不确定性声明**：当检索结果不足以回答问题时，明确说"现有文献未提及该细节"
3. **结构化输出**：复杂回答使用标题、列表、表格呈现。对比类问题必须使用表格
4. **基于证据**：只使用检索到的上下文和长期记忆中的信息来回答问题
5. **上下文优先**：注意对话摘要中的已讨论内容，避免重复

检索到的文献上下文：
{context}

对话摘要：
{summary}

近期对话历史：
{history}

相关长期记忆：
{long_term_memory}

当前分析状态：
{state}"""),
    ("human", "{question}"),
])


def create_hybrid_rag_chain(retriever, llm: ChatOpenAI):
    """
    构建混合 RAG 生成链（长期记忆 + 短期记忆 + 检索）。

    LCEL 管道解析：
      {
        "context": itemgetter("question") | retriever | format_retrieved_docs,
        "question": itemgetter("question"),
        "summary": itemgetter("summary"),
        "history": itemgetter("history"),
        "long_term_memory": itemgetter("long_term_memory"),
        "state": itemgetter("state"),
      }
      | RAG_PROMPT_HYBRID
      | llm
      | StrOutputParser()

    参数:
        retriever: 检索器
        llm:       ChatOpenAI 实例

    返回:
        Runnable: 可调用的混合 RAG 链
    """
    chain = (
        {
            "context": itemgetter("question") | retriever | format_retrieved_docs,
            "question": itemgetter("question"),
            "summary": itemgetter("summary"),
            "history": itemgetter("history"),
            "long_term_memory": itemgetter("long_term_memory"),
            "state": itemgetter("state"),
        }
        | RAG_PROMPT_HYBRID
        | llm
        | StrOutputParser()
    )
    print("[GENERATE] [OK] 混合 RAG 生成链（长期+短期记忆）已就绪")
    return chain
