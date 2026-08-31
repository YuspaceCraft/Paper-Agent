"""
技能名称 tools.py — 自定义工具

编写 LangChain @tool 装饰的工具函数，Agent 可以直接调用。

工具编写规则:
  1. 使用 @tool 装饰器
  2. 函数名 = 工具名（使用 snake_case）
  3. docstring = 工具描述（Agent 用来理解工具用途）
  4. 参数使用类型注解（str, int, float, bool 等）
  5. 返回结构化字符串（[ERR] / [WARN] / [INFO] 前缀）
  6. 不要抛出异常，用返回字符串描述错误

可用的共享资源（通过 agent.tools._ctx 访问）:
  - _ctx.vector_store: Chroma 向量库
  - _ctx.reranker:    BGE Cross-Encoder 精排器
  - _ctx.llm:         LLM 模型实例
  - _ctx.embeddings:  嵌入模型
  - _ctx.memory:      对话记忆对象

示例:
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def example_tool(query: str) -> str:
    """
    工具的简短描述（Agent 用来理解此工具的用途）。

    适用场景:
      - 场景A
      - 场景B

    :param query: 输入参数说明
    :return: 工具执行结果
    """
    # 访问共享资源
    from agent.tools import _ctx

    if _ctx.vector_store is None:
        return "[ERR] 向量数据库未初始化"

    # ... 你的逻辑 ...

    return f"✅ 示例结果: {query}"


# ============================================================
# 在 __all__ 中列出所有需要暴露的工具函数名
# ============================================================
__all__ = [
    "example_tool",
]
