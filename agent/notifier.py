"""
notifier.py — 后台任务完成通知生成器。

把一条后台任务状态（dict）通过一次聚焦的 LLM 调用转成 1~2 句面向用户的通知。
遵循 CLAUDE.md Prompt 设计原则：
  1. 结构化输出 — 只输出通知文本，无 preamble
  2. 零状态 — 任务事实全部内联注入 prompt，不依赖对话历史
  3. 模型无关 — 模型名由 _get_model("LLM_MODEL") 注入，prompt 不写模型名
  4. 上下文感知 — 任务状态/结果/错误作为前置约束注入，禁止臆造

由 web/api/routers/background.py 的 /api/agent/notify/stream 消费：前端检测到
任务完成（notify=true）时触发一次 notify 回合，让 assistant 主动告知用户。
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, HumanMessage

from .nodes import _get_model, _stream_llm

_NOTIFY_SYSTEM = """\
You are a background-task notifier for a research assistant. A job the user asked for just ended; write the completion notice to the user.

Task facts (report ONLY these facts, never invent extra):
- name: {paper_name}
- type: {kind}
- status: {status}
- progress: {progress}
- error: {error}
- result: {result}

Rules:
- status "done" → confirm completion with the concrete outcome from `result` (e.g. "已入库，可以在知识库中检索到。").
- status "failed" → state the failure and the reason from `error`; suggest retrying when it looks transient.
- any other status → reassure briefly; do NOT claim completion.
- Do NOT fabricate actions or outcomes not present in the facts.
- Output ONLY 1-2 sentences, plain text. No markdown, no bullet lists, no code, no tool names.
- Match the language of the user's messages: Chinese → Chinese; English → English.

Output ONLY the message."""


async def stream_task_notify(task: dict) -> str:
    """把任务状态流式转成通知文本（经当前请求的 emit() 事件通道逐 token 发出）。

    Notify 端点会先 set_event_queue()，_stream_llm 的 token 由此透出。返回合并文本。
    """
    result = task.get("result")
    if isinstance(result, dict):
        result = json.dumps(result, ensure_ascii=False)
    elif result is None:
        result = ""

    prompt = _NOTIFY_SYSTEM.format(
        paper_name=task.get("paper_name") or task.get("task_id") or "任务",
        kind=task.get("kind") or "通用",
        status=task.get("status") or "pending",
        progress=task.get("progress") or "",
        error=task.get("error") or "无",
        result=str(result),
    )

    model = _get_model({"configurable": {}})
    msg = await _stream_llm(model, [
        SystemMessage(content=prompt),
        HumanMessage(content="Notify the user now."),
    ])
    text = getattr(msg, "content", "") or ""
    return str(text).strip()