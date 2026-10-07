"""
trace_wrap.py — LLM 调用的计时 + 真实 token usage 采集（非流式路径）。

插桩策略（M2）：
- 流式路径统一在 agent/nodes.py `_stream_llm` 的外包 wrapper 里采集；
- 非流式路径（understand 结构化输出 / _ask_for_plan / _verify_goal /
  _subagent_synthesize / memory 摘要）替换为 traced_ainvoke 一行调用。

usage_of() 归一化三种形态：response_metadata.token_usage（OpenAI 风格）、
usage_metadata（langchain 风格 input/output/total）。取不到就返回 {}，
emit 端与调用点整体 try/except，任何失败都不影响原 LLM 返回。
"""

from __future__ import annotations

import time
from typing import Any

from .events import emit_llm_call


def usage_of(response: Any) -> dict:
    """从响应提取 {prompt_tokens, completion_tokens, total_tokens}。

    真实优先：response_metadata.token_usage（OpenAI 风格）→ usage_metadata
    （langchain 风格 input/output/total）。
    """
    if response is None:
        return {}
    try:
        meta = getattr(response, "response_metadata", None)
        meta = meta if isinstance(meta, dict) else {}
        tu = meta.get("token_usage") or {}
        if isinstance(tu, dict):
            out = {
                "prompt_tokens": tu.get("prompt_tokens"),
                "completion_tokens": tu.get("completion_tokens"),
                "total_tokens": tu.get("total_tokens"),
            }
            if any(v is not None for v in out.values()):
                return {k: (int(v) if v is not None else 0) for k, v in out.items()}

        um = getattr(response, "usage_metadata", None)
        if isinstance(um, dict) and um:
            return {
                "prompt_tokens": int(um.get("input_tokens") or 0),
                "completion_tokens": int(um.get("output_tokens") or 0),
                "total_tokens": int(um.get("total_tokens") or 0),
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


def usage_or_estimate(response: Any, messages=None) -> dict:
    """真实 usage 优先；provider（如 DashScope 兼容模式流式）拿不到时按字符估算，
    并标记 estimated 供报告降权。"""
    u = usage_of(response)
    if u:
        return u
    try:
        prompt_chars = 0
        for m in (messages or ()):
            c = getattr(m, "content", m)
            if isinstance(c, str):
                prompt_chars += len(c)
        out_chars = len(str(getattr(response, "content", "") or ""))
        return {
            "prompt_tokens": max(1, round(prompt_chars / 4)),
            "completion_tokens": max(1, round(out_chars * 0.32)),
            "total_tokens": max(2, round((prompt_chars + out_chars * 1.2) / 4)),
            "estimated": True,
        }
    except Exception:  # noqa: BLE001
        return {}


def _model_name(runnable: Any) -> str:
    try:
        return str(getattr(runnable, "model_name", None) or
                   getattr(runnable, "model", "") or "")
    except Exception:  # noqa: BLE001
        return ""


async def traced_ainvoke(runnable, messages, *, node: str = "", config=None) -> Any:
    """ainvoke + 计时 + usage（真实优先/估算兜底），原样返回；失败仍记录后抛。

    config：父节点 runnable config。传入后嵌套调用并进当前 LangSmith run 树
    （否则为独立根 run），是 P1 链路嵌套的关键。
    """
    from agent.core.token_budget import fit_for_context

    fit = fit_for_context(list(messages))
    messages = fit.messages
    if fit.changed:
        try:
            from agent.observability import log_event

            log_event("context_hard_budget", node=node or "llm",
                      **fit.trace_view())
        except Exception:  # noqa: BLE001
            pass
    t0 = time.perf_counter()
    kw = {} if config is None else {"config": config}
    try:
        resp = await runnable.ainvoke(messages, **kw)
        emit_llm_call(model=_model_name(runnable),
                      duration_ms=(time.perf_counter() - t0) * 1000,
                      mode="invoke", node=node,
                      tokens=usage_or_estimate(resp, messages))
        return resp
    except Exception as exc:
        emit_llm_call(model=_model_name(runnable),
                      duration_ms=(time.perf_counter() - t0) * 1000,
                      mode="invoke", node=node,
                      error=f"{type(exc).__name__}: {exc}")
        raise
