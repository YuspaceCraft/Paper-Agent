"""
tool_contract.py — 统一工具输出信封契约（INFO_FLOW_REVIEW P6 收敛）。

历史问题：库工具（builtin / creation / coding）成功与失败都返回 JSON envelope
（`{"ok": true/false, ...}`），而通用文件/数学工具（read_file、list_dir、
get_time、calculator、fetch_url）成功路径返回纯文本。于是 _salvage_tool_content、
_classify_tool_error、plan._ingest_guard 各自维护一份「双格式」试探解析 ——
模型上下文里同一批工具结果的形态不稳定，错误恢复字段名也各处硬编码。

本模块成为唯一权威：
- `ok()` / `err()` 生成统一 envelope（所有 provider 共用，输出形状只在此定一次）；
- `parse_tool_result()` 唯一解析入口 —— 先试 envelope，非 envelope 一律按纯文本分流；
- `truncate_tool_result()` 截断时尽量保持 envelope 可解析（原理：字符级截断把
  JSON 切断后整个 envelope 作废并回退成纯文本，fetch_content 的章节内容会丢）。

格式契约（同步各 provider 模块 docstring）：
- 结构化 / 库 / 错误 → JSON envelope
    {"ok": true,  "data": {...}}                                      （成功）
    {"ok": false, "error": "...", "error_type": "...", "next": "..."} （失败）
- 纯文本工具（read_file / list_dir / get_time / calculator / fetch_url 成功路径）
  → 保证「不是合法 envelope」的 UTF-8 文本；parse_tool_result 据此确定性分流。
  例外：文件内容恰好是带 "ok" 键的合法 envelope JSON 时按 envelope 处理，属文档化边界。

名称说明：`next` 只作 dict key，与内置函数名无冲突。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ---- 生成端：统一信封 ----

def ok(data: Any = None) -> str:
    """成功 envelope。data 为结构化负载（list/dict）；纯文本工具勿用本函数。"""
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False, indent=2)


def err(error_type: str = "unknown", detail: str = "", next_action: str = "",
        **ctx) -> str:
    """失败 envelope。ctx 可携带附加字段（available_papers / available_sections 等）。"""
    payload: dict = {
        "ok": False, "error": detail, "next": next_action,
        "error_type": error_type,
    }
    payload.update(ctx)
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---- 解析端：唯一入口 ----

@dataclass
class ToolResult:
    """解析后的统一视图。is_envelope 为 False ⇒ 纯文本成功结果（text 生效）。"""
    is_envelope: bool = False
    ok: bool = True
    data: Any = None               # 结构化负载（envelope 成功 / 失败时的 data 字段）
    text: str = ""                 # 纯文本结果（非 envelope）
    error: str = ""                # 失败详情
    error_type: str = ""           # param_error / transient / not_found / ...
    next_action: str = ""          # 建议的恢复动作
    extra: dict = field(default_factory=dict)  # envelope 的其余字段


def parse_tool_result(content) -> ToolResult:
    """唯一解析入口：先试 JSON envelope，失败按纯文本成功结果处理。"""
    raw = str(content)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        data = None

    if isinstance(data, dict) and "ok" in data:
        r = ToolResult(is_envelope=True, ok=bool(data.get("ok")))
        r.data = data.get("data")
        if not r.ok:
            r.error = str(data.get("error", ""))
            r.error_type = str(data.get("error_type", ""))
            r.next_action = str(data.get("next", ""))
        r.extra = {k: v for k, v in data.items()
                   if k not in ("ok", "data", "error", "error_type", "next")}
        return r
    # 非 envelope（含恰好是合法 JSON 但无 "ok" 键的对象）→ 纯文本
    return ToolResult(is_envelope=False, ok=True, text=raw)


# ---- 截断（对 envelope 保解析性） ----

def truncate_tool_result(text: str, limit: int) -> str:
    """字符级截断；原先是合法 envelope 时截在 data 内部，保持可解析。"""
    if len(text) <= limit:
        return text
    head = text[:limit]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and "ok" in data:
        return _truncate_envelope(data, len(text), limit)
    return head + f"\n…[truncated: result > {limit} chars]"


def _truncate_envelope(data: dict, total: int, limit: int) -> str:
    payload = data.get("data")
    budget = max(64, limit - 128)
    if isinstance(payload, str):
        data["data"] = (payload[:budget]
                        + ("" if len(payload) <= budget
                           else f"\n…[truncated: {len(payload)} chars]"))
    elif isinstance(payload, list):
        out: list = []
        used = 0
        for item in payload:
            enc = json.dumps(item, ensure_ascii=False)
            if used + len(enc) > budget:
                out.append(f"…[{len(payload) - len(out)} items omitted]")
                break
            out.append(item)
            used += len(enc) + 1
        data["data"] = out
    else:
        data["data"] = {"_truncated": True,
                        "note": f"payload too large ({total} chars) for the envelope"}

    serialized = json.dumps(data, ensure_ascii=False)
    if len(serialized) <= limit:
        return serialized
    # 序列化后仍超限（转义膨胀）→ 兜底为文本摘要信封
    return json.dumps({
        "ok": data.get("ok", True),
        "data": {"_truncated": True, "note": f"envelope dropped for size ({total} chars)"},
    }, ensure_ascii=False)[:limit]