"""
context.py — 对话级工作区上下文节点（对话中心化重构 L1）。

context_node 每回合确定性重建 state.context，为父 agent 与子 agent 注入
「这场对话正在哪个文档 / 哪个实验项目 / 哪个研究主题上」的会话级记忆：

    active_doc_id      — 最近一次文档工具（doc_create/set_outline/write_section）
                          调用的 doc_id；无则回退 state.doc_id（plan_node 持久化）
    active_project     — 最近一次 run_experiment/set_experiment_project/
                          experiment_project_state 结果里的 project；supervisor
                          派发路径经 task_collect 的 EXP:/PROJECT: footer 兜底
    study_topic        — 最近一次 study_context/study_add_hypothesis 调用的 topic
    recent_experiments — 本会话最近运行的 exp_id（新→旧，cap RECENT_CAP）

设计约束：
- 零状态/可幂等：同一条消息序列 → 相同 context。数据源只有 state 本身 + 确定性
  读取（exp 状态文件），不依赖 LLM 或隐藏通道。
- 子 agent 保持零状态：context 由父层节点推导，经 _subagent_task / system
  prompt 注入子任务字符串（见 plan.py / nodes.py）。
- supervisor 派发路径：worker 跑在 thread_id=task_id，其工具调用不进父消息
  列表；但 worker 终态总结（task_collect 的 output）携带 CODER_SYSTEM 约定的
  `PROJECT:` / `EXP:` footer，父层据此确定性回填。
"""
from __future__ import annotations

import re

from .state import AgentState

# 近期实验保留条数
RECENT_CAP = 5

# ---- 确定性 footer 解析（CODER_SYSTEM 约定，机器可解析） ----

_PROJECT_RE = re.compile(r"\bPROJECT:\s*([^\s\[\]]+)", re.IGNORECASE)
_EXP_RE = re.compile(r"\bEXP:\s*([0-9a-fA-F-]{6,40})", re.IGNORECASE)


def _footer_project(text: str) -> str | None:
    m = _PROJECT_RE.search(text or "")
    return m.group(1).strip() if m else None


def _footer_exps(text: str) -> list[str]:
    return [m.group(1) for m in _EXP_RE.finditer(text or "")]


# ---- 工具结果解析 ----

def _parse_tool_result(content) -> tuple[dict | None, str]:
    """返回 (data, raw)。结构化信封 → data；否则 raw 原文。"""
    try:
        from .tool_contract import parse_tool_result
        parsed = parse_tool_result(str(content))
        if parsed.is_envelope and parsed.ok and isinstance(parsed.data, dict):
            return parsed.data, ""
    except Exception:
        pass
    return None, str(content)


# ---- 文档工具名（args 扫描） ----

_DOC_TOOLS = ("doc_create", "doc_set_outline", "doc_write_section")
_STUDY_TOOLS = ("study_context", "study_add_hypothesis")
_PROJECT_RESULT_TOOLS = ("run_experiment", "set_experiment_project",
                         "experiment_project_state")
_EXP_result_TOOLS = ("run_experiment", "experiment_status")


async def context_node(state: AgentState, config=None) -> dict:
    """重建对话级 context。只读 state + 确定性文件读取，纯函数式返回更新。"""
    msgs = state.get("messages", []) or []
    context = {
        "active_doc_id": state.get("doc_id") or None,
        "active_project": None,
        "study_topic": None,
        "recent_experiments": [],
    }
    seen_exps: set[str] = set()

    # 逆序扫描：最后的调用为准（last-wins）。
    for msg in reversed(msgs):
        m_type = getattr(msg, "type", "")
        if m_type == "tool":
            name = getattr(msg, "name", "") or ""
            data, raw = _parse_tool_result(getattr(msg, "content", ""))
            if name in _PROJECT_RESULT_TOOLS:
                proj = (data or {}).get("project") if isinstance(data, dict) else None
                if proj and not context["active_project"]:
                    context["active_project"] = str(proj)
            if name in _EXP_result_TOOLS:
                exp_id = (data or {}).get("exp_id") if isinstance(data, dict) else None
                if exp_id and exp_id not in seen_exps:
                    seen_exps.add(str(exp_id))
                    context["recent_experiments"].append(str(exp_id))
            if name == "task_collect":
                out = (data or {}).get("output") if isinstance(data, dict) else raw
                if isinstance(out, str):
                    proj = _footer_project(out)
                    if proj and not context["active_project"]:
                        context["active_project"] = proj
                    for e in _footer_exps(out):
                        if e not in seen_exps:
                            seen_exps.add(e)
                            context["recent_experiments"].append(e)
        elif m_type == "ai":
            for tc in getattr(msg, "tool_calls", None) or []:
                name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                if name in _DOC_TOOLS:
                    doc_id = args.get("doc_id")
                    if doc_id and not context["active_doc_id"]:
                        context["active_doc_id"] = str(doc_id)
                elif name in _STUDY_TOOLS:
                    topic = args.get("topic")
                    if topic and not context["study_topic"]:
                        context["study_topic"] = str(topic)
                elif name == "run_experiment":
                    proj = args.get("project")
                    if proj and not context["active_project"]:
                        context["active_project"] = str(proj)

    # active_doc_id 无消息证据时回退 state.doc_id 已处理（初始化时即带上）。
    # recent_experiments 截断。
    context["recent_experiments"] = context["recent_experiments"][:RECENT_CAP]

    # 兜底：仅拿到 exp_id 而缺 project 时，从 exp 状态文件确定性解析 project
    # （supervisor 派发的 coder 跑的实验也归属其项目）。
    if not context["active_project"] and context["recent_experiments"]:
        try:
            from .domains.coding import _load_exp
            for e in context["recent_experiments"]:
                st = _load_exp(e)
                if st and st.get("project"):
                    context["active_project"] = str(st["project"])
                    break
        except Exception:
            pass

    return {"context": context}