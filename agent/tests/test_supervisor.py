"""
test_supervisor.py — 领导-部门制监督派发器自测（leader-departments supervisor）。

Run: C:/Users/30811/miniconda3/envs/demo/python.exe agent/tests/test_supervisor.py
ponytail: assert-based，无 pytest；无 LLM/API 调用（worker 图全部用确定性假图，
只有 LangGraph 原生组件本身被真跑：checkpointer / interrupt / Command(resume)）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.graph import StateGraph, END  # noqa: E402

import agent.supervisor as sv  # noqa: E402
import agent.task_registry as tr  # noqa: E402
from agent.nodes import after_agent, build_tools_node, route_intent  # noqa: E402
from agent.subagents import _gate_node, request_review  # noqa: E402
from agent.plan import decide_mode  # noqa: E402
from agent.state import AgentState  # noqa: E402

# ---- 共享事件循环（后台 asyncio task 存活需要） ----

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)


def R(coro):
    return _loop.run_until_complete(coro)


# ---- 隔离的临时环境（store db / 假 graph 工件） ----

_TMP = tempfile.mkdtemp(prefix="sv_test_")
TASK_DB = str(Path(_TMP) / "task_store.db")


async def _no_graph(meta):
    return None  # 测试态不重建真 worker 图（避免触发工具装配/网络）


async def _make_saver(db_path):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    conn = await aiosqlite.connect(str(db_path))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver, conn


def _fake_graph(answer="章节正文内容", sleep_s=0.0, emit_event=False, saver=None):
    """确定性假 worker 图：单 agent 节点答固定答案（无工具、无 LLM）。"""
    async def agent_node(state, config):
        if emit_event:
            from agent.stream import emit
            emit({"type": "token", "content": "leak?"})
        if sleep_s:
            await asyncio.sleep(sleep_s)
        return {"messages": [AIMessage(content=answer)]}

    sg = StateGraph(AgentState)
    sg.add_node("agent", agent_node)
    sg.set_entry_point("agent")
    sg.add_edge("agent", END)
    return sg.compile(checkpointer=saver)


def _gate_graph(saver):
    """真 gate 链路：agent 首轮调 request_review → after_agent 路由 gate →
    _gate_node interrupt → resume 后 agent 见 ToolMessage → 终答。"""
    def agent_node(state, config):
        has_resp = any(getattr(m, "type", "") == "tool" for m in state["messages"])
        if has_resp:
            return {"messages": [AIMessage(content="final-after-resume")]}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "request_review", "args": {"question": "确认用参数X?"},
             "id": "g1"}])]}

    sg = StateGraph(AgentState)
    sg.add_node("agent", agent_node)
    sg.add_node("tools", build_tools_node([request_review]))
    sg.add_node("gate", _gate_node)
    sg.set_entry_point("agent")
    sg.add_conditional_edges(
        "agent", after_agent,
        {"tools": "tools", "synthesize": END, "end": END, "gate": "gate"})
    sg.add_edge("tools", "agent")
    sg.add_edge("gate", "agent")
    return sg.compile(checkpointer=saver)


def _setup():
    sv.TASK_STORE_DB = Path(TASK_DB)
    sv._store = None
    sv._store_lock = None
    sv._running.clear()
    sv._graphs.clear()
    sv._graph_cache.clear()
    sv._graph_for_meta = _no_graph  # 测试态：禁止重建真图
    sv.init_supervisor(None)


async def wait_status(task_id, expect, timeout=6.0):
    deadline = asyncio.get_event_loop().time() + timeout
    last = {}
    while asyncio.get_event_loop().time() < deadline:
        last = await sv.progress(task_id)
        if last.get("status") == expect:
            return last
        await asyncio.sleep(0.03)
    raise AssertionError(f"task {task_id} not {expect}; last={last}")


# ---- tests ----

def test_route_intent_task_query():
    assert route_intent({"intent": "task_query", "confidence": 0.9}) == "task"
    assert route_intent({"intent": "literature_search", "confidence": 0.9}) == "resolve"
    assert route_intent({"intent": "general_chat", "confidence": 0.9}) == "chat"
    assert route_intent({"intent": "literature_search", "confidence": 0.3}) == "clarify"


def test_decide_mode_async_creation():
    async_q = {"domain": "creation",
               "messages": [HumanMessage(content="帮我后台写一篇 RMNet 的综述")]}
    sync_q = {"domain": "creation",
              "messages": [HumanMessage(content="帮我写一篇 RMNet 的综述")]}
    assert decide_mode(async_q) == "react", "explicit async writing → react (dispatch)"
    assert decide_mode(sync_q) == "plan", "default writing → plan (unchanged)"
    assert decide_mode({"domain": "coding", "messages": []}) == "plan"


def test_dispatch_progress_collect():
    _setup()
    sv._worker_graph = lambda role, gate: (_fake_graph(answer="唯一章节正文"), {
        "subagent_system": "t", "bound_tools": [], "max_steps": 3})
    task_id = R(sv.dispatch("creator", "写第1章", "写 Introduction 章节，自包含。",
                            parent_thread="th_test"))
    assert task_id and len(task_id) == 10
    card = R(wait_status(task_id, "done"))
    assert card.get("title") == "写第1章"
    assert R(sv.collect(task_id)) == "唯一章节正文"
    listing = R(sv.list_tasks())
    assert any(t["task_id"] == task_id for t in listing)


def test_cancel():
    _setup()
    sv._worker_graph = lambda role, gate: (_fake_graph(sleep_s=3.0), {
        "subagent_system": "t", "bound_tools": [], "max_steps": 3})
    task_id = R(sv.dispatch("coder", "慢任务", "sleep"))
    R(asyncio.sleep(0.1))
    R(sv.cancel(task_id))
    assert R(sv.progress(task_id)).get("status") == "cancelled"


def test_interrupt_resume():
    _setup()
    db = Path(_TMP) / "gate.db"
    saver, conn = R(_make_saver(db))
    try:
        g = _gate_graph(saver)
        sv._worker_graph = lambda role, gate: (g, {
            "subagent_system": "t", "bound_tools": ["request_review"],
            "max_steps": 3})
        task_id = R(sv.dispatch("creator", "需确认", "需要领导确认参数"))
        card = R(wait_status(task_id, "interrupted"))
        assert card.get("interrupted") is True
        assert "参数" in str(card.get("question", "")), card
        R(sv.resume(task_id, "同意，用参数X"))
        final = R(wait_status(task_id, "done"))
        assert R(sv.collect(task_id)) == "final-after-resume"
    finally:
        R(conn.close())


def test_event_isolation():
    _setup()
    sv._worker_graph = lambda role, gate: (_fake_graph(emit_event=True), {
        "subagent_system": "t", "bound_tools": [], "max_steps": 3})
    from agent.stream import set_event_queue, reset_event_queue
    q = asyncio.Queue()
    tok = set_event_queue(q)
    try:
        task_id = R(sv.dispatch("creator", "事件隔离", "emit"))
        R(wait_status(task_id, "done"))
        assert q.empty(), "后台 worker 事件泄漏进回合队列"
    finally:
        reset_event_queue(tok)


def test_registry_find():
    _setup()
    sv._worker_graph = lambda role, gate: (_fake_graph(), {
        "subagent_system": "t", "bound_tools": [], "max_steps": 3})
    R(sv._meta_put("a1b2c3d4e5", role="creator",
                   title="写综述｜跨模态综述", status="done", created_at="2026-09-01T10:00:00"))
    hit = R(tr.find_tasks("跨模态"))
    assert any(t["task_id"] == "a1b2c3d4e5" for t in hit), hit
    id_hit = R(tr.find_tasks("a1b2c3d4e5"))
    assert any(t["task_id"] == "a1b2c3d4e5" for t in id_hit)
    kind_hit = R(tr.find_tasks("写作"))
    assert any(t["kind"] == "creator" for t in kind_hit)


def test_provider_envelope():
    _setup()
    from agent.providers.task_provider import TaskProvider
    import json as _json
    prov = TaskProvider()
    raw = R(prov.call_tool("task_list", {}))
    data = _json.loads(raw)
    assert data.get("ok") is True, raw
    assert isinstance(data.get("data", {}).get("tasks", []), list)
    bad = R(prov.call_tool("task_progress", {"task_id": "nope99"}))
    assert _json.loads(bad).get("ok") is False


# ---- runner ----

def _teardown():
    """关掉 store 连接并取消残留后台任务，保证进程干净退出。"""
    for t in list(sv._running.values()):
        t.cancel()

    async def _close():
        # 注意：AsyncSqliteStore.aclose() 在该环境会挂起（与 __del__ 同类兼容
        # 问题）——直接关 aiosqlite 连接即可结束线程。
        if sv._store_conn is not None:
            try:
                await sv._store_conn.close()
            except Exception:
                pass

    try:
        _loop.run_until_complete(_close())
    except Exception:
        pass


def _main():
    try:
        for fn in [
            test_route_intent_task_query,
            test_decide_mode_async_creation,
            test_dispatch_progress_collect,
            test_cancel,
            test_interrupt_resume,
            test_event_isolation,
            test_registry_find,
            test_provider_envelope,
        ]:
            fn()
            print(f"  [OK] {fn.__name__}")
        print("supervisor self-check OK")
    finally:
        _teardown()


if __name__ == "__main__":
    _main()