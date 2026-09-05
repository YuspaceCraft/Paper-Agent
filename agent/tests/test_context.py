"""(对话中心化重构 L1) self-check — context_node conversation workspace binding.

Run: python agent/tests/test_context.py
ponytail: assert-based, no framework, no LLM calls. Uses langchain message
objects to feed state["messages"], workspace roots isolated to a temp dir.
"""
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from agent.context import context_node  # noqa: E402
from agent import workspace_config as wc  # noqa: E402


def _setup(tmp_root: Path):
    wc.set_override("experiments_path", tmp_root / "experiments")
    wc.set_override("study_root", tmp_root / "studies")


def _tool_result(name: str, data: dict) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"ok": True, "data": data}, ensure_ascii=False),
        tool_call_id=f"call-{name}", name=name,
    )


def _ai_tool(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"},
    ])


def R(coro):
    return asyncio.run(coro)


def test_derive_from_direct_tool_results():
    msgs = [
        HumanMessage(content="复现一下并跑实验"),
        _ai_tool("run_experiment", {"project": "RMNet-repro", "command": "python train.py"}),
        _tool_result("run_experiment", {"exp_id": "aaa111", "status": "running", "project": "RMNet-repro"}),
        _ai_tool("study_context", {"topic": "RMNet"}),
        _tool_result("study_context", {"topic": "RMNet", "recent_experiments": []}),
    ]
    out = R(context_node({"messages": msgs, "doc_id": None}))
    ctx = out["context"]
    assert ctx["active_project"] == "RMNet-repro", ctx
    assert ctx["study_topic"] == "RMNet", ctx
    assert "aaa111" in ctx["recent_experiments"], ctx


def test_derive_doc_and_cap():
    msgs = [
        HumanMessage(content="写一篇综述"),
        _ai_tool("doc_create", {"title": "RMNet Survey"}),
        _tool_result("doc_create", {"doc_id": "doc_abc", "title": "RMNet Survey"}),
        _ai_tool("doc_write_section", {"doc_id": "doc_abc", "section_id": "sec1", "content": "x"}),
        _tool_result("doc_write_section", {"doc_id": "doc_abc", "section_id": "sec1", "status": "done"}),
        # 6 个实验 → cap 5，新→旧取前 5
    ]
    msgs += [
        _tool_result("run_experiment", {"exp_id": f"e{i}", "status": "running", "project": "P"})
        for i in range(6)
    ]  # 构造 6 条 ToolMessage（各自 tool_call_id 冲突无妨，解析只看 content）
    out = R(context_node({"messages": msgs, "doc_id": None}))
    ctx = out["context"]
    assert ctx["active_doc_id"] == "doc_abc", ctx
    assert len(ctx["recent_experiments"]) == 5, ctx
    # 逆序 → 最新的 e5 在首位
    assert ctx["recent_experiments"][0] == "e5", ctx


def test_doc_id_state_fallback():
    """无 doc 工具消息时回退 state.doc_id。"""
    msgs = [HumanMessage(content="继续")]
    out = R(context_node({"messages": msgs, "doc_id": "doc_persisted"}))
    assert out["context"]["active_doc_id"] == "doc_persisted"


def test_coder_footer_and_exp_fallback(tmp_root):
    """supervisor 派发路径：task_collect 的 PROJECT:/EXP: footer + exp 状态文件兜底。"""
    _setup(tmp_root)
    # 造一个 exp 状态文件（确定性读取 project）
    from agent.domains import coding
    coding._save_state({
        "exp_id": "beef99", "project": "RMNet-repro", "name": "r1",
        "command": "python train.py", "status": "done", "exit_code": 0,
        "metrics": {"acc": 0.9}, "created_at": "", "finished_at": "",
    })
    summary = ("Ran r1: acc 0.84→0.9\n"
               "PROJECT: RMNet-repro\nEXP: beef99\nEXP: deadbe\n")
    msgs = [
        HumanMessage(content="复现并优化 RMNet"),
        _ai_tool("task_dispatch", {"role": "coder", "title": "RMNet repro", "task": "..."}),
        _tool_result("task_collect", {"task_id": "t1", "status": "done", "output": summary}),
    ]
    out = R(context_node({"messages": msgs, "doc_id": None}))
    ctx = out["context"]
    assert ctx["active_project"] == "RMNet-repro", ctx
    assert ctx["recent_experiments"][0] == "beef99", ctx
    assert "deadbe" in ctx["recent_experiments"], ctx


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp())
    _setup(root)
    try:
        test_derive_from_direct_tool_results()
        test_derive_doc_and_cap()
        test_doc_id_state_fallback()
        test_coder_footer_and_exp_fallback(root)
    finally:
        wc.clear_overrides()
        shutil.rmtree(root, ignore_errors=True)
    print("context_node self-check OK")