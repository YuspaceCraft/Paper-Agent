"""(v10 / Phase A) self-check — creation domain: route_domain heuristics + DocStore.

Run: python agent/tests/test_creation.py
ponytail: assert-based, no framework, no LLM/backend calls.
"""
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent import plan as plan_mod  # noqa: E402
from agent.domains import creation  # noqa: E402
from agent.nodes import route_domain  # noqa: E402


class _Msg:
    type = "human"

    def __init__(self, content):
        self.content = content


def _state(query="", domain="paper"):
    return {"messages": [_Msg(query)], "domain": domain}


# ---- domain routing ----

def test_route_domain_keyword_overrides_llm():
    # 强信号命中 → rule 覆盖 LLM label
    assert route_domain(_state(query="写一篇 RMNet 和 SRN 的对比综述", domain="paper")) == "creation"
    assert route_domain(_state(query="帮我润色这段", domain="coding")) == "creation"
    assert route_domain(_state(query="跑一下 RMNet 的实验复现", domain="paper")) == "coding"
    assert route_domain(_state(query="改代码调参跑通训练", domain="paper")) == "coding"
    # 无信号 → 回退 LLM label
    assert route_domain(_state(query="RMNet 的 loss 是什么", domain="paper")) == "paper"
    assert route_domain(_state(query="把这支实验的结果整理成表格", domain="coding")) == "coding"


def test_route_domain_mixed_falls_back_to_label():
    # 混合信号（写+实验）→ 信任 LLM label，不武断
    assert route_domain(_state(query="写实验报告并改进代码", domain="paper")) == "paper"
    assert route_domain(_state(query="写实验报告并改进代码", domain="creation")) == "creation"


def test_route_domain_paper_qa_not_coding():
    # 危险误判回归：paper 域常识问答不应被宽泛关键词拉到 coding
    assert route_domain(_state(query="RMNet 实验部分用了什么指标", domain="paper")) == "paper"
    assert route_domain(_state(query="这篇论文怎么实现双流结构的", domain="paper")) == "paper"


# ---- decide_mode: domain forces plan ----

def test_decide_mode_creation_forces_plan():
    assert plan_mod.decide_mode(_state(query="写一篇 RMNet 综述", domain="creation")) == "plan"
    assert plan_mod.decide_mode(_state(query="跑实验", domain="coding")) == "plan"
    # domain=paper、无 plan 信号 → react（零回归）
    assert plan_mod.decide_mode(_state(query="RMNet 的 loss 是什么", domain="paper")) == "react"


# ---- DocStore lifecycle ----

def test_doc_lifecycle():
    tmp = Path(tempfile.mkdtemp())
    orig = creation.get_writing_dir
    creation.get_writing_dir = lambda: tmp
    try:
        doc_id = asyncio.run(creation._ensure_writing_doc("测试综述", ["intro", "method"], [
            {"args": {"section_id": "intro", "title": "1 引言",
                      "section_type": "introduction", "cites": ["RMNet"]}},
            {"args": {"section_id": "method", "title": "2 方法",
                      "section_type": "method", "cites": ["SRN"]}},
        ]))
        assert doc_id and _safe(doc_id)

        st = _data(creation.doc_get_state, {"doc_id": doc_id})
        assert st["status"] == "writing"
        assert [o["section_id"] for o in st["outline"]] == ["intro", "method"]
        assert st["outline"][0]["cites"] == ["RMNet"]

        # 写到全部章节 → 状态 done + 主 md 拼接
        _data(creation.doc_write_section, {"doc_id": doc_id, "section_id": "intro",
                                           "content": "RMNet 提出双流结构 [1]。"})
        st = _data(creation.doc_get_state, {"doc_id": doc_id})
        assert st["status"] == "writing"          # 只写了一章
        _data(creation.doc_write_section, {"doc_id": doc_id, "section_id": "method",
                                           "content": "SRN 使用稀疏表示 [2]。"})
        st = _data(creation.doc_get_state, {"doc_id": doc_id})
        assert st["status"] == "done"
        assert "双流结构" in st["assembled_md"] and "稀疏表示" in st["assembled_md"]

        # 不在 outline 的 section 拒绝写入
        bad = json.loads(asyncio.run(creation.doc_write_section.ainvoke(
            {"doc_id": doc_id, "section_id": "hack", "content": "x"})))
        assert bad["ok"] is False and bad["error_type"] == "param_error"

        # docx 导出 → 文件存在且 python-docx 可读回（WORKSPACE_DOCS 已隔离到 tmp）
        exp = _data(creation.doc_export_docx, {"doc_id": doc_id})
        assert exp["export_path"].endswith(f"{doc_id}.docx")
        from docx import Document
        d = Document(str(tmp / doc_id / "exports" / f"{doc_id}.docx"))
        texts = [p.text for p in d.paragraphs]
        assert any("双流结构" in t for t in texts), "docx must contain the section text"

        # 列表可见
        docs = _data(creation.doc_list, {})
        assert any(x["doc_id"] == doc_id for x in docs["docs"])
    finally:
        creation.get_writing_dir = orig
        shutil.rmtree(tmp, ignore_errors=True)


def test_safe_id_blocks_path_traversal():
    assert creation._safe_id("../../etc/passwd") == "etcpasswd"
    assert creation._safe_id("") == ""
    assert creation._safe_id("My Section 2!", "sec") == "mysection2"


class _FakeModel:
    """Plain model stub: returns pre-set reply text (no tool_calls)."""

    def __init__(self, reply: str):
        self.reply = reply

    async def ainvoke(self, msgs):
        class R:
            content = self.reply
            tool_calls = None
        return R()


def test_parse_steps_accepts_creator_target():
    # 回归：PlanStep.target Literal 曾缺 "creator" → 创作 outline 计划整单被
    # ValidationError 丢弃 → 写作链路静默断裂（chat 里"写一篇…"无任何产出）。
    parsed = plan_mod._parse_steps(
        '{"steps": [{"id": "ch-1", "description": "Write 1 引言", "target": "creator",'
        ' "args": {"section_id": "intro", "title": "1 引言", "section_type": "introduction",'
        ' "cites": ["RMNet"]}, "depends_on": []}]}'
    )
    assert parsed and parsed[0]["target"] == "creator"
    assert parsed[0]["args"]["cites"] == ["RMNet"]


def test_creation_plan_creates_doc():
    """fake LLM 产 outline → _creation_plan 建 doc + doc_id 注入每个步骤 args."""
    tmp = Path(tempfile.mkdtemp())
    orig = creation.get_writing_dir
    creation.get_writing_dir = lambda: tmp
    try:
        reply = ('{"steps": [{"id": "ch-1", "description": "Write 1 引言", "target": "creator",'
                 ' "args": {"section_id": "intro", "title": "1 引言", "section_type": "introduction",'
                 ' "cites": ["RMNet"]}, "depends_on": []},'
                 '{"id": "ch-2", "description": "Write 2 方法", "target": "creator",'
                 ' "args": {"section_id": "method", "title": "2 方法", "section_type": "method",'
                 ' "cites": []}, "depends_on": []}]}')
        out = asyncio.run(plan_mod._creation_plan(
            _FakeModel(reply), _state(query="写一篇 RMNet 综述", domain="creation"),
            "写一篇 RMNet 综述", "(none)", "(no resolved hints)"))
        assert out["doc_id"], "doc must be created"
        for s in out["plan"]:
            assert s["args"]["doc_id"] == out["doc_id"], "every step must carry doc_id"
        st = _data(creation.doc_get_state, {"doc_id": out["doc_id"]})
        assert [o["section_id"] for o in st["outline"]] == ["intro", "method"]
    finally:
        creation.get_writing_dir = orig
        shutil.rmtree(tmp, ignore_errors=True)


def test_agent_state_declares_domain():
    """回归：AgentState schema 必须显式声明 `domain`。

    LangGraph 以 schema 为准，节点返回未声明的 key 会被静默丢弃——此前只往
    UnderstandResult 加了 domain、漏了 AgentState，图内 state["domain"] 恒缺失
    → plan_node 永远走默认 paper 分支 → 「写综述」静默退化为普通检索（v10
    连通性断裂根因）。这条断言把根因钉死。
    """
    from agent.state import AgentState

    ann = AgentState.__annotations__
    assert "domain" in ann, "AgentState must declare `domain` (LangGraph drops undeclared keys)"
    assert "doc_id" in ann, "AgentState must declare `doc_id` (created by the creation plan)"


def test_executor_dispatches_creator_step():
    """target=creator 的 plan 步骤 → executor 折叠 launch args 成 task 调 creator 工具；
    creator 步骤以 doc 落盘为准(章节 done 才算产出,不是纯文本答复)。"""
    calls: dict = {}

    class _FakeCreator:
        async def ainvoke(self, args: dict, config=None) -> str:
            calls["task"] = args.get("task", "")
            return "intro | 50 words | wrote via doc_write_section"

    tmp = Path(tempfile.mkdtemp())
    orig_ws = creation.get_writing_dir
    import agent.tools as at
    orig = at.get_cached_tools
    creator_tool = _FakeCreator()
    creator_tool.name = "creator"  # _run_step 只需 dict 键，name 不参与
    at.get_cached_tools = lambda: [creator_tool]
    try:
        # 预置一份 doc,intro 已在 outline 且 done —— 落盘校验以此为准
        creation.get_writing_dir = lambda: tmp
        d = tmp / "abc123"
        (d / "sections").mkdir(parents=True)
        doc = {
            "doc_id": "abc123", "title": "t", "status": "writing",
            "outline": [{"section_id": "intro", "title": "1 引言",
                         "section_type": "introduction", "cites": [], "status": "done"}],
            "sections": {"intro": {"status": "done", "word_count": 50, "updated_at": ""}},
            "created_at": "", "updated_at": "",
        }
        (d / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

        step = {"id": "ch-1", "description": "Write intro", "target": "creator",
                "args": {"doc_id": "abc123", "section_id": "intro"}, "depends_on": []}
        out = asyncio.run(plan_mod._run_step(step, {}, {}))
        assert out["ok"], out
        assert "doc_id: abc123" in calls["task"], "args must fold into the task command block"
        assert "section_id: intro" in calls["task"]
        assert out["output"].startswith("intro | 50 words"), "output is the verified status line"
    finally:
        at.get_cached_tools = orig
        creation.get_writing_dir = orig_ws
        shutil.rmtree(tmp, ignore_errors=True)


def test_creator_step_fails_when_section_not_written():
    """回归：creator subagent 仅以纯文本作答、doc 里没有落盘 → 步骤必须判失败,
    不得把整章正文当作产出转发给 synthesize(此前「doc 只写第三章、聊天却回全文」)。"""

    async def _not_written_ainvoke(self, args: dict, config=None) -> str:
        return "这是一整章正文内容,哪怕是再完整的文本也不算落盘。"

    tmp = Path(tempfile.mkdtemp())
    orig_ws = creation.get_writing_dir
    import agent.tools as at
    orig = at.get_cached_tools
    at.get_cached_tools = lambda: [type("_F", (), {"name": "creator",
                                                   "ainvoke": _not_written_ainvoke})()]
    try:
        creation.get_writing_dir = lambda: tmp  # 空 workspace: abc123 不存在 → 未落盘
        step = {"id": "ch-1", "description": "Write intro", "target": "creator",
                "args": {"doc_id": "abc123", "section_id": "intro"}, "depends_on": []}
        out = asyncio.run(plan_mod._run_step(step, {}, {}))
        assert out["ok"] is False, "未落盘的章节必须失败"
        assert "未落盘" in out["error"], out["error"]
        assert out["output"] == "", "正文内容不得进入 subagent_results"
    finally:
        at.get_cached_tools = orig
        creation.get_writing_dir = orig_ws
        shutil.rmtree(tmp, ignore_errors=True)


def test_creation_plan_serializes_steps():
    """章节串行回归：_creation_plan 必须链上 depends_on,避免并行写 doc 竞争。"""
    tmp = Path(tempfile.mkdtemp())
    orig_ws = creation.get_writing_dir
    creation.get_writing_dir = lambda: tmp
    try:
        reply = ('{"steps": [{"id": "ch-1", "description": "Write 1", "target": "creator",'
                 ' "args": {"section_id": "s1", "title": "1"}, "depends_on": []},'
                 '{"id": "ch-2", "description": "Write 2", "target": "creator",'
                 ' "args": {"section_id": "s2", "title": "2"}, "depends_on": []}]}')
        out = asyncio.run(plan_mod._creation_plan(
            _FakeModel(reply), _state(query="写综述", domain="creation"),
            "写综述", "(none)", "(no resolved hints)"))
        assert out["plan"][0]["depends_on"] == []
        assert out["plan"][1]["depends_on"] == ["ch-1"], "后续章节必须依赖前章(串行)"
    finally:
        creation.get_writing_dir = orig_ws
        shutil.rmtree(tmp, ignore_errors=True)


def _safe(uid: str) -> bool:
    return creation._safe_id(uid) == uid


def _data(fn, args: dict) -> dict:
    raw = asyncio.run(fn.ainvoke(args))
    payload = json.loads(raw)
    assert payload.get("ok") is True, payload
    return payload["data"]


if __name__ == "__main__":
    test_route_domain_keyword_overrides_llm()
    test_route_domain_mixed_falls_back_to_label()
    test_route_domain_paper_qa_not_coding()
    test_decide_mode_creation_forces_plan()
    test_doc_lifecycle()
    test_safe_id_blocks_path_traversal()
    test_parse_steps_accepts_creator_target()
    test_creation_plan_creates_doc()
    test_agent_state_declares_domain()
    test_executor_dispatches_creator_step()
    test_creator_step_fails_when_section_not_written()
    test_creation_plan_serializes_steps()
    print("Phase A creation self-check OK")