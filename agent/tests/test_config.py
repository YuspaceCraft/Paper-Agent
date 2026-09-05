"""(配置中心) self-check — config_store 持久化 + 工具/skills 停用过滤 + .mcp.json 读写。

Run: python agent/tests/test_config.py
ponytail: assert-based, no framework, no LLM calls. Config file paths redirected
to a temp dir via module-level CONFIG_PATH monkeypatch.
"""
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent import config_store  # noqa: E402
from agent.config_store import CONFIG_PATH  # noqa: E402


def _with_tmp_config():
    """返回 (tmp_root_ctx, config_path)。当前 CONFIG_PATH 指到临时目录。"""
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "config.json"
    original = CONFIG_PATH
    config_store.CONFIG_PATH = cfg_path
    return tmp, original, cfg_path


def _teardown(tmp: str, original):
    shutil.rmtree(tmp, ignore_errors=True)
    config_store.CONFIG_PATH = original
    config_store.clear_overrides()


def test_store_roundtrip_and_defaults():
    tmp, original, cfg_path = _with_tmp_config()
    try:
        # 默认值
        assert config_store.get("experiment", "delegate_prefer") == "mcp"
        assert config_store.get("tools", "disabled") == {
            "parent": [], "arxiv": [], "ingest": [], "creator": [], "coder": []}
        # 写 → 读
        config_store.set("experiment", "delegate_prefer", "cli")
        config_store.set_many("experiment", {"delegate_timeout": 60})
        assert config_store.get("experiment", "delegate_prefer") == "cli"
        assert config_store.get_delegate_timeout() == 60
        assert cfg_path.exists()
        # 类型化 getter
        config_store.set_many("tools", {"disabled": {"parent": ["write_file"], "arxiv": []}})
        d = config_store.get_disabled_tools()
        assert d["parent"] == ["write_file"]
        assert d["arxiv"] == []
        assert config_store.get_delegate_prefer() == "cli"
        # 非法值回退
        config_store.set("experiment", "delegate_prefer", "bogus")
        assert config_store.get_delegate_prefer() == "mcp"
    finally:
        _teardown(tmp, original)


def test_store_corrupted_file_falls_back():
    tmp, original, cfg_path = _with_tmp_config()
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text("{not json", encoding="utf-8")
        assert config_store.get("experiment", "delegate_prefer") == "mcp"
        assert config_store.get_disabled_skills() == []
    finally:
        _teardown(tmp, original)


def test_subagent_disabled_filter():
    """停用某 subagent 全部工具 → 该 subagent 不出现在 as_tool 列表；部分停用仍保留。"""
    tmp, original, cfg_path = _with_tmp_config()
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class _Args(BaseModel):
            pass

        async def _noop(**_) -> str:
            return "ok"

        def _mk(name: str):
            return StructuredTool(name=name, description=name, args_schema=_Args, coroutine=_noop)

        tools = {
            "arxiv__search_papers": _mk("arxiv__search_papers"),
            "arxiv__get_paper_data": _mk("arxiv__get_paper_data"),
            "ingest_paper": _mk("ingest_paper"),
        }

        from agent.subagents import SUBAGENTS, build_subagents

        # 部分停用 arxiv（仍保留一个工具）→ subagent 仍然存在
        config_store.set_many("tools", {"disabled": {"arxiv": ["arxiv__search_papers"]}})
        subs = build_subagents(tools)
        assert any(getattr(s, "name", "") == "arxiv" for s in subs), "arxiv subagent should survive partial disable"
        assert any(getattr(s, "name", "") == "ingest" for s in subs)

        # 全量停用 ingest → ingest subagent 被省略
        config_store.set_many("tools", {"disabled": {"ingest": ["ingest_paper"]}})
        subs = build_subagents(tools)
        assert not any(getattr(s, "name", "") == "ingest" for s in subs), "ingest should be omitted"
    finally:
        _teardown(tmp, original)


def test_skill_disabled_filter():
    tmp, original, cfg_path = _with_tmp_config()
    skills_tmp = tempfile.mkdtemp()
    try:
        for name in ("alpha", "beta"):
            d = Path(skills_tmp) / name
            d.mkdir()
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {name} skill\n---\nbody",
                encoding="utf-8")

        config_store.set_many("skills", {"disabled": ["beta"]})

        from agent.providers.skill_provider import SkillProvider
        p = SkillProvider(skills_tmp)
        listing = p._do_list()
        assert "alpha" in listing and "beta" not in listing, listing
    finally:
        shutil.rmtree(skills_tmp, ignore_errors=True)
        _teardown(tmp, original)


def test_mcp_json_roundtrip_preserves_unknown_fields():
    from agent.providers.mcp_provider import (
        read_mcp_config_raw,
        write_mcp_config,
    )
    tmp = tempfile.mkdtemp()
    try:
        path = Path(tmp) / ".mcp.json"
        path.write_text(json.dumps({
            "custom_top": {"note": "keep me"},
            "mcpServers": {
                "arxiv": {"command": "python", "args": ["-m", "mcp_simple_arxiv"]},
            },
        }), encoding="utf-8")

        write_mcp_config(
            {"arxiv": {"command": "python", "args": ["-m", "mcp_simple_arxiv"], "disabled": True}},
            path,
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["custom_top"] == {"note": "keep me"}, "unknown top-level key must survive"
        assert raw["mcpServers"]["arxiv"]["disabled"] is True

        # 缺失文件 → 空结构回退
        missing = Path(tmp) / "nope" / ".mcp.json"
        assert read_mcp_config_raw(missing) == {"mcpServers": {}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_workspace_paths_unaffected():
    """config_store 写实验配置不触碰 workspace_config 的路径设置。"""
    tmp, original, cfg_path = _with_tmp_config()
    try:
        from agent import workspace_config as wc
        wc.set_override("project_root", Path(tmp) / "proj")
        config_store.set_many("experiment", {"delegate_timeout": 120})
        # 路径 override 未被动过
        assert wc.get_project_root() == Path(tmp) / "proj"
    finally:
        _teardown(tmp, original)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")