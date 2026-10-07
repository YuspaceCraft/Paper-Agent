"""Phase 2 通用工具集 self-check.

Run: python agent/tests/test_generic.py
ponytail: assert-based, no framework. Covers the three core invariants:
路径越界被拒、calculator 拒绝非算术输入、user 角色无法触发 write_file。
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.providers.generic_provider import GenericProvider, GENERIC_FUNCS
from agent.providers import ToolDef
from agent.providers.mcp_provider import MCPProvider
from agent.core.tool_registry import tool_spec_from_def


def _run(coro):
    return asyncio.run(coro)


def _is_err(result: str) -> bool:
    try:
        payload = json.loads(result)
        return payload.get("outcome") == "failed"
    except Exception:
        return False


def test_path_escape_rejected():
    # read_file 路径越界 → permission_denied
    async def _go():
        gp = GenericProvider()
        r = await gp.call_tool("read_file", {"path": "../../../../etc/passwd"})
        assert _is_err(r), r
        assert "permission_denied" in r, r

    _run(_go())


def test_calculator_rejects_non_arithmetic():
    async def _go():
        # 属性/调用被 ast 白名单拒绝
        r = await GENERIC_FUNCS["calculator"]("__import__('os').system('id')")
        assert _is_err(r), r
        assert "param_error" in r, r
        # 字符串参与运算被拒
        r2 = await GENERIC_FUNCS["calculator"]("1 + 'a'")
        assert _is_err(r2), r2

    _run(_go())


def test_calculator_works():
    async def _go():
        r = await GENERIC_FUNCS["calculator"]("2 * (3 + 4) ** 2")
        assert r.strip() == "98", r

    _run(_go())


def test_user_role_cannot_write():
    async def _go():
        os.environ["AGENT_USER_ROLE"] = "user"
        gp = GenericProvider()
        r = await gp.call_tool("write_file", {"path": "should_not_exist.txt", "content": "x"})
        assert _is_err(r), r
        assert "permission_denied" in r, r

    _run(_go())


def test_read_and_list_work():
    async def _go():
        gp = GenericProvider()
        ls = await gp.call_tool("list_dir", {"path": "agent"})
        assert "safety.py" in ls, ls
        rd = await gp.call_tool("read_file", {"path": "agent/safety.py"})
        assert "mask_pii" in rd, rd

    _run(_go())


def test_artifact_read_continues_spilled_result(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    from agent.core.artifact_store import persist_text

    ref = persist_text("abcdefghij" * 2000, threshold=100)
    assert ref is not None

    async def _go():
        r = await GENERIC_FUNCS["artifact_read"](
            ref["artifact_id"], offset=0, max_chars=1000,
        )
        assert r.startswith("## Artifact")
        assert "continue with artifact_read" in r

    _run(_go())


def test_mcp_missing_annotations_fail_closed():
    td = ToolDef(
        name="srv__unknown",
        description="",
        parameters={},
        source="mcp",
        annotations={
            "readOnlyHint": None,
            "destructiveHint": None,
            "idempotentHint": None,
        },
    )
    spec = tool_spec_from_def(td)
    assert spec.side_effect is True
    assert spec.requires_approval() is True


def test_mcp_side_effect_call_is_not_retried():
    attempts = {"n": 0}

    class Session:
        async def call_tool(self, name, arguments):
            attempts["n"] += 1
            raise RuntimeError("connection lost")

    provider = MCPProvider({})
    td = ToolDef(
        name="srv__write",
        description="",
        parameters={},
        source="mcp",
        annotations={"readOnlyHint": False, "idempotentHint": False},
    )
    provider._tool_index["srv__write"] = (td, "srv")

    async def no_discover():
        return None

    async def connect(_server):
        return Session()

    provider._discover = no_discover
    provider._connect = connect
    try:
        _run(provider.call_tool("srv__write", {}))
    except RuntimeError as exc:
        assert "connection lost" in str(exc)
    else:
        raise AssertionError("side-effecting MCP call must surface the error")
    assert attempts["n"] == 1


if __name__ == "__main__":
    test_path_escape_rejected()
    test_calculator_rejects_non_arithmetic()
    test_calculator_works()
    test_user_role_cannot_write()
    test_read_and_list_work()
    test_mcp_missing_annotations_fail_closed()
    test_mcp_side_effect_call_is_not_retried()
    print("generic tools self-check OK")
