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


def _run(coro):
    return asyncio.run(coro)


def _is_err(result: str) -> bool:
    try:
        return json.loads(result).get("ok") is False
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


if __name__ == "__main__":
    test_path_escape_rejected()
    test_calculator_rejects_non_arithmetic()
    test_calculator_works()
    test_user_role_cannot_write()
    test_read_and_list_work()
    print("generic tools self-check OK")
