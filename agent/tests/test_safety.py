"""Phase 2 self-check — permission gate wired into BuiltinProvider.

Run: python agent/tests/test_safety.py
ponytail: assert-based, no framework. Only calls tools whose permission
check short-circuits BEFORE any network I/O (destructive under user role).
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.providers.builtin_provider import BuiltinProvider


def test_permission_gate_blocks_destructive():
    async def _run():
        os.environ["AGENT_USER_ROLE"] = "user"
        bp = BuiltinProvider()
        # destructive tool rejected for user role — returns before network
        r = await bp.call_tool("ingest_paper", {"paper_name": "whatever"})
        assert '"ok": false' in r, r
        assert '"permission_denied"' in r, r

    asyncio.run(_run())


if __name__ == "__main__":
    test_permission_gate_blocks_destructive()
    print("Phase 2 permission-gate integration OK")
