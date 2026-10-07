"""
verify_smith.py — LangSmith 链路接入最小验证（P0，join 键打通）。

跑一条真实对话，然后用 langsmith Client 回读该 trace：
- 断言 LangSmith 根 run id == 本地 trace_id（同值同键 → 双向定位成立）
- 汇总嵌套结构：graph 节点 chain、LLM span（含真实 token）、工具 span、错误数
- 输出 LangSmith 控制台链接（瀑布图/时间线可视化）

用法：
    C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation.verify_smith \
        "RMNet 的 loss 函数是什么？" --mode react
    C:/Users/30811/miniconda3/envs/demo/python.exe -m evaluation.verify_smith \
        "hello" --mode auto        # 快速冒烟（无工具，仅 LLM/graph spans）
可选 --thread 固定会话 id（默认自动生成 verify:<ts>）。

已知环境坑：agent react 循环用 MCP stdio 服务（如 arxiv）时，进程收尾可能抛
「Attempted to exit cancel scope」的 MCP 关闭噪音并中断 CLI；脚本对 agent_run
做了容错（对话本地已成功仍继续回读 LangSmith），但建议从 uvicorn/前端 SSE 会话
观察完整链路（该场景不触发收尾噪音）。MCP 收尾问题见 TROUBLESHOOTING。

退出码：0 = 全部断言通过；1 = 失败。采集是 best-effort：本地对话本身已成功执行。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence

_PROJECT = os.getenv("LANGSMITH_PROJECT", "paper-agent")


def _slug(s: str) -> str:
    """uuid 字符串规范化：hex/hashed → dashed（LangSmith API 用 dashed）。"""
    try:
        return str(_uuid.UUID(s))
    except (ValueError, TypeError):
        return s


def _num(v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v or 0)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _usage(run) -> tuple[float, float, float, float]:
    """从 run.extra.metadata 提取 (input, output, total, cached) token。

    langchain 各版本键名不一（prompt/completion vs input/output），防御式兼容。
    """
    meta = ((run.extra or {}).get("metadata") or {}) if hasattr(run, "extra") else {}
    usage = meta.get("usage") or {}
    full = meta.get("full_usage") or usage

    def n(*keys: str) -> float:
        for k in keys:
            if k in full:
                return _num(full[k])
            if k in usage:
                return _num(usage[k])
        return 0.0

    cached = 0.0
    details = full.get("input_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(details, dict):
        cached = _num(details.get("cached_tokens"))
    return n("input_tokens", "prompt_tokens"), n("output_tokens", "completion_tokens"), \
        n("total_tokens"), cached


def _err_of(run) -> str:
    return getattr(run, "error", None) or ""


def _halt(code: int) -> int:
    """绕过 asyncio 收尾直接退出。

    MCP stdio（如 arxiv）子进程在事件循环关闭（shutdown_asyncgens）时触发
    cancel-scope wedge 会挂死进程（见 TROUBLESHOOTING「agent / MCP stdio 收尾」）；
    也顺带避免孤儿子进程占住管道导致 CLI 假死。调用前务必 flush。
    """
    sys.stdout.flush()
    os._exit(code)


async def _amain(query: str, mode: str, thread_id: str) -> int:
    from agent.graph import run as agent_run
    from agent.observability import get_trace_id

    print(f"[1/5] 跑对话  query={query!r}  mode={mode}  thread={thread_id}", flush=True)
    t0 = time.monotonic()
    try:
        await agent_run(query, thread_id=thread_id, mode=mode)
        print(f"[2/5] 对话完成（{time.monotonic() - t0:.1f}s，本地 trace 已记录）", flush=True)
    except Exception as exc:  # noqa: BLE001 — MCP stdio 收尾噪音等，对话本地可能已成功
        print(f"[2/5] 对话端异常（{time.monotonic() - t0:.1f}s）：{type(exc).__name__}: {exc}",
              "\n      本地 turn 已落 trace_store；继续回读 LangSmith。", flush=True)

    trace_id = get_trace_id() or ""
    rid_slug = _slug(trace_id)
    print(f"[3/5] 本地 trace_id={trace_id}", flush=True)

    import langsmith

    client = langsmith.Client()

    def _find_root(slug: str):
        """回读根 run。先按 run_id 查（索引延迟轮询 ≤60s），再按时间窗 + id 扫描兜底。"""
        # langsmith >= 0.8 把 get_run 改名为 read_run；read_root_run 两者兼容。
        from agent.core.trace_export import read_root_run

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                return read_root_run(client, slug)
            except Exception:  # noqa: BLE001 — 尚未索引到
                time.sleep(2)
        # 兜底：window 扫描根 run
        try:
            start = (datetime.now(timezone.utc) - timedelta(minutes=20)).replace(microsecond=0)
            for r in client.list_runs(project_name=_PROJECT, start_time=start, limit=100):
                if getattr(r, "parent_run_id", None):
                    continue
                if _slug(str(getattr(r, "id", ""))) == slug:
                    return r
        except Exception:  # noqa: BLE001
            pass
        return None

    print("[4/5] 等待 LangSmith 回读根 run（轮询 ≤60s，SDK 异步队列可能延迟）…", flush=True)
    root = _find_root(rid_slug) if rid_slug else None
    if root is None:
        print(f"[FAIL] 60s 内未能在 LangSmith 看到 run {trace_id}")
        print("      检查：.env 的 LANGSMITH_TRACING / LANGSMITH_API_KEY / 网络可达（api.smith.langchain.com）")
        print("      本地对话已完成，trace 事件仍写入 trace_store.db（best-effort 兜底）。")
        return _halt(1)

    if _slug(str(root.id)) != rid_slug:
        print(f"[FAIL] join 键断裂：LangSmith 根 run id={root.id} != 本地 trace_id={trace_id}")
        return _halt(1)
    print(f"[4/5] ✓ join 键断言成立：LangSmith 根 run id == 本地 trace_id == {trace_id}", flush=True)
    print(f"      root: {getattr(root, 'name', '?')} | run_type={getattr(root, 'run_type', '?')} | "
          f"status={'ok' if not _err_of(root) else 'error'} | session={getattr(root, 'session_id', None)}", flush=True)

    spans = list(client.list_runs(trace_id=rid_slug))
    llms = [s for s in spans if s.run_type == "llm"]
    tools = [s for s in spans if s.run_type == "tool"]
    erred = [s for s in spans if _err_of(s)]

    t_in = t_out = t_cached = 0.0
    for s in llms:
        i, o, _tot, c = _usage(s)
        t_in += i
        t_out += o
        t_cached += c

    children = [s for s in spans if _slug(str(getattr(s, "parent_run_id", ""))) == rid_slug]
    direct = [f"{s.name}" for s in children if s.run_type == "chain"]

    # 游离根 run：同一时间窗内独立成 trace 的 LLM/工具/子链（未嵌套入 graph）。
    # 现状（P1 待修）：节点内调用 _stream_llm/traced_ainvoke/stool.ainvoke 未透传
    # runnable config → LangSmith 以独立根 run 记录。见 docs/agent评测体系构建.md §六。
    detached: list = []
    try:
        win = list(client.list_runs(
            project_name=_PROJECT,
            start_time=getattr(root, "start_time").replace(microsecond=0) - timedelta(seconds=5),
            end_time=getattr(root, "end_time").replace(microsecond=999999),
            limit=100))
        detached = [
            r for r in win
            if not getattr(r, "parent_run_id", None)
            and _slug(str(getattr(r, "id", ""))) != rid_slug
        ]
    except Exception:  # noqa: BLE001
        pass

    print("[5/5] 结构统计：", flush=True)
    print(f"      graph 根下嵌套 spans={len(spans)}（含首层节点 + 嵌套 LLM/工具）", flush=True)
    print(f"      graph 首层节点: {' → '.join(direct) if direct else '(空)'}", flush=True)
    print(f"      graph 内 LLM spans={len(llms)}（token in≈{t_in:.0f} out≈{t_out:.0f} cached≈{t_cached:.0f}）", flush=True)
    if tools:
        print(f"      graph 内 tool spans={len(tools)}: {', '.join(sorted({s.name for s in tools}))}", flush=True)
    else:
        print("      graph 内 tool spans=0", flush=True)
    if detached:
        names = {f"{r.name or '?'}({r.run_type})" for r in detached if r.run_type in {"llm", "tool"}}
        print(f"      ⚠ 游离根 run={len(detached)} 个",
              f"{'（LLM/工具:' + ', '.join(sorted(names)) + '）' if names else ''}", flush=True)
        print("        └ P1：节点内调用透传 runnable config 后并入 graph trace（见设计文档）", flush=True)
    print(f"      error spans={len(erred)}" + ("" if not erred else f"  ← {[s.name for s in erred]}"), flush=True)

    nested = len(spans) > 1  # 至少根 + 首层节点
    nesting_gap = bool(detached)
    ok = nested and not erred
    print(flush=True)
    if ok:
        print("PASS：join 键成立 + graph run 树已落地 LangSmith 且无错误。", flush=True)
        try:
            print(f"控制台：{client.get_run_url(run=root, project_name=_PROJECT)}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"控制台搜索：project={_PROJECT} 内搜索 trace_id={trace_id}", flush=True)
    else:
        print("WARN：join 键成立但 graph run 树异常（嵌套缺失或有 error span——见上）。", flush=True)
    if nesting_gap:
        print("提示：存在游离 LLM/工具 run，完整瀑布图（nesting 修复）见 P1。", flush=True)
    return _halt(0 if ok else 1)


def main() -> int:
    # Windows 控制台默认 GBK，输出 UTF-8 字符（✓/≈/→）会 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="LangSmith 链路接入最小验证")
    ap.add_argument("query", nargs="?", default="RMNet 的 loss 函数是什么？")
    ap.add_argument("--mode", default="react")
    ap.add_argument("--thread", default="")
    args = ap.parse_args()
    thread_id = args.thread or f"verify:{datetime.now():%Y%m%d%H%M%S}"
    return asyncio.run(_amain(args.query, args.mode, thread_id))


if __name__ == "__main__":
    raise SystemExit(main())
