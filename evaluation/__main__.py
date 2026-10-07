"""
evaluation CLI — 评测跑批与查询。

用法：
  python -m evaluation run   --manifest <path> [--limit N] [--no-judge]
  python -m evaluation recovery --manifest <path> [--limit N]
  python -m evaluation list
  python -m evaluation show <run_id> [--badcases]
  python -m evaluation qrels --manifest <path> --out <path>
  python -m evaluation prune [--older-than-days 30] [--run-id <id>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import EvalConfig
from . import datasets


def _cfg() -> EvalConfig:
    return EvalConfig.from_env()


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _cfg()
    if args.limit:
        cfg.max_queries = args.limit
    from .trace_store import new_run_id
    from .runner import run_eval
    qas = datasets.load_manifest(args.manifest)
    print(f"[eval] loading {len(qas)} QA from {args.manifest}")
    if args.limit:
        qas = qas[: args.limit]
        print(f"[eval] limited to {len(qas)} queries (small-sample pilot)")
    coverage = datasets.validate_ground_truth_coverage(qas)
    if coverage.get("available"):
        print(
            "[eval] ground-truth coverage: "
            f"{coverage['matched_count']}/{coverage['ground_truth_count']} "
            f"({coverage['coverage']:.1%}) in {coverage['rag_path']}"
        )
        if coverage["stale_query_count"] and not args.allow_stale_dataset:
            print(
                "[eval] ERROR: manifest is stale against the current RAG corpus "
                f"({coverage['stale_query_count']} queries have no matching ground truth). "
                "Regenerate it with `python -m retrieval_orchestrator generate`, "
                "or pass --allow-stale-dataset to run anyway.",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if args.dataset:
        ds_id = args.dataset
    else:
        ds_id = datasets.dataset_id_for(qas)
    run_id = args.run_id or new_run_id()
    print(f"[eval] run_id={run_id}  （可在前端评测面板订阅同一 run 的实时进度）",
          flush=True)

    from .runner import ensure_running_row

    def _progress(i: int, total: int, qid: str) -> None:
        print(f"[eval] {i}/{total}  qid={qid} ...", flush=True)

    async def _run_once() -> dict:
        from .trace_store import get_trace_store

        await ensure_running_row(
            get_trace_store(), run_id, ds_id, len(qas),
        )
        return await _run_with_progress(
            cfg, qas, ds_id, run_id, use_judge=not args.no_judge,
            on_progress=_progress,
        )

    report = asyncio.run(_run_once())

    print("\n=== EVAL REPORT ===")
    print("run_id    :", report["run_id"])
    print("dataset_id:", report["dataset_id"])
    print("overall   :", json.dumps(report["overall"], ensure_ascii=False))
    print("judge     :", json.dumps(report.get("judge", {}), ensure_ascii=False))
    print("gate      :", report["baseline_delta"]["gate"])
    bc = len(report["badcases"])
    print(f"badcases  : {bc}  →  `python -m evaluation show {report['run_id']} --badcases`")
    if bc:
        for b in report["badcases"][:5]:
            print(f"  - [{b['category']}] {b.get('query', '')[:60]}  "
                  f"mrr={b.get('mrr')} trace={b.get('trace_id')}")


def _fmt(x) -> str:
    return "—" if x is None else str(x)


async def _run_with_progress(cfg, qas, ds_id: str, run_id: str, *,
                             use_judge: bool, on_progress=None) -> dict:
    """跑批 + 实时打印：订阅与前端同一条 live 事件流（逐条指标 + 累计聚合）。

    打印的数字与 SSE / 报告完全同源（都来自 runner 发布的 aggregate），不存在
    「终端一套算法、界面另一套」的问题。
    """
    from .live import stream_run_events
    from .runner import run_eval

    async def _printer() -> None:
        try:
            async for ev in stream_run_events(run_id, poll_s=0.5):
                t = ev.get("type")
                if t == "query_finished":
                    print(f"[eval]   └ [{ev.get('category')}] "
                          f"recall@5={_fmt(ev.get('recall@5'))} "
                          f"mrr={_fmt(ev.get('mrr'))} "
                          f"{_fmt(ev.get('duration_s'))}s "
                          f"tokens={_fmt(ev.get('tokens'))} "
                          f"ok={ev.get('success')}", flush=True)
                elif t == "aggregate":
                    ov = ev.get("overall") or {}
                    eta = ev.get("eta_s")
                    print(f"[eval] 累计 {ev.get('done')}/{ev.get('total')} "
                          f"recall@5={_fmt(ov.get('recall@5'))} "
                          f"mrr={_fmt(ov.get('mrr'))} "
                          f"task={_fmt(ov.get('task_success_rate'))} "
                          f"工具={_fmt(ov.get('tool_success_rate'))} "
                          f"tokens={_fmt(ov.get('tokens_total'))} "
                          f"${_fmt(ov.get('cost_usd'))}"
                          + (f"  ETA {eta}s" if eta else ""), flush=True)
                elif t == "judge_started":
                    print(f"[eval] judge 采样 {ev.get('sample_size')} 条…",
                          flush=True)
                elif t == "judge_finished":
                    print(f"[eval] judge 完成: context_hit="
                          f"{_fmt(ev.get('context_hit'))} faithfulness="
                          f"{_fmt(ev.get('faithfulness'))}", flush=True)
        except Exception:  # noqa: BLE001 — 实时打印失败不影响跑批
            pass

    printer = asyncio.create_task(_printer())
    try:
        return await run_eval(cfg, qas, run_id=run_id, dataset_id=ds_id,
                              with_judge=use_judge, on_progress=on_progress)
    finally:
        try:
            await asyncio.wait_for(printer, timeout=5)
        except Exception:  # noqa: BLE001 — 收尾等待失败就把打印任务取消
            printer.cancel()
        from agent.supervisor import shutdown_supervisor
        from agent.tools import close_tools

        await shutdown_supervisor()
        await close_tools()


def cmd_recovery(args: argparse.Namespace) -> None:
    cfg = _cfg()
    if args.limit:
        cfg.max_queries = args.limit
    from .runner import run_recovery_test
    qas = datasets.load_manifest(args.manifest)
    res = asyncio.run(run_recovery_test(cfg, qas[: args.limit or 3]))
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_list(_args: argparse.Namespace) -> None:
    from .trace_store import get_trace_store

    async def _go():
        store = get_trace_store()
        try:
            # 读路径收敛：进程已消失却还挂在 running 的 run → interrupted
            from .runner import reconcile_stale_runs
            await reconcile_stale_runs(store)
            conn = await store._ensure_conn()
            cur = await conn.execute(
                "SELECT run_id, dataset_id, status, query_count, started_at,"
                " overall FROM eval_runs ORDER BY started_at DESC LIMIT 20")
            rows = await cur.fetchall()
        finally:
            await store.close()
        print(f"{'run_id':<24} {'dataset':<18} {'status':<9} {'n':<4} "
              f"{'started_at':<20} overall")
        for r in rows:
            ov = {}
            try:
                ov = json.loads(r[5] or "{}")
            except ValueError:
                pass
            brief = {k: ov.get(k) for k in ("recall@5", "mrr", "task_success_rate")}
            print(f"{r[0]:<24} {r[1]:<18} {r[2]:<9} {r[3] or 0:<4} {r[4]:<20} {brief}")
    asyncio.run(_go())


def cmd_show(args: argparse.Namespace) -> None:
    from .trace_store import get_trace_store

    async def _go():
        store = get_trace_store()
        try:
            from .runner import reconcile_stale_runs
            await reconcile_stale_runs(store)
            conn = await store._ensure_conn()
            cur = await conn.execute(
                "SELECT * FROM eval_runs WHERE run_id=?", (args.run_id,))
            return await cur.fetchone()
        finally:
            await store.close()
    row = asyncio.run(_go())
    if not row:
        print(f"run not found: {args.run_id}")
        sys.exit(1)
    names = ["run_id", "dataset_id", "started_at", "finished_at", "status",
             "query_count", "metadata", "overall", "dimension", "badcases",
             "judge", "tool_metrics", "task_metrics", "baseline_delta",
             "cost_estimate", "notes"]
    rd = dict(zip(names, row))
    print("run_id    :", rd["run_id"])
    print("dataset_id:", rd["dataset_id"])
    print("status    :", rd["status"], " n =", rd["query_count"])
    print("started_at:", rd["started_at"], " finished_at:", rd["finished_at"])
    if rd["overall"]:
        ov = json.loads(rd["overall"])
        if ov.get("duration_s") is not None:
            print("duration_s:", ov["duration_s"], "s（完整用时：启停计时）")
        print("overall   :", json.dumps(ov, ensure_ascii=False, indent=2))
    if rd["judge"]:
        print("judge     :", json.dumps(json.loads(rd["judge"]),
                                        ensure_ascii=False, indent=2))
    if rd["baseline_delta"]:
        bd = json.loads(rd["baseline_delta"])
        print("gate      :", bd.get("gate"))
    if args.badcases and rd["badcases"]:
        print("\nbadcases  :")
        for b in json.loads(rd["badcases"]):
            print(f"  [{b.get('category')}] {b.get('query_id')} "
                  f"mrr={b.get('mrr')} errors={b.get('tool_errors')}")
            print(f"      query: {b.get('query', '')[:80]}")


def cmd_qrels(args: argparse.Namespace) -> None:
    from .metrics.retrieval import export_qrels
    export_qrels(args.manifest, args.out)
    print(f"qrels written: {args.out}")


def cmd_prune(args: argparse.Namespace) -> None:
    from .trace_store import get_trace_store

    async def _go():
        store = get_trace_store()
        deleted = await store.prune(older_than_days=args.older_than_days,
                                    run_id=args.run_id)
        await store.close()
        print(f"pruned {deleted} trace events (live >{args.older_than_days}d"
              f"{f', run={args.run_id}' if args.run_id else ''})")
    asyncio.run(_go())


def main() -> None:
    p = argparse.ArgumentParser(prog="evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="评测跑批（agent 全链路）")
    pr.add_argument("--manifest", required=True)
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--dataset", default=None)
    pr.add_argument("--no-judge", action="store_true")
    pr.add_argument(
        "--allow-stale-dataset",
        action="store_true",
        help="Run even when ground-truth chunk IDs are absent from the current corpus.",
    )
    pr.add_argument("--run-id", default=None,
                    help="指定 run_id（便于前端订阅同一 run 的实时进度）")
    pr.set_defaults(func=cmd_run)

    prr = sub.add_parser("recovery", help="中断恢复批量测试")
    prr.add_argument("--manifest", required=True)
    prr.add_argument("--limit", type=int, default=0)
    prr.set_defaults(func=cmd_recovery)

    pl = sub.add_parser("list", help="最近评测 run 列表")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="查看 run 报告")
    ps.add_argument("run_id")
    ps.add_argument("--badcases", action="store_true")
    ps.set_defaults(func=cmd_show)

    pq = sub.add_parser("qrels", help="导出 TREC qrels")
    pq.add_argument("--manifest", required=True)
    pq.add_argument("--out", required=True)
    pq.set_defaults(func=cmd_qrels)

    pp = sub.add_parser("prune", help="清理 trace")
    pp.add_argument("--older-than-days", type=int, default=30)
    pp.add_argument("--run-id", default=None)
    pp.set_defaults(func=cmd_prune)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
