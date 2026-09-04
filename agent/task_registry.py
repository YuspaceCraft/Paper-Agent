"""
task_registry.py — 统一任务监督视图（领导-部门制：领导的可读「任务面板」）。

只读叠加层（DSN 式合并，无 LLM、无领域逻辑），把四类长期任务的存储/运行时压成
统一条目，供 task_node（监督台）与 /api/tasks 薄端点消费：

  kind=ingest|coder|creator|arxiv|…  派发子 agent 舱（supervisor.store + checkpoint）
  kind=experiment                   实验记录（domains/coding.ExperimentStore）
  kind=doc                          写作文档（domains/creation.DocStore）
  kind=pipeline|index|ingest_task   后台任务栈（web/api Redis task store）

统一条目：
  {task_id, kind, title, status, progress, detail, created_at}
status 归一化 pending|running|done|failed|other；原始状态留在 detail。
"""
from __future__ import annotations

import json

_KIND_SYNONYMS = {
    "experiment": ("实验", "exp", "experiment", "跑", "训练"),
    "coder": ("编码", "code", "委托", "delegate"),
    "creator": ("写作", "写论文", "写综述", "写文章", "文档", "doc"),
    "doc": ("写作文档", "论文文档", "doc", "writing"),
    "ingest": ("入库", "导入", "ingest", "索引", "index"),
    "pipelines": ("pdf", "解析", "pipeline", "处理"),
}


def _norm_status(s: str, *extra: str) -> tuple[str, str]:
    """状态归一化：pending|running|done|failed|other。返回 (normal, original)。"""
    s = (s or "unknown").strip().lower()
    tok = f" {s} "
    if any(t in tok for t in (" pending ", " waiting ", " outline ")):
        return "pending", s
    if any(t in tok for t in (" running ", " writing ", " processing ", " ready ")):
        return "running", s
    if any(t in tok for t in (" done ", " success ", " finished ", " completed ")):
        return "done", s
    if any(t in tok for t in (" failed ", " error ", " cancelled ", " rejected ")):
        return "failed", s
    return "other", s


async def _dispatched() -> list[dict]:
    """监督派发器的子 agent 舱（supervisor store 快照）。kind = 子 agent role。"""
    from .supervisor import list_tasks
    try:
        cards = await list_tasks()
    except Exception:
        return []
    out = []
    for m in cards or []:
        out.append({
            "task_id": m.get("task_id", ""),
            "kind": m.get("role") or m.get("kind") or "subagent",
            "title": m.get("title", ""),
            "status": m.get("status", "unknown"),
            "progress": m.get("progress", ""),
            "detail": {},
            "created_at": m.get("created_at", ""),
        })
    return out


async def _experiments() -> list[dict]:
    """实验记录（ExperimentStore）。kind=experiment。"""
    from .domains.coding import _list_experiments
    out = []
    for e in _list_experiments():
        norm, orig = _norm_status(e.get("status", ""))
        out.append({
            "task_id": e.get("exp_id", ""),
            "kind": "experiment",
            "title": f"{e.get('name', '')} [{e.get('project', '')}]"
                     if e.get("project") else e.get("name", ""),
            "status": norm,
            "progress": f"{orig} exit={e.get('exit_code')}",
            "detail": {"original_status": orig, "metrics": e.get("metrics", {})},
            "created_at": e.get("created_at", ""),
        })
    return out


async def _docs() -> list[dict]:
    """写作文档（DocStore）。kind=doc。"""
    from .domains.creation import doc_list
    try:
        pay = json.loads(await doc_list.ainvoke({"status": ""}))
    except Exception:
        return []
    if not pay.get("ok"):
        return []
    out = []
    for d in pay.get("data", {}).get("docs", []):
        norm, orig = _norm_status(d.get("status", ""))
        out.append({
            "task_id": d.get("doc_id", ""),
            "kind": "doc",
            "title": f"写作｜{d.get('title', '')}",
            "status": norm,
            "progress": f"章节数 {d.get('n_sections', 0)}",
            "detail": {"original_status": orig, "n_sections": d.get("n_sections", 0)},
            "created_at": d.get("created_at", "") or d.get("updated_at", ""),
        })
    return out


async def _redis_tasks() -> list[dict]:
    """后台任务栈（ingest/pdf/index）。kind 沿用 task.kind，"" → pipeline/index。"""
    try:
        from web.api.routers import _task_list
        tasks = _task_list(limit=50)
    except Exception:
        return []
    out = []
    for t in tasks or []:
        kind = t.get("kind") or ""
        if kind == "ingest":
            kind = "ingest"
        else:
            kind = t.get("stage") or "pipeline"
        out.append({
            "task_id": t.get("task_id", ""),
            "kind": kind,
            "title": f"{kind}｜{t.get('paper_name', t.get('task_id', ''))}",
            "status": t.get("status", "unknown"),
            "progress": t.get("progress", ""),
            "detail": {"stage": t.get("stage", ""), "error": t.get("error", "")},
            "created_at": t.get("created_at", ""),
        })
    return out


async def list_tasks(kind: str = "") -> list[dict]:
    """全部长期任务并集（latest first，cap 50）。kind 过滤（精确 kind 名）。"""
    merged = (await _dispatched()) + (await _experiments()) + (await _docs()) \
        + (await _redis_tasks())
    entries = []
    for m in merged:
        entries.append({
            "task_id": m.get("task_id", ""),
            "kind": m.get("kind", ""),
            "title": m.get("title", m.get("task_id", "")),
            "status": m.get("status", "unknown"),
            "progress": m.get("progress", ""),
            "detail": m.get("detail", {}),
            "created_at": m.get("created_at", ""),
        })
    if kind:
        entries = [e for e in entries if e["kind"] == kind]
    def _key(e):
        return _created_sort(e.get("created_at", ""))
    entries.sort(key=_key, reverse=True)
    return entries[:50]


def _created_sort(c: str) -> int:
    """created_at 字符串排序键：空值最小。yyyymmddhhmmss 紧凑可比较。"""
    c = str(c or "")
    return int("".join(ch for ch in c if ch.isdigit()) or "0") if c else 0


def _match_kind(term: str) -> tuple[list[str], bool]:
    """term 命中 kind 近义词 → 返回候选 kind 列表 + 是否命中。"""
    tl = term.lower()
    hits: list[str] = []
    for kind, syns in _KIND_SYNONYMS.items():
        if any(s in tl for s in syns):
            hits.append(kind)
    return hits, bool(hits)


def _looks_like_id(term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return False
    if t.startswith("task:") or t.startswith("exp_") or t.startswith("doc_"):
        return True
    compact = "".join(ch for ch in t if ch.isalnum())
    return len(compact) >= 6 and compact.isalnum() and compact.isascii()


async def find_tasks(term: str = "", kind: str = "") -> list[dict]:
    """按 kind 近义词 / id 形似 / 标题模糊匹配任务条目。term 空 → 全量。"""
    entries = await list_tasks(kind=kind)
    term = (term or "").strip()
    if not term:
        return entries

    kinds, kind_hit = _match_kind(term)
    if kind_hit:
        entries = [e for e in entries if e["kind"] in kinds]

    if _looks_like_id(term):
        compact = term.replace("task:", "").strip()
        ids = [e for e in entries if compact in e["task_id"]]
        if ids:
            return ids

    # 标题模糊（大小写折叠 + 子串）
    tl = term.lower()
    title_hits = [
        e for e in entries
        if tl in str(e.get("title", "")).lower()
        or tl in str(e.get("task_id", "")).lower()
    ]
    if title_hits:
        return title_hits

    # 兜底：kind 命中仍空 → 返回 kind 下前几条（避免空手）
    if kind_hit:
        return entries[:8]
    return []


async def summarize(term: str = "", kind: str = "") -> dict:
    """压缩快照（task_node / 未来 prompt 注入用）。"""
    tasks = await find_tasks(term=term, kind=kind)
    running = [t for t in tasks if t["status"] == "running"]
    return {
        "total": len(tasks),
        "running": len(running),
        "tasks": [
            {"task_id": t["task_id"], "kind": t["kind"], "title": t["title"],
             "status": t["status"], "progress": t["progress"]}
            for t in tasks[:12]
        ],
    }