"""
tasks.py — 统一任务监督视图的 HTTP 薄封装（领导-部门制，v12）。

薄封装：只做序列化，领域逻辑在 agent/task_registry.py（合并叠加层）。
GET /api/tasks        全部长期任务（派发子 agent 舱 + 实验 + 写作文档 + Redis 任务）
GET /api/tasks/search 按 term/kind 过滤（任务中心/前端可选消费）
POST /api/tasks/{task_id}/recover 接管 lease 过期的 orphaned 子 agent 任务
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("")
async def list_tasks(kind: str = ""):
    from agent.task_registry import list_tasks
    tasks = await list_tasks(kind=kind)
    return {"tasks": tasks, "count": len(tasks)}


@router.get("/search")
async def search_tasks(term: str = "", kind: str = ""):
    from agent.task_registry import find_tasks
    tasks = await find_tasks(term=term, kind=kind)
    return {"tasks": tasks, "count": len(tasks)}


@router.post("/{task_id}/recover")
async def recover_task(task_id: str):
    from agent.supervisor import recover

    try:
        started = await recover(task_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {
        "task_id": task_id,
        "started": started,
        "status": "recovering" if started else "already_running",
    }
