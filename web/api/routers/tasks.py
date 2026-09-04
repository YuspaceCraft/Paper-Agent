"""
tasks.py — 统一任务监督视图的 HTTP 薄封装（领导-部门制，v12）。

薄封装：只做序列化，领域逻辑在 agent/task_registry.py（合并叠加层）。
GET /api/tasks        全部长期任务（派发子 agent 舱 + 实验 + 写作文档 + Redis 任务）
GET /api/tasks/search 按 term/kind 过滤（任务中心/前端可选消费）
"""
from __future__ import annotations

from fastapi import APIRouter

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