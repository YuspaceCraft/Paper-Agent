"""
task_provider.py — 领导-部门制监督工具（supervisor.py 的 agent 工具暴露层）。

纯薄层：信封 + 权限门 + 参数校验，领域逻辑全部在 agent/supervisor.py。
主 agent（领导）通过这些工具派发/监督/干预/验收隔离子 agent 任务：
  task_dispatch — 派发（长任务后台运行，立即回 task_id）
  task_progress — 读状态栈（next 节点 / iteration / messages / interrupt 问题）
  task_collect  — 收产出（质量验收素材）
  task_resume   — 回复 interrupt 暂停的子任务并续跑
  task_cancel   — 取消在跑任务
  task_list     — 全部派发任务快照
"""

from __future__ import annotations

import asyncio

from ..providers import ToolDef, ToolProvider
from ..safety import tool_allowed
from ..tool_contract import ok as _ok_contract, err as _err_contract


def _guard(task_id: str) -> str | None:
    if not (task_id or "").strip():
        return "task_id is required (returned by task_dispatch)"
    return None


# ---- 工具实现（返回统一信封） ----

async def _dispatch(role: str, title: str, task: str, context: str = "",
                    parent_thread: str = "") -> str:
    from ..supervisor import dispatch
    if not role or not task:
        return _err_contract(
            "param_error", "role and task are required — task must be a "
            "self-contained prompt (zero-state) for the sub-agent.")
    if context:
        task = f"{task}\n\n## Context from leader\n{context}"
    try:
        task_id = await dispatch(role, title, task,
                                 parent_thread=parent_thread)
        return _ok_contract({
            "task_id": task_id, "role": role, "status": "running",
            "message": f"已派发 {role} 子任务 {task_id}（后台运行）。"
                       f"可用 task_progress('{task_id}') 查进展，"
                       f"task_collect('{task_id}') 收产出。",
        })
    except ValueError as exc:
        return _err_contract("param_error", str(exc))
    except Exception as exc:
        return _err_contract(
            "transient", f"{type(exc).__name__}: {exc}",
            next_action="重试一次，若持续失败检查 supervisor init / 工具装配。")


async def _progress(task_id: str) -> str:
    from ..supervisor import progress
    g = _guard(task_id)
    if g:
        return _err_contract("param_error", g)
    try:
        card = await progress(task_id)
        return _ok_contract(card)
    except ValueError as exc:
        return _err_contract("param_error", str(exc))
    except Exception as exc:
        return _err_contract("transient", f"{type(exc).__name__}: {exc}")


async def _collect(task_id: str) -> str:
    from ..supervisor import collect
    g = _guard(task_id)
    if g:
        return _err_contract("param_error", g)
    try:
        out = await collect(task_id)
        return _ok_contract({"task_id": task_id, "status": "done",
                             "output": out})
    except ValueError as exc:
        return _err_contract("unknown", str(exc),
                             next_action="状态非 done 时不可 collect；先 task_progress。")
    except Exception as exc:
        return _err_contract("transient", f"{type(exc).__name__}: {exc}")


async def _resume(task_id: str, reply: str) -> str:
    from ..supervisor import resume
    g = _guard(task_id)
    if g:
        return _err_contract("param_error", g)
    if not reply:
        return _err_contract("param_error", "reply is required")
    try:
        await resume(task_id, reply)
        return _ok_contract({"task_id": task_id, "status": "resumed",
                             "message": "已回复子任务并续跑。"})
    except ValueError as exc:
        return _err_contract("param_error", str(exc))
    except Exception as exc:
        return _err_contract("transient", f"{type(exc).__name__}: {exc}")


async def _cancel(task_id: str) -> str:
    from ..supervisor import cancel
    g = _guard(task_id)
    if g:
        return _err_contract("param_error", g)
    try:
        await cancel(task_id)
        return _ok_contract({"task_id": task_id, "status": "cancelled"})
    except Exception as exc:
        return _err_contract("transient", f"{type(exc).__name__}: {exc}")


async def _list(kind: str = "") -> str:
    from ..supervisor import list_tasks
    try:
        tasks = await list_tasks(kind=kind)
        return _ok_contract({"tasks": tasks, "count": len(tasks)})
    except Exception as exc:
        return _err_contract("transient", f"{type(exc).__name__}: {exc}")


# ---- ToolDef + Provider ----

SUPERVISOR_TOOLDEFS = [
    ToolDef(
        name="task_dispatch",
        description=(
            "Dispatch a SELF-CONTAINED long-running job to an isolated sub-agent "
            "role (arxiv | ingest | creator | coder). Returns task_id immediately; "
            "the job runs in the BACKGROUND with its own toolset and its own state "
            "stack — the leader can task_progress(task_id) anytime, task_collect "
            "the output when done, task_resume an interrupt, or task_cancel. "
            "Returns {\"ok\":true,\"data\":{task_id,role,message}}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role": {"type": "string",
                         "description": "Sub-agent role: arxiv | ingest | creator | coder."},
                "title": {"type": "string", "default": "",
                          "description": "Short task label for supervision lists."},
                "task": {"type": "string",
                         "description": "SELF-CONTAINED task prompt for the worker (zero-state)."},
                "context": {"type": "string", "default": "",
                            "description": "Optional leader context appended to the task (resolved paper names etc.)."},
                "parent_thread": {"type": "string", "default": ""},
            },
            "required": ["role", "title", "task"],
        },
        source="builtin",
        annotations={"readOnlyHint": False},
    ),
    ToolDef(
        name="task_progress",
        description=(
            "Read a dispatched sub-agent task's STATE STACK: status, current next "
            "nodes, execution iteration, message count, and the leader question if "
            "the worker interrupted waiting for input. Use whenever the user asks "
            "'那任务到哪了 / 进展如何 / 好了吗'."
        ),
        parameters={"type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"]},
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
    ToolDef(
        name="task_collect",
        description=(
            "Collect a dispatched task's FINAL OUTPUT (only valid once status=done). "
            "The leader uses it for quality acceptance of the delegated result."
        ),
        parameters={"type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"]},
        source="builtin",
        annotations={"readOnlyHint": True},
    ),
    ToolDef(
        name="task_resume",
        description=(
            "Resume an interrupted dispatched task by replying to the worker's "
            "pause question (it called request_review via interrupt()). Execution "
            "continues after this reply. Leader intervention for decisions/approval."
        ),
        parameters={"type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "reply": {"type": "string",
                                  "description": "Leader's answer/decision to the worker."}},
                    "required": ["task_id", "reply"]},
        source="builtin",
        annotations={"readOnlyHint": False},
    ),
    ToolDef(
        name="task_cancel",
        description=(
            "Cancel a running dispatched task (leader intervention). Marks it "
            "cancelled; its checkpoint/output stays for inspection."
        ),
        parameters={"type": "object",
                    "properties": {"task_id": {"type": "string"}},
                    "required": ["task_id"]},
        source="builtin",
        annotations={"readOnlyHint": False},
    ),
    ToolDef(
        name="task_list",
        description=(
            "List all dispatched sub-agent tasks (newest first), optional role "
            "filter (arxiv|ingest|creator|coder). Status: pending/running/done/"
            "failed/cancelled/interrupted/orphaned."
        ),
        parameters={"type": "object",
                    "properties": {
                        "kind": {"type": "string", "default": "",
                                 "description": "Role filter."}}},
        source="builtin",
        annotations={"readOnlyHint": True, "idempotentHint": True},
    ),
]

_FUNC_MAP = {
    "task_dispatch": _dispatch,
    "task_progress": _progress,
    "task_collect": _collect,
    "task_resume": _resume,
    "task_cancel": _cancel,
    "task_list": _list,
}


class TaskProvider(ToolProvider):
    """领导-部门制监督工具提供商（父 agent 工具面可用）。"""

    name = "supervisor"

    def __init__(self):
        self._tool_map = {td.name: td for td in SUPERVISOR_TOOLDEFS}

    async def list_tools(self) -> list[ToolDef]:
        return list(self._tool_map.values())

    async def call_tool(self, name: str, arguments: dict):
        if name not in _FUNC_MAP:
            raise KeyError(f"Supervisor tool '{name}' not found")
        td = self._tool_map[name]
        if td is not None and not tool_allowed(td.annotations):
            return _err_contract(
                f"Action '{name}' is not authorized for the current role.",
                error_type="permission_denied")
        fn = _FUNC_MAP[name]
        # 本 provider 的工具是普通 async fn（非 @tool），直接 await 调用。
        return await fn(**dict(arguments))