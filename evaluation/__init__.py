"""
evaluation — Agent 评测体系（评估端）。

职责：全链路 trace 结构化采集（评估端唯一事实源）→ 四类指标（检索 / 工具 /
任务 / LLM-as-judge）→ run 报告与 badcase 归因 →/api/eval 与前端评测页。

与 agent 框架的关系：
- agent/observability.log_event 经 sink 双写进 trace_store.db（零侵入透传）；
- 插桩新增的显式事件（intent / llm_call / retrieved_context / plan /
  final_answer / turn_end）由 events.py 的 emit_* 写入；
- 评测与指标复用 agent / retrieval_orchestrator / retrieval 现有设施。

快速使用（M3 后）：
    python -m evaluation run --manifest eval_output/datasets/manifest_v1_115qa.jsonl --limit 5
    python -m evaluation show <run_id>
"""

from .config import EvalConfig
from .trace_store import TRACE_DB, TraceStore, get_trace_store, new_run_id

__all__ = [
    "EvalConfig",
    "TRACE_DB",
    "TraceStore",
    "get_trace_store",
    "new_run_id",
]

__version__ = "0.1.0"