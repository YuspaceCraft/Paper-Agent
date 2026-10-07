"""
evaluation.metrics — 四类指标引擎（纯函数，输入 trace 事件 / 评测 manifest）。

- retrieval：复用 retrieval_orchestrator.evaluator 指标函数，逐条 badcase 导出
- tools：工具成功率 / 耗时分布 / 错误归因
- task：任务执行成功率 / 中断恢复率
- judge：LLM-as-judge（上下文命中 / 忠实度 / 意图与模式决策准确）
"""

from .retrieval import aggregate, export_badcases, export_qrels, per_query_metrics
from .tools import aggregate_tools
from .contracts import evaluate_task_contract
from .task import aggregate_task_metrics, task_success

__all__ = [
    "aggregate", "export_badcases", "export_qrels", "per_query_metrics",
    "aggregate_tools", "aggregate_task_metrics", "evaluate_task_contract",
    "task_success",
]
