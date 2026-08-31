"""
debug_logger.py — 结构化调试日志系统
==============
JSON Lines 格式，流式记录 Agent 完整执行生命周期。

每条日志记录包含：timestamp、phase、event_type、data。
日志文件命名：logs/query_{YYYYMMDD_HHMMSS}_{question_hash8}.jsonl

事件类型覆盖：
  query_start, intent_detected, query_rewritten, plan_generated,
  iteration_start, tool_selected, tool_invoked, tool_result_summarized,
  step_completed, react_tool_chain, completeness_eval, supplement_generated,
  answer_synthesized, reflection_start, reflection_verdict, correction_applied,
  loop_result, error

使用方式：
  from agent.debug_logger import DebugLogger
  logger = DebugLogger(question)
  logger.log("intent_detected", phase="planning", data={"intent_type": "knowledge_retrieval"})
  # ... 最后
  log_path = logger.get_log_path()
"""

from __future__ import annotations

import json
import hashlib
import traceback as _traceback
from datetime import datetime
from pathlib import Path
from typing import Any


class DebugLogger:
    """结构化 JSON Lines 调试日志器。

    每个查询创建一个实例，写入独立的 .jsonl 文件。
    线程安全：单线程 Agent 循环，无需加锁。
    """

    def __init__(self, question: str = "", log_dir: str = "logs",
                 resume_path: str | None = None):
        self._question = question
        self._question_hash = hashlib.md5(
            question.encode("utf-8")
        ).hexdigest()[:8] if question else "resume"
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        if resume_path:
            # 恢复模式：追加到已有日志文件
            self._log_path = Path(resume_path)
            self._file = open(self._log_path, "a", encoding="utf-8")
            # 读取已有事件数作为起始 seq
            self._event_count = self._count_existing_events()
        else:
            # 新建模式
            self._log_path = (
                self._log_dir
                / f"query_{self._timestamp}_{self._question_hash}.jsonl"
            )
            self._file = open(self._log_path, "w", encoding="utf-8")
            self._event_count = 0

            # 写入查询开始事件
            if question:
                self.log("query_start", phase="init", data={
                    "question": question[:500],
                    "question_hash": self._question_hash,
                })

    def _count_existing_events(self) -> int:
        """读取已有日志文件中的事件数量（用于恢复时设置 seq 起点）。"""
        count = 0
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            pass
        return count

    # ── 公共 API ─────────────────────────────────────────────

    def log(
        self,
        event_type: str,
        phase: str = "",
        data: dict[str, Any] | None = None,
    ):
        """写入一条日志事件。

        Args:
            event_type: 事件类型标识（如 "tool_invoked", "reflection_verdict"）
            phase:     当前阶段（planning/executing/reflecting/responding/waiting）
            data:      事件携带的数据字典
        """
        self._event_count += 1
        record = {
            "seq": self._event_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "phase": phase,
            "event": event_type,
            "data": data or {},
        }
        # 写入一行 JSON + 换行符
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_error(
        self,
        error_type: str,
        message: str,
        phase: str = "",
        exc_info: bool = False,
    ):
        """写入一条错误事件（自动附加 traceback）。"""
        data: dict[str, Any] = {
            "error_type": error_type,
            "message": message[:1000],
        }
        if exc_info:
            data["traceback"] = _traceback.format_exc()[-2000:]
        self.log("error", phase=phase, data=data)

    def log_phase_start(self, phase: str, message: str = ""):
        """记录阶段开始（便捷方法）。"""
        self.log(f"{phase}_start", phase=phase, data={"message": message})

    def get_log_path(self) -> str:
        """返回日志文件绝对路径（供 UI 引用）。"""
        return str(self._log_path.resolve())

    def close(self):
        """关闭日志文件句柄。"""
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()

    @staticmethod
    def cleanup_old_logs(log_dir: str = "logs", max_files: int = 50):
        """清理旧日志文件，保留最近 max_files 个。"""
        log_path = Path(log_dir)
        if not log_path.exists():
            return
        files = sorted(
            log_path.glob("query_*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for old_file in files[max_files:]:
            try:
                old_file.unlink()
            except OSError:
                pass
