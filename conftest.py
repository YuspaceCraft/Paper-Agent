"""Shared pytest isolation for the local agent and evaluation suites."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# Disable external tracing before test modules import langchain/langgraph.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["AGENT_TRACE_ENABLED"] = "0"
# Unit tests exercise retry/idempotency mechanics without a live LangGraph
# approval context. Approval-specific tests opt in with monkeypatch.setenv().
os.environ["AGENT_TOOL_APPROVAL"] = "0"


@pytest.fixture(autouse=True)
def isolate_agent_runtime(tmp_path: Path):
    """Keep persisted workspace settings and process metrics out of unit tests."""
    from agent import config_store, observability, workspace_config
    from evaluation import report as evaluation_report

    code_root = Path(__file__).resolve().parent
    baseline_path = evaluation_report.BASELINE_PATH
    evaluation_report.BASELINE_PATH = tmp_path / ".baseline.json"
    workspace_config.clear_overrides()
    workspace_config.set_override("project_root", code_root)
    config_store.clear_overrides()
    observability.reset_metrics()
    try:
        yield
    finally:
        evaluation_report.BASELINE_PATH = baseline_path
        workspace_config.clear_overrides()
        config_store.clear_overrides()
        observability.reset_metrics()


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Compatibility fixture used by assert-style agent test modules."""
    return tmp_path
