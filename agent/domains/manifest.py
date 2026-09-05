"""
manifest.py — 实验项目 manifest（对话中心化重构 L4：委托信息交换契约）。

每个实验项目一个 `{experiments_root}/{project}/project.json`，作为主 agent、
外部 coding agent、UI 三方共享的**持久项目契约**：

    project        项目 slug
    paper          关联论文（文献↔实验连通的关键字段）
    entry          {run, data, config} 运行入口
    key_files      关键文件清单
    metrics_schema 指标语义（lower_is_better 等）
    baseline       换代比对锚（{exp_id, metrics}）
    status         draft | running | done
    last_run       最近一次实验 exp_id
    last_delegate  最近一次外部委托摘要
    changed_files  最近一次委托产生的改动文件
    changelog      [{kind, summary, at}] 追加型事件日志
    last_commit_sha 最近一次 git 提交

设计约束：
- 纯文件操作；**不 import agent/domains/coding.py**（coding.py 会 import 本模块，
  双向依赖）。slug 规则与 workspace_config 一致地内联复用。
- 零状态/机器可解析：delegate prompt 携带 manifest → coding agent 知道入口/关键
  文件/结果上报格式；run_experiment/git_commit/delegate_code_task 回写。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

from .. import workspace_config

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")

DEFAULT_MANIFEST = {
    "project": "",
    "paper": "",
    "description": "",
    "entry": {"run": "", "data": "", "config": ""},
    "key_files": [],
    "metrics_schema": {},
    "baseline": {},
    "status": "draft",
    "last_run": "",
    "last_delegate": "",
    "changed_files": [],
    "changelog": [],
    "last_commit_sha": "",
}


def _slug(name: str) -> str:
    return _SLUG_RE.sub("_", (name or "").strip()) or "default"


def _root() -> Path:
    return workspace_config.get_experiments_path()


def manifest_path(project: str) -> Path:
    """项目 manifest 文件路径（不创建）。project 经安全 slug。"""
    return (_root() / _slug(project) / "project.json").resolve()


def _default(project: str) -> dict:
    return {**DEFAULT_MANIFEST, "project": _slug(project)}


def load_manifest(project: str) -> dict:
    """读 manifest；缺失返回默认结构（不建文件）。"""
    p = manifest_path(project)
    if not p.exists():
        return _default(project)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default(project)
        return {**_default(project), **data}
    except (ValueError, OSError):
        return _default(project)


def ensure_manifest(project: str) -> dict:
    """缺失时创建默认 manifest 并落盘；返回当前 manifest。"""
    p = manifest_path(project)
    if p.exists():
        return load_manifest(project)
    data = _default(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(p, data)
    return data


def update_manifest(project: str, **patch) -> dict:
    """合并 patch 并原子落盘；返回更新后 manifest。任何值替换（除 changelog）。"""
    data = ensure_manifest(project)
    data.update({k: v for k, v in patch.items() if v is not None})
    # changelog 追加型：保留而非替换
    if patch.get("changelog"):
        data["changelog"] = (data.get("changelog") or []) + patch["changelog"]
    data["changelog"] = data["changelog"][-100:]
    _write_atomic(manifest_path(project), data)
    return data


def log_event(project: str, kind: str, summary: str) -> dict:
    """追加一条 changelog（时间戳自动填）。"""
    return update_manifest(project, changelog=[{
        "kind": kind, "summary": str(summary)[:400], "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }])


def _write_atomic(p: Path, data: dict) -> None:
    """原子写（临时文件 + os.replace），防写到一半崩溃留下半截。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise