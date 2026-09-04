"""(v10 / Phase C) self-check — coding domain: ExperimentStore / metrics / study / delegate.

Run: python agent/tests/test_coding.py
ponytail: assert-based, no framework, no LLM calls. Real subprocesses only for
tiny 0.2s scripts; workspace roots are isolated to a temp dir.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.domains import coding  # noqa: E402

# run_experiment 是后台任务：必须让后台 _watch task 跑在同一个持续事件循环上。
# asyncio.run() 每次建新循环、函数返回即关闭 → 后台 task 被杀（真实 uvicorn
# 循环常驻无此问题）。测试用共享 loop。
_loop = asyncio.new_event_loop()


def R(coro):
    return _loop.run_until_complete(coro)


def _wait_exp_idle(exp_id: str, rounds: int = 60) -> dict:
    """轮询直到 exp 进入终态（期间让出 loop 使后台 task/子进程运行）。"""
    for _ in range(rounds):
        stj = json.loads(R(coding.experiment_status.ainvoke({"exp_id": exp_id})))
        assert stj["ok"] is True, stj
        if stj["data"]["status"] in ("done", "failed"):
            return stj["data"]
        R(asyncio.sleep(0.2))
    raise AssertionError("experiment did not reach a terminal state")


def _setup(tmp_root: Path):
    # 实验/研究根走 workspace_config override（隔离到临时目录）
    from agent import workspace_config as wc
    wc.set_override("experiments_path", tmp_root / "experiments")
    wc.set_override("study_root", tmp_root / "studies")


def test_run_experiment_lifecycle(tmp_root: Path):
    """run → 后台完成 → done + metrics 解析 + study 自动归档。"""
    _setup(tmp_root)
    cmd = (
        "python -c \"import json,time; time.sleep(0.2); "
        "print('hello experiment'); "
        "json.dump({'acc': 0.93, 'loss': 0.11}, open('metrics.json','w'))\""
    )
    payload = json.loads(R(coding.run_experiment.ainvoke(
        {"project": "demo", "command": cmd, "name": "smoke"})))
    assert payload["ok"] is True, payload
    exp_id = payload["data"]["exp_id"]

    st = _wait_exp_idle(exp_id)
    assert "hello experiment" in st["log_tail"]
    assert st["metrics"]["acc"] == 0.93, st["metrics"]
    assert st["exit_code"] == 0

    # study 归档存在
    study = coding.load_study("demo")
    assert any(e["exp_id"] == exp_id for e in study["experiments"]), study
    rec = next(e for e in study["experiments"] if e["exp_id"] == exp_id)
    assert rec["metric_summary"]["acc"] == 0.93


def test_read_metrics_and_list(tmp_root: Path):
    _setup(tmp_root)
    cmd = "python -c \"import csv; open('metrics.csv','w').write('epoch,acc\\n1,0.5\\n2,0.7\\n')\""
    exp_id = json.loads(R(coding.run_experiment.ainvoke(
        {"project": "demo", "command": cmd})))["data"]["exp_id"]
    _wait_exp_idle(exp_id)
    raw = json.loads(R(coding.read_metrics.ainvoke({"exp_id": exp_id, "metric_key": "acc"})))
    assert raw["ok"] and raw["data"]["value"] == 0.7, raw  # CSV 取最后一行

    lst = json.loads(R(coding.experiment_list.ainvoke({"project": "demo"})))
    assert any(e["exp_id"] == exp_id for e in lst["data"]["experiments"])

    # 不存在的 metric → 结构化错误
    bad = json.loads(R(coding.read_metrics.ainvoke({"exp_id": exp_id, "metric_key": "nope"})))
    assert bad["ok"] is False and bad["error_type"] == "param_error"


def test_delegate_no_backend_structured_error(tmp_root: Path):
    """无 AGENT_CODING_CMD 且无 claude/codex 时 → ok:false 信封（不 raise）。"""
    _setup(tmp_root)
    old_cmd, old_path = os.environ.get("AGENT_CODING_CMD"), os.environ.get("PATH")
    os.environ["AGENT_CODING_CMD"] = ""
    os.environ["PATH"] = ""
    try:
        raw = json.loads(R(coding.delegate_code_task.ainvoke(
            {"project": "demo", "prompt": "hello"})))
        assert raw["ok"] is False
        assert "no coding backend" in raw["error"]
    finally:
        if old_cmd is None:
            os.environ.pop("AGENT_CODING_CMD", None)
        else:
            os.environ["AGENT_CODING_CMD"] = old_cmd
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path


def test_safe_project_blocks_escape(tmp_root: Path):
    _setup(tmp_root)
    # "/" 被 slug 化（点号保留），因此可能含 "." 但绝无路径分隔符/穿越段
    slugged = coding._safe_project("../../etc")
    assert "/" not in slugged and "\\" not in slugged
    assert slugged.replace(".", "x") == "xx_xx_etc"  # 结构等价于 ../../
    # 路径限定：project 无论如何必须落在 EXPERIMENTS_ROOT 内（resolve 兜底）
    from agent.domains.coding import _project_dir
    d = _project_dir("../../evil")
    assert d.is_relative_to(coding._experiments_root())
    assert "/" not in d.name and "\\" not in d.name


def test_study_hypothesis_append(tmp_root: Path):
    _setup(tmp_root)
    raw = json.loads(R(coding.study_add_hypothesis.ainvoke(
        {"topic": "demo", "hypothesis": "数据增强应提升鲁棒性"})))
    assert raw["ok"]
    study = coding.load_study("demo")
    assert any(h["text"].startswith("数据增强") for h in study["hypotheses"])
    ctx = json.loads(R(coding.study_context.ainvoke({"topic": "demo"})))
    assert ctx["data"]["topic"] == "demo"


def test_parse_steps_accepts_coder_target():
    from agent import plan as plan_mod

    parsed = plan_mod._parse_steps(
        '{"steps": [{"id": "code-1", "description": "Run demo train.py",'
        ' "target": "coder", "args": {"project": "demo", "goal": "train"},'
        ' "depends_on": []}]}'
    )
    assert parsed and parsed[0]["target"] == "coder"
    assert parsed[0]["args"]["project"] == "demo"


def test_coding_plan_calls_coder():
    """_coding_plan: fake LLM 产 coder 步骤 → 透传 plan（不建 doc）。"""
    from agent import plan as plan_mod

    class _FakeModel:
        def __init__(self, reply: str):
            self.reply = reply

        async def ainvoke(self, msgs):
            class R:
                content = self.reply
                tool_calls = None
            return R()

    reply = ('{"steps": [{"id": "code-1", "description": "Run demo train.py",'
             ' "target": "coder", "args": {"project": "demo", "goal": "train"},'
             ' "depends_on": []}]}')
    out = asyncio.run(plan_mod._coding_plan(
        _FakeModel(reply), "跑一下 train.py", "(none)", "(no resolved hints)"))
    assert out["mode"] == "plan"
    assert out["plan"][0]["target"] == "coder"


def test_git_tools_when_available(tmp_root: Path):
    """git 仓库可用时 commit/diff 正常；无仓库 → 结构化错误。"""
    _setup(tmp_root)
    proj = coding._experiments_root() / "demogit"
    proj.mkdir(parents=True, exist_ok=True)
    if not shutil.which("git"):
        return
    subprocess.run(["git", "-C", str(proj), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(proj), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(proj), "config", "user.name", "t"], check=True)
    (proj / "run.py").write_text("print(1)\n", encoding="utf-8")
    r = json.loads(R(coding.git_commit.ainvoke({"project": "demogit", "message": "init"})))
    assert r["ok"] is True, r
    assert r["data"]["sha"]
    (proj / "run.py").write_text("print(2)\n", encoding="utf-8")
    r2 = json.loads(R(coding.git_diff.ainvoke({"project": "demogit"})))
    assert r2["ok"] is True and "+print(2)" in r2["data"]["output"]

    r3 = json.loads(R(coding.git_status.ainvoke({"project": "not_a_repo"})))
    assert r3["ok"] is False and r3["error_type"] == "param_error"


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp())
    try:
        test_run_experiment_lifecycle(root)
        test_read_metrics_and_list(root)
        test_delegate_no_backend_structured_error(root)
        test_safe_project_blocks_escape(root)
        test_study_hypothesis_append(root)
        test_git_tools_when_available(root)
        test_parse_steps_accepts_coder_target()
        test_coding_plan_calls_coder()
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _loop.close()
    print("Phase C coding self-check OK")