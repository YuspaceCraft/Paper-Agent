"""
config.py — 评测配置（env 优先，供 runner / judge / trace 采集读取）。

所有配置都有环境变量覆盖，OVERRIDE 链：代码默认 < env < 显式传参。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip()) if os.getenv(name, "") else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip()) if os.getenv(name, "") else default
    except ValueError:
        return default


# 每 1k token 成本（USD），估算 cost_usd 用；EVAL_PRICE_<MODEL>=in,out 可覆盖
_PRICES: dict = {
    "qwen-plus": (0.0008, 0.002),
    "qwen-max": (0.0016, 0.0064),
    "kimi-k2.6": (0.0006, 0.0015),
    "default": (0.001, 0.002),
}


@dataclass
class EvalConfig:
    """评测跑批与 judge 的配置。字段均可被 env / 调用方 override。"""

    # ---- trace 采集 ----
    trace_enabled: bool = True            # AGENT_TRACE_ENABLED != "0"
    trace_db: Path = PROJECT_ROOT / "trace_store.db"

    # ---- judge（LLM-as-judge 采样评审）----
    judge_model: str = "kimi-k2.6"        # EVAL_JUDGE_MODEL
    judge_sample_size: int = 10           # EVAL_JUDGE_SAMPLE
    judge_budget_usd: float = 0.5         # EVAL_JUDGE_BUDGET_USD，累计即停

    # ---- runner ----
    max_queries: int = 0                  # 0 = 全部；EVAL_MAX_QUERIES
    concurrency: int = 1                  # EVAL_CONCURRENCY（>1 时 thread_id 加后缀）
    turn_timeout: float = 900.0           # 单条 QA 超时护栏（沿用 agent.TURN_TIMEOUT）
    run_token_budget: int = 0             # 0 = 不限；EVAL_RUN_TOKEN_BUDGET

    # ---- 进度心跳 / 终态收敛 ----
    # 长 query（分钟级）期间周期性刷新 eval_runs.metadata.heartbeat；读路径据此
    # 把「进程已消失仍挂在 running」的行收敛成 interrupted。
    progress_heartbeat_s: float = 20.0    # EVAL_PROGRESS_HEARTBEAT_S
    stale_run_after_s: float = 180.0      # EVAL_STALE_RUN_S（0 = 不收敛）

    run_dir: Path = PROJECT_ROOT / "eval_output" / "runs"
    dataset_dir: Path = PROJECT_ROOT / "eval_output" / "datasets"

    prices: dict = field(default_factory=lambda: dict(_PRICES))

    @classmethod
    def from_env(cls, **overrides) -> "EvalConfig":
        prices = dict(_PRICES)
        envp = os.getenv("EVAL_PRICES", "")
        if envp:
            for segment in envp.split(";"):
                parts = segment.split(",")
                if len(parts) != 3:
                    continue
                name, in_p, out_p = parts
                prices[name.strip()] = (float(in_p), float(out_p))
        cfg = cls(
            trace_enabled=os.getenv("AGENT_TRACE_ENABLED", "1") != "0",
            judge_model=os.getenv("EVAL_JUDGE_MODEL", cls.judge_model),
            judge_sample_size=_env_int("EVAL_JUDGE_SAMPLE", cls.judge_sample_size),
            judge_budget_usd=_env_float("EVAL_JUDGE_BUDGET_USD", cls.judge_budget_usd),
            max_queries=_env_int("EVAL_MAX_QUERIES", cls.max_queries),
            concurrency=_env_int("EVAL_CONCURRENCY", cls.concurrency),
            turn_timeout=_env_float("EVAL_TURN_TIMEOUT", cls.turn_timeout),
            run_token_budget=_env_int("EVAL_RUN_TOKEN_BUDGET", cls.run_token_budget),
            progress_heartbeat_s=_env_float("EVAL_PROGRESS_HEARTBEAT_S",
                                            cls.progress_heartbeat_s),
            stale_run_after_s=_env_float("EVAL_STALE_RUN_S",
                                         cls.stale_run_after_s),
            prices=prices,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg
