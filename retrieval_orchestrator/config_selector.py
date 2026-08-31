"""
config_selector.py — 最优配置选择器
===================================

1. Filter configs by business thresholds (Recall@10 >= 0.85, MRR >= 0.7, etc.)
2. Rank by performance-cost tradeoff (prefer smaller Top-K)
3. Quality gate: Recall@5 < 0.6 → BLOCKING ERROR
4. Regression check vs historical best
5. Output optimal_retrieval_config.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class QualityGateError(RuntimeError):
    """Recall@5 below 0.6 — blocking error, no config can be selected."""
    pass


class RegressionWarning(Warning):
    """Key metric degraded >5% vs historical best."""
    pass


def select_optimal(
    experiment_reports: list[dict],
    thresholds: dict | None = None,
    historical_best_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict | None:
    """Select the optimal retrieval configuration.

    Args:
        experiment_reports: List of {experiment_id, config, evaluation_report, results_path}
        thresholds: Dict of metric→threshold. Default: {"recall@10": 0.85, "mrr": 0.7}
        historical_best_path: Path to previous best report (for regression check)
        output_path: Where to write optimal_retrieval_config.yaml

    Returns:
        Selected config dict, or None if none qualify.
    """
    if thresholds is None:
        thresholds = {"recall@10": 0.85, "mrr": 0.7}

    # ---- Filter by business thresholds ----
    qualified = []
    for exp in experiment_reports:
        report = exp.get("evaluation_report", {})
        overall = report.get("overall_metrics", {})
        passes = True
        for metric, threshold in thresholds.items():
            if overall.get(metric, 0) < threshold:
                passes = False
                break
        if passes:
            qualified.append(exp)

    if not qualified:
        print("  [SELECTOR] No configuration met all thresholds:")
        for metric, threshold in thresholds.items():
            best = max(
                (e.get("evaluation_report", {}).get("overall_metrics", {}).get(metric, 0)
                 for e in experiment_reports),
                default=0,
            )
            print(f"    {metric}: best={best:.4f}, required={threshold}")
        return None

    print(f"  [SELECTOR] {len(qualified)}/{len(experiment_reports)} configs qualified")

    # ---- Rank: prefer smaller Top-K, then higher MRR ----
    def _rank_key(exp: dict) -> tuple:
        cfg = exp.get("config", {})
        overall = exp.get("evaluation_report", {}).get("overall_metrics", {})
        top_k = cfg.get("top_k", 999)
        mrr = overall.get("mrr", 0)
        r10 = overall.get("recall@10", 0)
        # prefer small top_k, high mrr, high recall
        return (top_k, -mrr, -r10)

    qualified.sort(key=_rank_key)
    best = qualified[0]

    # ---- Quality Gate: selected config must meet min Recall@5 ----
    best_r5 = best.get("evaluation_report", {}).get("overall_metrics", {}).get("recall@5", 0)
    if best_r5 < 0.6:
        raise QualityGateError(
            f"Recall@5 = {best_r5:.2%} for best config '{best['experiment_id']}'. "
            f"Threshold is 0.6. Check upstream chunking/embedding quality."
        )

    # ---- Regression check ----
    if historical_best_path and Path(historical_best_path).exists():
        hist = json.loads(Path(historical_best_path).read_text(encoding="utf-8"))
        hist_metrics = hist.get("overall_metrics", {})
        curr_metrics = best.get("evaluation_report", {}).get("overall_metrics", {})

        regressions = []
        for metric in ["recall@5", "recall@10", "mrr", "ndcg@10"]:
            prev = hist_metrics.get(metric, 0)
            curr = curr_metrics.get(metric, 0)
            if prev > 0 and curr < prev * 0.95:  # >5% drop
                regressions.append(f"{metric}: {prev:.4f} → {curr:.4f} ({(curr-prev)/prev:.1%})")

        if regressions:
            msg = "REGRESSION DETECTED:\n  " + "\n  ".join(regressions)
            print(f"  [SELECTOR] ⚠️  {msg}")
            best["regression_warnings"] = regressions

    # ---- Output ----
    if output_path:
        import yaml
        optimal_cfg = {
            "schema_version": "1.0",
            "selected_at": _now_iso(),
            "source_experiment": best["experiment_id"],
            "metrics": best.get("evaluation_report", {}).get("overall_metrics", {}),
            "config": best.get("config", {}),
            "regression_warnings": best.get("regression_warnings", []),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.dump(optimal_cfg, allow_unicode=True, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        print(f"  [SELECTOR] Optimal config written to {output_path}")

    return best


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
