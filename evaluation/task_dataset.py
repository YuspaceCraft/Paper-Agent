"""Build the agent task benchmark for RAG and paper operations.

The legacy retrieval manifest models one question over a small set of ground
truth chunks.  This module builds a broader, task-oriented manifest while
remaining backward compatible with ``evaluation.datasets.load_manifest``.

The v3 benchmark is a balanced 4 x 3 x 3 matrix:

* task family: RAG retrieval, paper download, paper read/precision, paper ingest
* difficulty: easy, medium, hard
* task length: short, medium, long

Every row carries machine-checkable expectations, budgets and metric specs.
The generated JSONL is the source of truth for a run; this module exists to
make the matrix reproducible and easy to extend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT

DATASET_ID = "agent-rag-paperops-v3"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "eval_output" / "datasets"
    / "agent_rag_paper_ops_v3.jsonl"
)
CHUNKS_PATH = PROJECT_ROOT / "eval_output" / "all_rag_chunks.json"

ARTIFACT_ROOT = "eval_output/agent_eval_artifacts/v3"

TASK_TYPES = ("rag_retrieval", "paper_download", "paper_read", "paper_ingest")
DIFFICULTIES = ("easy", "medium", "hard")
TASK_LENGTHS = ("short", "medium", "long")

_LENGTH_BUDGETS = {
    "short": {"timeout_s": 90, "max_steps": 8, "max_tool_calls": 3,
              "max_tokens": 12_000},
    "medium": {"timeout_s": 180, "max_steps": 16, "max_tool_calls": 6,
               "max_tokens": 24_000},
    "long": {"timeout_s": 420, "max_steps": 30, "max_tool_calls": 12,
             "max_tokens": 48_000},
}

_METRICS = {
    "rag_retrieval": {
        "retrieval": [
            "recall@5", "recall@10", "precision@5", "mrr", "ndcg@10",
            "hit@5", "context_precision",
        ],
        "task": ["success", "tool_success_rate", "step_count"],
        "cost": [
            "duration_s", "answer_latency_p50_s", "answer_latency_p95_s",
            "prompt_tokens", "completion_tokens", "tokens_total", "cost_usd",
        ],
    },
    "paper_download": {
        "task": [
            "success", "tool_success_rate", "expected_tool_coverage",
            "artifact_correctness", "path_correctness", "no_accidental_ingest",
            "idempotency", "retry_count",
        ],
        "cost": [
            "duration_s", "metadata_ms", "download_ms", "write_ms",
            "prompt_tokens", "completion_tokens", "tokens_total", "cost_usd",
        ],
    },
    "paper_read": {
        "retrieval": [
            "recall@5", "precision@5", "mrr", "ndcg@10", "context_precision",
        ],
        "task": [
            "success", "field_coverage", "exact_numeric_match",
            "citation_precision", "citation_recall", "unsupported_claim_rate",
        ],
        "cost": [
            "duration_s", "prompt_tokens", "completion_tokens", "tokens_total",
            "cost_usd",
        ],
    },
    "paper_ingest": {
        "task": [
            "success", "tool_success_rate", "enqueue_success",
            "end_to_end_completion", "indexed_state", "chunk_count",
            "duplicate_index_rate", "rollback_on_failure", "idempotency",
        ],
        "cost": [
            "duration_s", "queue_wait_s", "parse_s", "index_s",
            "prompt_tokens", "completion_tokens", "tokens_total", "cost_usd",
        ],
    },
}


def _case(
    case_id: str,
    task_type: str,
    difficulty: str,
    task_length: str,
    query: str,
    *,
    ground_truth_ids: Iterable[str] = (),
    required_tools: Iterable[str] = (),
    forbidden_tools: Iterable[str] = (),
    tool_args: Iterable[dict] | None = None,
    answer_contains: Iterable[str] = (),
    answer_not_contains: Iterable[str] = (),
    artifacts: Iterable[dict] | None = None,
    postconditions: Iterable[dict] | None = None,
    setup: dict | None = None,
    budgets: dict | None = None,
    tags: Iterable[str] = (),
    expected_outcome: str = "succeeded",
) -> dict[str, Any]:
    """Create one manifest row with explicit, machine-checkable expectations."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty: {difficulty}")
    if task_length not in TASK_LENGTHS:
        raise ValueError(f"unknown task_length: {task_length}")

    gt = list(dict.fromkeys(ground_truth_ids))
    retrieval_enabled = task_type in {"rag_retrieval", "paper_read"}
    if retrieval_enabled and not gt:
        raise ValueError(f"{case_id}: retrieval task requires ground_truth_ids")
    if not retrieval_enabled and gt:
        raise ValueError(f"{case_id}: non-retrieval task cannot define ground_truth_ids")

    effective_budgets = {**_LENGTH_BUDGETS[task_length], **(budgets or {})}
    setup_payload = {
        "isolated_workspace": task_type in {"paper_download", "paper_ingest"},
        "network": task_type in {"paper_download", "paper_ingest"},
        "mutates_state": task_type == "paper_ingest" or any(
            str(item.get("path", "")).startswith(ARTIFACT_ROOT)
            for item in (artifacts or [])
        ),
        **(setup or {}),
    }
    expected = {
        "outcome": expected_outcome,
        "required_tools": list(dict.fromkeys(required_tools)),
        "forbidden_tools": list(dict.fromkeys(forbidden_tools)),
        "tool_args": list(tool_args or []),
        "answer_contains": list(answer_contains),
        "answer_not_contains": list(answer_not_contains),
        "artifacts": list(artifacts or []),
    }
    return {
        "schema_version": "3.0",
        "dataset_id": DATASET_ID,
        "id": case_id,
        "query": query,
        "ground_truth_ids": gt,
        "metadata_filters": {},
        "difficulty_level": difficulty,
        "generation_mode": "handcrafted_task",
        "task_type": task_type,
        "task_length": task_length,
        "tags": list(dict.fromkeys([task_type, difficulty, task_length, *tags])),
        "evaluation": {
            "retrieval": retrieval_enabled,
            "task_execution": True,
            "paper_precision": task_type == "paper_read",
            "side_effects": task_type in {"paper_download", "paper_ingest"},
        },
        "expected": expected,
        "budgets": effective_budgets,
        "metrics": _METRICS[task_type],
        "setup": setup_payload,
        "postconditions": list(postconditions or []),
        "review": {
            "status": "draft",
            "reviewer": "",
            "required": True,
        },
    }


def _download_case(
    case_id: str,
    difficulty: str,
    task_length: str,
    query: str,
    *,
    artifacts: list[dict],
    tool_args: list[dict],
    required_tools: list[str] | None = None,
    answer_contains: Iterable[str] = (),
    setup: dict | None = None,
    tags: Iterable[str] = (),
    budgets: dict | None = None,
) -> dict[str, Any]:
    return _case(
        case_id, "paper_download", difficulty, task_length, query,
        required_tools=required_tools or ["ingest", "download_paper"],
        forbidden_tools=["ingest_paper"],
        tool_args=tool_args,
        answer_contains=answer_contains,
        artifacts=artifacts,
        setup={
            "clean_paths": [item["path"] for item in artifacts],
            **(setup or {}),
        },
        tags=tags,
        budgets=budgets,
    )


def _ingest_case(
    case_id: str,
    difficulty: str,
    task_length: str,
    query: str,
    *,
    paper_name: str,
    pdf_path: str,
    required_tools: list[str] | None = None,
    tool_args: list[dict] | None = None,
    artifacts: list[dict] | None = None,
    postconditions: list[dict] | None = None,
    setup: dict | None = None,
    answer_contains: Iterable[str] = (),
    tags: Iterable[str] = (),
    budgets: dict | None = None,
    allow_download: bool = False,
) -> dict[str, Any]:
    required = required_tools or ["ingest", "ingest_paper"]
    if allow_download and "download_paper" not in required:
        required = [*required, "download_paper"]
    return _case(
        case_id, "paper_ingest", difficulty, task_length, query,
        required_tools=required,
        forbidden_tools=[] if allow_download else ["download_paper"],
        tool_args=tool_args or [],
        answer_contains=answer_contains,
        artifacts=artifacts or [],
        postconditions=postconditions or [{
            "kind": "paper_state",
            "paper_name": paper_name,
            "allowed": ["indexed"],
            "poll_timeout_s": 180,
        }],
        setup={
            "paper_name": paper_name,
            "pdf_path": pdf_path,
            "requires_isolated_backend": True,
            **(setup or {}),
        },
        tags=tags,
        budgets=budgets,
    )


def build_cases() -> list[dict[str, Any]]:
    """Return the complete, deterministic v3 task matrix."""
    cases: list[dict[str, Any]] = [
        # ---- RAG retrieval -------------------------------------------------
        _case(
            "rag_easy_short_attention_bleu",
            "rag_retrieval", "easy", "short",
            "Attention Is All You Need 在 WMT 2014 英德翻译任务上报告的 BLEU 是多少？",
            ground_truth_ids=["Attention__chunk_0002"],
            required_tools=["search_papers"],
            answer_contains=["28.4"],
            tags=["factual", "en", "zh"],
        ),
        _case(
            "rag_easy_medium_textcaps_scale",
            "rag_retrieval", "easy", "medium",
            "TextCaps 数据集包含多少张图像和多少条 caption？请给出规模。",
            ground_truth_ids=["2003.12462v2__chunk_0006"],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["28,408", "145,329"],
            tags=["dataset-statistics"],
        ),
        _case(
            "rag_easy_long_remote_sensing_survey",
            "rag_retrieval", "easy", "long",
            "从本地知识库中找出至少 5 篇遥感图像变化描述论文，逐篇说明核心方法。",
            ground_truth_ids=[
                "2022_chenyang_liu_remote_dataset__chunk_0000",
                "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote__chunk_0005",
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0005",
                "SAM_Guided_Semantic_and_Motion_Changed_Region_Mining__chunk_0010",
                "Mask_Approximation_Net_A_Novel_Diffusion_Model_Approach_for_Remote_Sensing__chunk_0010",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["RSICCformer", "Diffusion-RSCC", "CTM", "SAGE-CC"],
            tags=["multi-paper", "coverage"],
        ),
        _case(
            "rag_medium_short_ctm_whu_metrics",
            "rag_retrieval", "medium", "short",
            "CTM 在 WHU-CDC 上的 BLEU-4、METEOR 和 CIDEr 分别是多少？",
            ground_truth_ids=[
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0020",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["72.36", "46.98", "153.29"],
            tags=["numeric"],
        ),
        _case(
            "rag_medium_medium_textcaps_vs_coco",
            "rag_retrieval", "medium", "medium",
            "TextCaps 与 COCO 在 OCR token 覆盖上的关键差异是什么？请给出具体百分比。",
            ground_truth_ids=[
                "2003.12462v2__chunk_0009",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["81.3%", "96.9%", "2.7%", "12.7%"],
            tags=["comparison", "numeric"],
        ),
        _case(
            "rag_medium_long_ctm_cross_dataset",
            "rag_retrieval", "medium", "long",
            "比较 CTM 在 LEVIR-CC 与 WHU-CDC 上的结果，分别列出 BLEU-4、METEOR、CIDEr，并指出数据集差异。",
            ground_truth_ids=[
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0017",
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0020",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["65.48", "40.63", "138.78", "72.36", "46.98", "153.29"],
            tags=["comparison", "multi-hop"],
        ),
        _case(
            "rag_hard_short_levircc_conflict",
            "rag_retrieval", "hard", "short",
            "本地论文中 LEVIR-CC 是 10077 对图像还是 1077 对图像？请核对来源并解释冲突。",
            ground_truth_ids=[
                "2022_chenyang_liu_remote_dataset__chunk_0000",
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0011",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["10077", "1077"],
            tags=["contradiction", "source-check"],
        ),
        _case(
            "rag_hard_medium_diffusion_evidence",
            "rag_retrieval", "hard", "medium",
            "基于本地论文，对比 Diffusion-RSCC 与 MaskApproxNet 使用 diffusion 的核心机制和证据。",
            ground_truth_ids=[
                "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote__chunk_0014",
                "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote__chunk_0015",
                "Mask_Approximation_Net_A_Novel_Diffusion_Model_Approach_for_Remote_Sensing__chunk_0010",
                "Mask_Approximation_Net_A_Novel_Diffusion_Model_Approach_for_Remote_Sensing__chunk_0011",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["forward", "reverse", "mask", "denoising"],
            tags=["multi-paper", "mechanism"],
        ),
        _case(
            "rag_hard_long_three_paper_synthesis",
            "rag_retrieval", "hard", "long",
            "撰写一份三篇论文对比摘要：RSICCformer、Diffusion-RSCC、CTM，分别说明数据规模、核心模块和评测结果。",
            ground_truth_ids=[
                "2022_chenyang_liu_remote_dataset__chunk_0000",
                "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote__chunk_0005",
                "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote__chunk_0014",
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0005",
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0017",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["RSICCformer", "Diffusion-RSCC", "CTM"],
            tags=["multi-paper", "synthesis", "long-context"],
        ),

        # ---- Paper download ------------------------------------------------
        _download_case(
            "download_easy_short_attention",
            "easy", "short",
            "下载 arXiv 1706.03762，保存到 eval_output/agent_eval_artifacts/v3/d1，文件名 Attention；只要下载，不要入库。",
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d1/Attention.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "1706.03762"},
                {"tool": "download_paper", "arg": "filename", "equals": "Attention"},
            ],
            answer_contains=["Attention.pdf"],
            tags=["single-file", "arxiv-id"],
        ),
        _download_case(
            "download_easy_medium_textcaps_by_title",
            "easy", "medium",
            "找到 TextCaps 论文并下载到 eval_output/agent_eval_artifacts/v3/d2，文件名 TextCaps；返回实际标题和路径，不要入库。",
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d2/TextCaps.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "2003.12462"},
                {"tool": "download_paper", "arg": "filename", "equals": "TextCaps"},
            ],
            answer_contains=["TextCaps"],
            tags=["title-resolution"],
        ),
        _download_case(
            "download_easy_long_batch_two",
            "easy", "long",
            "批量下载 arXiv 1706.03762 和 2003.12462 到 eval_output/agent_eval_artifacts/v3/d3，文件名分别为 AttentionBatch 和 TextCapsBatch；不要入库。",
            artifacts=[
                {
                    "path": f"{ARTIFACT_ROOT}/d3/AttentionBatch.pdf",
                    "min_size_bytes": 1000,
                    "modified_after_query_start": True,
                },
                {
                    "path": f"{ARTIFACT_ROOT}/d3/TextCapsBatch.pdf",
                    "min_size_bytes": 1000,
                    "modified_after_query_start": True,
                },
            ],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "1706.03762"},
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "2003.12462"},
            ],
            tags=["batch", "two-files"],
        ),
        _download_case(
            "download_medium_short_versioned_rmnet",
            "medium", "short",
            "下载 arXiv 2305.03195v2 到 eval_output/agent_eval_artifacts/v3/d4，文件名 RMNet_v2；不要入库。",
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d4/RMNet_v2.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "2305.03195v2"},
                {"tool": "download_paper", "arg": "filename", "equals": "RMNet_v2"},
            ],
            answer_contains=["2305.03195"],
            tags=["version-suffix", "canonical-id"],
        ),
        _download_case(
            "download_medium_medium_custom_path",
            "medium", "medium",
            "把 1706.03762 下载到 eval_output/agent_eval_artifacts/v3/d5/custom/deep，文件名 AttentionCustom。",
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d5/custom/deep/AttentionCustom.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "destination", "contains": f"{ARTIFACT_ROOT}/d5/custom/deep"},
                {"tool": "download_paper", "arg": "filename", "equals": "AttentionCustom"},
            ],
            tags=["custom-path"],
        ),
        _download_case(
            "download_medium_long_precheck_no_ingest",
            "medium", "long",
            "先检查本地是否已有 2003.12462v2，再仅下载到 eval_output/agent_eval_artifacts/v3/d6，文件名 TextCapsChecked；绝对不要入库。",
            required_tools=["check_paper", "ingest", "download_paper"],
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d6/TextCapsChecked.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "2003.12462v2"},
                {"tool": "download_paper", "arg": "filename", "equals": "TextCapsChecked"},
            ],
            tags=["precheck", "no-ingest"],
        ),
        _download_case(
            "download_hard_short_idempotent_same_path",
            "hard", "short",
            "将 1706.03762 重复安全地下载到 eval_output/agent_eval_artifacts/v3/d7，文件名 AttentionIdempotent，确保同一路径文件完整且不触发入库。",
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d7/AttentionIdempotent.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "1706.03762"},
            ],
            tags=["idempotency", "duplicate-safe"],
        ),
        _download_case(
            "download_hard_medium_versioned_longcot",
            "hard", "medium",
            "核验并下载 arXiv 2502.03373v1 到 eval_output/agent_eval_artifacts/v3/d8，文件名 LongCoTVerified；最终必须报告工具返回的真实标题，不得猜测。",
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/d8/LongCoTVerified.pdf",
                "min_size_bytes": 1000,
                "modified_after_query_start": True,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "arxiv_id", "equals": "2502.03373v1"},
            ],
            answer_contains=["LongCoTVerified.pdf"],
            tags=["identity-verification", "version-suffix"],
        ),
        _download_case(
            "download_hard_long_batch_three",
            "hard", "long",
            "依次下载 1706.03762、2003.12462、2502.03373 到 eval_output/agent_eval_artifacts/v3/d9 的三个不同文件名，汇报每个真实路径；不得入库。",
            artifacts=[
                {
                    "path": f"{ARTIFACT_ROOT}/d9/attention_hard.pdf",
                    "min_size_bytes": 1000,
                    "modified_after_query_start": True,
                },
                {
                    "path": f"{ARTIFACT_ROOT}/d9/textcaps_hard.pdf",
                    "min_size_bytes": 1000,
                    "modified_after_query_start": True,
                },
                {
                    "path": f"{ARTIFACT_ROOT}/d9/longcot_hard.pdf",
                    "min_size_bytes": 1000,
                    "modified_after_query_start": True,
                },
            ],
            tool_args=[
                {"tool": "download_paper", "arg": "filename", "equals": "attention_hard"},
                {"tool": "download_paper", "arg": "filename", "equals": "textcaps_hard"},
                {"tool": "download_paper", "arg": "filename", "equals": "longcot_hard"},
            ],
            tags=["batch", "three-files", "long-context"],
        ),

        # ---- Paper read / precision ---------------------------------------
        _case(
            "read_easy_short_attention_training",
            "paper_read", "easy", "short",
            "Attention Is All You Need 的训练用了多长时间、多少块 GPU？",
            ground_truth_ids=["Attention__chunk_0002"],
            required_tools=["fetch_content"],
            answer_contains=["3.5", "GPU"],
            tags=["numeric"],
        ),
        _case(
            "read_easy_medium_textcaps_splits",
            "paper_read", "easy", "medium",
            "TextCaps 的训练、验证、测试图像 split 分别是多少？",
            ground_truth_ids=["2003.12462v2__chunk_0006"],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["21,953", "3,166", "3,289"],
            tags=["structured-extraction"],
        ),
        _case(
            "read_easy_long_medqa_fields",
            "paper_read", "easy", "long",
            "从 llm_medqa 中提取 5 个结构化字段：任务、方法模块、最佳整体指标、模型规模消融结论、主要结论。",
            ground_truth_ids=[
                "llm_medqa__chunk_0000",
                "llm_medqa__chunk_0008",
                "llm_medqa__chunk_0023",
                "llm_medqa__chunk_0024",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["MedQA", "case", "77%", "70B"],
            tags=["structured-extraction", "long-context"],
        ),
        _case(
            "read_medium_short_medqa_final_accuracy",
            "paper_read", "medium", "short",
            "llm_medqa 最终报告的最佳准确率约是多少？对比方法约是多少？",
            ground_truth_ids=["llm_medqa__chunk_0024"],
            required_tools=["fetch_content"],
            answer_contains=["77%", "70%"],
            tags=["numeric", "comparison"],
        ),
        _case(
            "read_medium_medium_ctm_clip_backbone",
            "paper_read", "medium", "medium",
            "CTM 使用 Pretrained-CLIP backbone 时 CIDEr 和 BLEU-4 是多少？与其他 backbone 相比结论是什么？",
            ground_truth_ids=[
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0017",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["139.56", "66.21", "CIDEr", "BLEU-4"],
            tags=["numeric", "backbone"],
        ),
        _case(
            "read_medium_long_textcaps_six_stats",
            "paper_read", "medium", "long",
            "提取 TextCaps 的 6 个关键统计数据：图像数、caption 数、三类 split、OCR 覆盖率、unique OCR tokens、zero-shot tokens。",
            ground_truth_ids=[
                "2003.12462v2__chunk_0006",
                "2003.12462v2__chunk_0009",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["28,408", "145,329", "39.7k", "2901"],
            tags=["structured-extraction", "long-context"],
        ),
        _case(
            "read_hard_short_diffusion_table_grounding",
            "paper_read", "hard", "short",
            "Diffusion-RSCC 表格摘要中 BLEU-4 范围上限 60.90 对应的具体方法是什么？如果证据不足必须明确说不能确定。",
            ground_truth_ids=[
                "Diffusion-RSCC_Diffusion_Probabilistic_Model_for_Change_Captioning_in_Remote__chunk_0028",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["60.90", "不能确定"],
            tags=["abstention", "grounding", "table"],
        ),
        _case(
            "read_hard_medium_ctm_conflicting_scores",
            "paper_read", "hard", "medium",
            "核对 CTM 的 CIDEr 是 139.56 还是 153.29，并说明两个数值各自对应的数据集或实验条件。",
            ground_truth_ids=[
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0017",
                "Cross-Temporal_Remote_Sensing_Image_Change_Captioning_A_Manifold_Mapping_and__chunk_0020",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["139.56", "153.29", "CLIP", "WHU-CDC"],
            tags=["source-check", "numeric"],
        ),
        _case(
            "read_hard_long_positional_encoding_compare",
            "paper_read", "hard", "long",
            "对比 Attention Is All You Need 与 positional_encodings_nonessential，说明 positional encoding 是否必需、成立条件和证据。",
            ground_truth_ids=[
                "Attention__chunk_0002",
                "positional_encodings_nonessential__chunk_0002",
                "positional_encodings_nonessential__chunk_0006",
                "positional_encodings_nonessential__chunk_0007",
            ],
            required_tools=["search_papers", "fetch_content"],
            answer_contains=["multi-layer", "one-layer", "position"],
            tags=["multi-paper", "conditional-reasoning"],
        ),

        # ---- Paper ingest --------------------------------------------------
        _ingest_case(
            "ingest_easy_short_textcaps",
            "easy", "short",
            "将 data/downloads/2003.12462v2.pdf 入库，paper_name 使用 eval_v3_textcaps_short，返回 task_id。",
            paper_name="eval_v3_textcaps_short",
            pdf_path="data/downloads/2003.12462v2.pdf",
            tool_args=[
                {"tool": "ingest_paper", "arg": "paper_name", "equals": "eval_v3_textcaps_short"},
                {"tool": "ingest_paper", "arg": "pdf_path", "equals": "data/downloads/2003.12462v2.pdf"},
            ],
            tags=["existing-pdf", "enqueue"],
        ),
        _ingest_case(
            "ingest_easy_medium_clinical_precheck",
            "easy", "medium",
            "先检查 eval_v3_clinical_medium 是否已入库；若未入库，将 data/downloads/clinical_notes_llm.pdf 入库。",
            paper_name="eval_v3_clinical_medium",
            pdf_path="data/downloads/clinical_notes_llm.pdf",
            required_tools=["check_paper", "ingest", "ingest_paper"],
            tags=["precheck", "conditional"],
        ),
        _ingest_case(
            "ingest_easy_long_download_then_index",
            "easy", "long",
            "下载 arXiv 2502.03373 到 eval_output/agent_eval_artifacts/v3/ingest/e3，文件名 2502.03373，再入库为 eval_v3_longcot_long并持续检查到 indexed。",
            paper_name="eval_v3_longcot_long",
            pdf_path="",
            required_tools=[
                "ingest", "download_paper", "ingest_paper", "check_task_status",
            ],
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/ingest/e3/2502.03373.pdf",
                "min_size_bytes": 1000,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "destination",
                 "contains": f"{ARTIFACT_ROOT}/ingest/e3"},
                {"tool": "download_paper", "arg": "filename",
                 "equals": "2502.03373"},
            ],
            setup={"clean_paths": [f"{ARTIFACT_ROOT}/ingest/e3/2502.03373.pdf"]},
            tags=["download-and-ingest", "completion-polling"],
            budgets={"timeout_s": 600},
            allow_download=True,
        ),
        _ingest_case(
            "ingest_medium_short_custom_pdf_path",
            "medium", "short",
            "把 data/downloads/llm_medqa.pdf 入库为 eval_v3_medqa_path，并原样传递自定义 pdf_path。",
            paper_name="eval_v3_medqa_path",
            pdf_path="data/downloads/llm_medqa.pdf",
            tool_args=[
                {"tool": "ingest_paper", "arg": "pdf_path", "equals": "data/downloads/llm_medqa.pdf"},
            ],
            tags=["custom-path"],
        ),
        _ingest_case(
            "ingest_medium_medium_verify_searchable",
            "medium", "medium",
            "将 data/downloads/2003.12462v2.pdf 入库为 eval_v3_textcaps_search，完成后检查状态并确认可被搜索。",
            paper_name="eval_v3_textcaps_search",
            pdf_path="data/downloads/2003.12462v2.pdf",
            required_tools=["ingest", "ingest_paper", "check_paper", "search_papers"],
            postconditions=[
                {
                    "kind": "paper_state",
                    "paper_name": "eval_v3_textcaps_search",
                    "allowed": ["indexed"],
                    "poll_timeout_s": 180,
                },
                {
                    "kind": "retrieval_probe",
                    "query": "TextCaps OCR captions",
                    "paper_name": "eval_v3_textcaps_search",
                    "min_hits": 1,
                },
            ],
            tags=["post-ingest-retrieval"],
        ),
        _ingest_case(
            "ingest_medium_long_batch_two",
            "medium", "long",
            "依次把 data/downloads/clinical_notes_llm.pdf 和 data/downloads/llm_medqa.pdf 入库为 eval_v3_clinical_batch 与 eval_v3_medqa_batch。",
            paper_name="eval_v3_clinical_batch",
            pdf_path="data/downloads/clinical_notes_llm.pdf",
            postconditions=[
                {
                    "kind": "paper_state",
                    "paper_name": "eval_v3_clinical_batch",
                    "allowed": ["indexed"],
                    "poll_timeout_s": 240,
                },
                {
                    "kind": "paper_state",
                    "paper_name": "eval_v3_medqa_batch",
                    "allowed": ["indexed"],
                    "poll_timeout_s": 240,
                },
            ],
            tool_args=[
                {"tool": "ingest_paper", "arg": "paper_name", "equals": "eval_v3_clinical_batch"},
                {"tool": "ingest_paper", "arg": "paper_name", "equals": "eval_v3_medqa_batch"},
            ],
            tags=["batch", "two-papers"],
            budgets={"timeout_s": 600, "max_tool_calls": 10},
        ),
        _ingest_case(
            "ingest_hard_short_duplicate_guard",
            "hard", "short",
            "对 data/downloads/2003.12462v2.pdf 重复提交同一入库目标 eval_v3_textcaps_duplicate，确认不会产生重复论文或重复 chunk。",
            paper_name="eval_v3_textcaps_duplicate",
            pdf_path="data/downloads/2003.12462v2.pdf",
            required_tools=["check_paper", "ingest", "ingest_paper"],
            postconditions=[
                {
                    "kind": "paper_state",
                    "paper_name": "eval_v3_textcaps_duplicate",
                    "allowed": ["indexed"],
                    "poll_timeout_s": 180,
                },
                {
                    "kind": "duplicate_free",
                    "paper_name": "eval_v3_textcaps_duplicate",
                    "max_duplicate_groups": 0,
                },
            ],
            tags=["idempotency", "duplicate-index"],
        ),
        _ingest_case(
            "ingest_hard_medium_fault_recovery",
            "hard", "medium",
            "注入 ingest_paper 首次失败，确认失败后不留下半入库状态，再重试完成 eval_v3_medqa_recovery。",
            paper_name="eval_v3_medqa_recovery",
            pdf_path="data/downloads/llm_medqa.pdf",
            required_tools=["ingest", "ingest_paper", "check_paper"],
            postconditions=[{
                "kind": "paper_state",
                "paper_name": "eval_v3_medqa_recovery",
                "allowed": ["indexed"],
                "poll_timeout_s": 240,
            }],
            setup={
                "failure_injection": {
                    "tool": "ingest_paper",
                    "mode": "raise",
                    "max_failures": 1,
                    "then_recover": True,
                },
            },
            tags=["fault-injection", "recovery", "rollback"],
            budgets={"timeout_s": 600},
        ),
        _ingest_case(
            "ingest_hard_long_download_index_verify",
            "hard", "long",
            "下载 arXiv 1706.03762 到 eval_output/agent_eval_artifacts/v3/ingest/e9，文件名 1706.03762，入库为 eval_v3_attention_endtoend，完成后验证 chunk_count>0、可检索且无重复索引。",
            paper_name="eval_v3_attention_endtoend",
            pdf_path="",
            required_tools=[
                "ingest", "download_paper", "ingest_paper",
                "check_task_status", "check_paper", "fetch_content",
            ],
            artifacts=[{
                "path": f"{ARTIFACT_ROOT}/ingest/e9/1706.03762.pdf",
                "min_size_bytes": 1000,
            }],
            tool_args=[
                {"tool": "download_paper", "arg": "destination",
                 "contains": f"{ARTIFACT_ROOT}/ingest/e9"},
                {"tool": "download_paper", "arg": "filename",
                 "equals": "1706.03762"},
            ],
            postconditions=[
                {
                    "kind": "paper_state",
                    "paper_name": "eval_v3_attention_endtoend",
                    "allowed": ["indexed"],
                    "poll_timeout_s": 300,
                },
                {
                    "kind": "chunk_count",
                    "paper_name": "eval_v3_attention_endtoend",
                    "min": 1,
                },
                {
                    "kind": "retrieval_probe",
                    "query": "scaled dot-product attention",
                    "paper_name": "eval_v3_attention_endtoend",
                    "min_hits": 1,
                },
                {
                    "kind": "duplicate_free",
                    "paper_name": "eval_v3_attention_endtoend",
                    "max_duplicate_groups": 0,
                },
            ],
            setup={"clean_paths": [f"{ARTIFACT_ROOT}/ingest/e9/1706.03762.pdf"]},
            tags=["end-to-end", "download-index-retrieve", "long-context"],
            budgets={"timeout_s": 900, "max_tool_calls": 14, "max_tokens": 60_000},
            allow_download=True,
        ),
    ]
    return cases


def validate_cases(cases: list[dict[str, Any]],
                   *, chunks_path: Path = CHUNKS_PATH) -> dict[str, Any]:
    """Validate matrix balance and current-corpus ground-truth coverage."""
    ids = [row.get("id") for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids")

    matrix = {
        (task_type, difficulty, task_length)
        for task_type in TASK_TYPES
        for difficulty in DIFFICULTIES
        for task_length in TASK_LENGTHS
    }
    actual = {
        (row.get("task_type"), row.get("difficulty_level"), row.get("task_length"))
        for row in cases
    }
    if actual != matrix:
        missing = sorted(matrix - actual)
        extra = sorted(actual - matrix)
        raise ValueError(f"matrix mismatch; missing={missing}, extra={extra}")

    corpus = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_ids = {
        str(chunk.get("chunk_id"))
        for chunk in corpus.get("chunks", [])
        if chunk.get("chunk_id")
    }
    missing_refs: dict[str, list[str]] = {}
    for row in cases:
        missing = [
            str(chunk_id)
            for chunk_id in row.get("ground_truth_ids", [])
            if str(chunk_id) not in chunk_ids
        ]
        if missing:
            missing_refs[str(row.get("id"))] = missing
    if missing_refs:
        raise ValueError(f"ground truth ids missing from corpus: {missing_refs}")

    return {
        "count": len(cases),
        "task_type_mix": {
            task_type: sum(1 for row in cases
                           if row.get("task_type") == task_type)
            for task_type in TASK_TYPES
        },
        "difficulty_mix": {
            difficulty: sum(1 for row in cases
                            if row.get("difficulty_level") == difficulty)
            for difficulty in DIFFICULTIES
        },
        "task_length_mix": {
            length: sum(1 for row in cases if row.get("task_length") == length)
            for length in TASK_LENGTHS
        },
        "retrieval_cases": sum(
            1 for row in cases
            if (row.get("evaluation") or {}).get("retrieval")
        ),
        "non_retrieval_cases": sum(
            1 for row in cases
            if not (row.get("evaluation") or {}).get("retrieval")
        ),
        "stateful_cases": sum(
            1 for row in cases
            if (row.get("setup") or {}).get("mutates_state")
        ),
        "ground_truth_refs": len({
            chunk_id
            for row in cases
            for chunk_id in row.get("ground_truth_ids", [])
        }),
        "corpus_chunks": len(chunk_ids),
    }


def write_manifest(output_path: Path | str = DEFAULT_OUTPUT,
                   *, chunks_path: Path = CHUNKS_PATH) -> dict[str, Any]:
    """Validate and atomically write the v3 JSONL manifest."""
    cases = build_cases()
    summary = validate_cases(cases, chunks_path=chunks_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cases) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **summary,
        "output": str(output),
        "summary": str(summary_path),
        "dataset_id": DATASET_ID,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v3 task benchmark")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = write_manifest(args.output)
    from .datasets import registry_update

    registry_update(
        DATASET_ID,
        path=str(Path(args.output).resolve()),
        source="handcrafted_v3",
        task_type_mix=summary["task_type_mix"],
        difficulty_mix=summary["difficulty_mix"],
        task_length_mix=summary["task_length_mix"],
        retrieval_cases=summary["retrieval_cases"],
        non_retrieval_cases=summary["non_retrieval_cases"],
        stateful_cases=summary["stateful_cases"],
        review_status="draft",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
