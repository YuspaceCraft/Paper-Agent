"""
retrieval_orchestrator — Offline retrieval evaluation framework.

Validates that vector-store content is accurately recallable.
Produces evaluation reports and optimal retrieval configs.
Read-only: never modifies upstream data or writes new indices.

Usage:
  python -m retrieval_orchestrator evaluate --config retrieval_orchestrator/evaluation.yaml
"""
