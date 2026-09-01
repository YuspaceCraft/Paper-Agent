"""
study.py — 研究知识库 HTTP 薄封装（v10 / Phase C）。

`web/workspace/studies/{topic}/knowledge.json` 的只读上下文 + 写（假设追加）。
领域逻辑在 agent/domains/coding.py；agent 工具 `study_context` 直接调模块。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.domains.coding import load_study, study_add_hypothesis

router = APIRouter(prefix="/api/study", tags=["study"])


class HypothesisBody(BaseModel):
    hypothesis: str = Field(min_length=1, max_length=2000)


@router.get("/context")
async def context(topic: str = "general"):
    return load_study(topic)


@router.post("/hypotheses")
async def add_hypothesis(topic: str = "general", body: HypothesisBody | None = None):
    h = body.hypothesis if body else ""
    if not h.strip():
        raise HTTPException(400, "hypothesis is required")
    result = json.loads(await study_add_hypothesis.ainvoke(
        {"topic": topic, "hypothesis": h}))
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "failed"))
    return result["data"]