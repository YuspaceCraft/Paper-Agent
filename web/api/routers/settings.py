"""
settings.py — 项目路径 / 实验根配置 HTTP 薄封装（可配置工作区根）。

GET /api/settings                        → 当前路径配置
PUT /api/settings  {project_path?, experiments_path?}  → 更新并返回

领域逻辑在 agent/workspace_config.py（纯 Python，agent/domains 与 API 共用）。
- project_path 未显式设置（null）→ 项目根=代码根、写作目录=web/workspace/docs（旧行为）；
  显式设置 → 项目根=所选路径、写作目录={project_path}/writing。
- 清除自定义路径：传空串 ""（null = 不改动）。
- experiments_path 独立于 project_path，默认 web/workspace/experiments。
PUT 用 asyncio.Lock 包 read-modify-write；原子写盘由 workspace_config.save_settings
（临时文件 + os.replace）保证。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent import workspace_config

router = APIRouter(prefix="/api/settings", tags=["settings"])

_lock = asyncio.Lock()


class UpdateSettingsBody(BaseModel):
    project_path: str | None = None
    experiments_path: str | None = None


def _payload() -> dict:
    return {
        "project_path": workspace_config.get_project_path(),   # None = 未显式设置
        "project_root": str(workspace_config.get_project_root()),
        "experiments_path": str(workspace_config.get_experiments_path()),
        "writing_dir": str(workspace_config.get_docs_dir()),
    }


@router.get("")
async def get_settings():
    return _payload()


@router.put("")
async def update_settings(body: UpdateSettingsBody):
    async with _lock:
        try:
            workspace_config.set_paths(
                project_path=body.project_path,
                experiments_path=body.experiments_path,
            )
        except (ValueError, OSError) as e:
            raise HTTPException(400, str(e))
    return _payload()