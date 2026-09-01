"""
library_api.py — 本地知识库后端的连接守卫（熔断 + 短连接超时）。

背景（TROUBLESHOOTING「agent / 库工具反复超时拖垮整轮」）：内置工具通过
AGENT_API_BASE 访问本地 FastAPI。后端未启动 / 端口错配时，每次库调用都要
等满 httpx 默认 10s（本地端口丢包时连 connect 也要 10s 才断），agent 依据
"transient → Retry once" 反馈反复调用，最终把整轮 TURN_TIMEOUT(300s) 烧光，
用户只能看到「回答超时」。根因是缺少快速失败。

本模块做两件事：
1. 短连接超时：后端在本机 127.0.0.1，connect 2s 未通即视为不可达；
2. 熔断窗口：接口连续失败后，在窗口期（默认 45s）内后续库调用直接速断，
   不再走网络等待。窗口过后自动重新探测（后端可能已恢复）。

熔断状态按 api_base 分桶、进程内共享——react 轮 / plan executor / subagent
三条路径全部在 builtin_provider 汇合，共用同一份判定，无需逐条路径打补丁。
"""
from __future__ import annotations

import os
import time

import httpx

# 熔断窗口（秒）：窗口内库调用直接速断返回；窗口后重新探测。
_DOWN_TTL = float(os.getenv("AGENT_API_BREAKER_TTL", "45"))
# 状态: {api_base: 熔断截止的 monotonic 时间戳}
_DOWN_STATE: dict[str, float] = {}


def api_is_down(api: str) -> bool:
    """True 表示该 api_base 正处于熔断窗口内（后端刚失败过，应快速失败）。"""
    ts = _DOWN_STATE.get(api)
    return bool(ts and ts > time.monotonic())


def api_mark_down(api: str) -> None:
    _DOWN_STATE[api] = time.monotonic() + _DOWN_TTL


def api_mark_up(api: str) -> None:
    _DOWN_STATE.pop(api, None)


def api_timeout(default: float = 10.0, connect: float = 2.0) -> httpx.Timeout:
    """本地后端在 127.0.0.1——connect 2s 足够；不要为死端口白等 10s。"""
    return httpx.Timeout(default, connect=connect)