"""Resilient model invocation with endpoint, key and circuit-breaker fallback."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Iterable

from .errors import ModelGatewayError


@dataclass(frozen=True)
class ModelCandidate:
    """One independently callable model endpoint."""

    label: str
    model: str
    base_url: str
    api_key: str
    request_timeout: float = 120.0
    max_retries: int = 3

    def kwargs(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }

    def trace_view(self) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
        }


def _json_env(name: str, default: Any) -> Any:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def build_model_candidates(
    *,
    primary_model: str,
    primary_base_url: str,
    primary_api_key: str,
    request_timeout: float,
    max_retries: int,
) -> list[ModelCandidate]:
    """Build model candidates from the primary route plus optional env fallbacks.

    Supported layouts:

    ``AGENT_MODEL_FALLBACKS``:
      ``[{"model": "...", "base_url": "...", "api_key_env": "ALT_KEY"}]``

    ``AGENT_MODEL_FALLBACK_KEY_ENVS``:
      ``["ALT_KEY_1", "ALT_KEY_2"]`` for the same model and endpoint.
    """
    candidates: list[ModelCandidate] = [ModelCandidate(
        label="primary",
        model=primary_model,
        base_url=primary_base_url,
        api_key=primary_api_key,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )]

    key_envs = _json_env("AGENT_MODEL_FALLBACK_KEY_ENVS", [])
    if isinstance(key_envs, list):
        for env_name in key_envs:
            name = str(env_name or "").strip()
            if not name:
                continue
            key = os.getenv(name, "")
            if not key:
                continue
            candidates.append(ModelCandidate(
                label=f"key:{name}",
                model=primary_model,
                base_url=primary_base_url,
                api_key=key,
                request_timeout=request_timeout,
                max_retries=max_retries,
            ))

    fallbacks = _json_env("AGENT_MODEL_FALLBACKS", [])
    if isinstance(fallbacks, list):
        for index, item in enumerate(fallbacks, start=1):
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or primary_model).strip()
            if not model:
                continue
            base_url = str(item.get("base_url") or primary_base_url).strip()
            api_key_env = str(item.get("api_key_env") or "").strip()
            api_key = os.getenv(api_key_env, "") if api_key_env else str(
                item.get("api_key") or ""
            )
            if not api_key:
                continue
            candidates.append(ModelCandidate(
                label=str(item.get("label") or f"fallback:{index}"),
                model=model,
                base_url=base_url,
                api_key=api_key,
                request_timeout=float(
                    item.get("request_timeout") or request_timeout
                ),
                max_retries=int(item.get("max_retries") or max_retries),
            ))

    deduped: list[ModelCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.model, candidate.base_url, candidate.api_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _status_code(exc: BaseException) -> int | None:
    for value in (
        exc,
        getattr(exc, "response", None),
        getattr(exc, "cause", None),
        getattr(exc, "__cause__", None),
    ):
        raw = getattr(value, "status_code", None)
        try:
            if raw is not None:
                return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def should_fallback(exc: BaseException) -> bool:
    """Return whether another model candidate may safely handle this error."""
    status = _status_code(exc)
    if status in {401, 403, 404, 408, 409, 429}:
        return True
    if status is not None and 500 <= status <= 599:
        return True
    if status is not None:
        return False
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        ),
    ):
        return True
    name = type(exc).__name__.casefold()
    return any(token in name for token in (
        "authentication",
        "permissiondenied",
        "notfound",
        "ratelimit",
        "connection",
        "timeout",
        "internalserver",
        "serviceunavailable",
    ))


class ResilientChatModel:
    """Small async wrapper that preserves chat-model binding APIs."""

    def __init__(
        self,
        models: Iterable[Any],
        candidates: Iterable[ModelCandidate],
        *,
        breaker_threshold: int | None = None,
        breaker_cooldown: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._models = list(models)
        self._candidates = list(candidates)
        if len(self._models) != len(self._candidates):
            raise ValueError("models and candidates must have equal length")
        self._breaker_threshold = max(
            1,
            int(
                breaker_threshold
                if breaker_threshold is not None
                else os.getenv("AGENT_MODEL_BREAKER_THRESHOLD", "3")
            ),
        )
        self._breaker_cooldown = max(
            0.0,
            float(
                breaker_cooldown
                if breaker_cooldown is not None
                else os.getenv("AGENT_MODEL_BREAKER_COOLDOWN", "30")
            ),
        )
        self._clock = clock
        self._breakers: dict[str, dict] = {}

    @property
    def model_name(self) -> str:
        return str(getattr(self._models[0], "model_name", "") or "")

    @property
    def kwargs(self) -> dict:
        return dict(getattr(self._models[0], "kwargs", {}) or {})

    @property
    def candidates(self) -> tuple[ModelCandidate, ...]:
        return tuple(self._candidates)

    def bind_tools(self, tools: list, **kwargs: Any) -> "ResilientChatModel":
        return ResilientChatModel(
            [model.bind_tools(tools, **kwargs) for model in self._models],
            self._candidates,
            breaker_threshold=self._breaker_threshold,
            breaker_cooldown=self._breaker_cooldown,
            clock=self._clock,
        )

    def with_structured_output(
        self, schema: Any, **kwargs: Any,
    ) -> "ResilientChatModel":
        return ResilientChatModel(
            [
                model.with_structured_output(schema, **kwargs)
                for model in self._models
            ],
            self._candidates,
            breaker_threshold=self._breaker_threshold,
            breaker_cooldown=self._breaker_cooldown,
            clock=self._clock,
        )

    def _available_indexes(self) -> list[int]:
        now = self._clock()
        available: list[int] = []
        for index, candidate in enumerate(self._candidates):
            state = self._breakers.get(candidate.label)
            opened_at = state.get("opened_at") if state else None
            if opened_at is None or now - opened_at >= self._breaker_cooldown:
                if opened_at is not None:
                    self._breakers.pop(candidate.label, None)
                available.append(index)
        return available

    def _record_success(self, candidate: ModelCandidate) -> None:
        self._breakers.pop(candidate.label, None)

    def _record_failure(
        self, candidate: ModelCandidate, exc: BaseException,
    ) -> None:
        if not should_fallback(exc):
            return
        state = self._breakers.setdefault(
            candidate.label, {"failures": 0, "opened_at": None},
        )
        state["failures"] += 1
        if state["failures"] >= self._breaker_threshold:
            state["opened_at"] = self._clock()

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        errors: list[BaseException] = []
        for index in self._available_indexes():
            candidate = self._candidates[index]
            try:
                result = await self._models[index].ainvoke(
                    input, config=config, **kwargs,
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                self._record_failure(candidate, exc)
                if not should_fallback(exc):
                    raise
                continue
            self._record_success(candidate)
            return result
        if errors:
            raise ModelGatewayError(
                "all model candidates failed",
                attempts=len(errors),
                cause=errors[-1],
            )
        raise ModelGatewayError("all model candidates are circuit-open")

    async def astream(
        self, input: Any, config: Any = None, **kwargs: Any,
    ) -> AsyncIterator[Any]:
        errors: list[BaseException] = []
        for index in self._available_indexes():
            candidate = self._candidates[index]
            emitted = False
            try:
                async for chunk in self._models[index].astream(
                    input, config=config, **kwargs,
                ):
                    emitted = True
                    yield chunk
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                self._record_failure(candidate, exc)
                if emitted or not should_fallback(exc):
                    raise
                continue
            self._record_success(candidate)
            return
        if errors:
            raise ModelGatewayError(
                "all model candidates failed",
                attempts=len(errors),
                cause=errors[-1],
            )
        raise ModelGatewayError("all model candidates are circuit-open")

    async def probe(self) -> dict:
        """Run a minimal real invocation against the first available candidate."""
        from langchain_core.messages import HumanMessage

        available = self._available_indexes()
        if not available:
            return {
                "status": "degraded",
                "error": "all model candidates are circuit-open",
                "candidates": [candidate.trace_view()
                                for candidate in self._candidates],
            }
        index = available[0]
        candidate = self._candidates[index]
        try:
            await asyncio.wait_for(
                self._models[index].ainvoke(
                    [HumanMessage(content="Reply with OK.")],
                    config={"metadata": {"probe": True}},
                ),
                timeout=float(os.getenv("AGENT_MODEL_PROBE_TIMEOUT", "15")),
            )
        except BaseException as exc:  # noqa: BLE001
            self._record_failure(candidate, exc)
            return {
                "status": "degraded",
                "error": type(exc).__name__,
                "candidates": [item.trace_view() for item in self._candidates],
            }
        self._record_success(candidate)
        return {
            "status": "ready",
            "candidate": candidate.label,
            "candidates": [item.trace_view() for item in self._candidates],
        }


def model_candidate_metadata(model: Any) -> dict:
    if isinstance(model, ResilientChatModel):
        return {
            "model_candidates": [
                candidate.trace_view() for candidate in model.candidates
            ],
        }
    return {}
