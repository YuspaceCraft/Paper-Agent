"""Loopback integration test for real OpenAI-compatible failover."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent.core.model_gateway import ModelCandidate, ResilientChatModel


class _ModelHandler(BaseHTTPRequestHandler):
    counts: dict[str, int] = {}
    moderation_flagged = True

    def log_message(self, *_args):  # noqa: ANN002
        return

    def do_POST(self):  # noqa: N802
        path = self.path
        self.counts[path] = self.counts.get(path, 0) + 1
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if path.startswith("/primary/"):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": {
                    "message": "invalid api key",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            }).encode("utf-8"))
            return
        if path.startswith("/moderation"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "results": [{
                    "flagged": self.moderation_flagged,
                    "categories": {
                        "harassment": self.moderation_flagged,
                    },
                    "category_scores": {
                        "harassment": 0.98 if self.moderation_flagged else 0.0,
                    },
                }]
            }).encode("utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "created": 1,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "fallback ok"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }).encode("utf-8"))


def test_real_openai_compatible_auth_failure_falls_back():
    _ModelHandler.counts = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_port}"
    try:
        primary = ChatOpenAI(
            model="fake-model",
            api_key="bad-key",
            base_url=f"{host}/primary/v1",
            max_retries=0,
            request_timeout=5,
        )
        secondary = ChatOpenAI(
            model="fake-model",
            api_key="good-key",
            base_url=f"{host}/secondary/v1",
            max_retries=0,
            request_timeout=5,
        )
        gateway = ResilientChatModel(
            [primary, secondary],
            [
                ModelCandidate(
                    "primary",
                    "fake-model",
                    f"{host}/primary/v1",
                    "bad-key",
                    max_retries=0,
                ),
                ModelCandidate(
                    "secondary",
                    "fake-model",
                    f"{host}/secondary/v1",
                    "good-key",
                    max_retries=0,
                ),
            ],
            breaker_threshold=1,
            breaker_cooldown=60,
        )

        result = asyncio.run(gateway.ainvoke([
            HumanMessage(content="ping"),
        ]))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.content == "fallback ok"
    assert _ModelHandler.counts.get("/primary/v1/chat/completions") == 1
    assert _ModelHandler.counts.get("/secondary/v1/chat/completions") == 1


def test_remote_moderation_endpoint_blocks_flagged_content(monkeypatch):
    from agent.core.content_safety import classify_content_safety

    _ModelHandler.counts = {}
    _ModelHandler.moderation_flagged = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("AGENT_MODERATION_ENDPOINT", f"{host}/moderation")
    monkeypatch.setenv("AGENT_MODERATION_API_KEY_ENV", "MODERATION_TEST_KEY")
    monkeypatch.setenv("MODERATION_TEST_KEY", "test-key")
    try:
        decision = asyncio.run(classify_content_safety("neutral text"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert decision.blocked is True
    assert decision.source == "remote"
    assert decision.category == "HARASSMENT"


def test_provider_canary_reports_ready_for_model_and_benign_moderation(monkeypatch):
    from agent.core.provider_canary import run_provider_canary

    class FakeModel:
        async def probe(self):
            return {
                "status": "ready",
                "candidate": "primary",
                "candidates": [{"label": "primary"}],
            }

    _ModelHandler.counts = {}
    _ModelHandler.moderation_flagged = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("AGENT_MODERATION_ENDPOINT", f"{host}/moderation")
    try:
        result = asyncio.run(run_provider_canary(model=FakeModel()))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["status"] == "ready"
    assert [item["status"] for item in result["checks"]] == [
        "ready", "ready",
    ]
