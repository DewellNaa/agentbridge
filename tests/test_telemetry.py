"""Tests for telemetry metrics, dashboard display and server integration."""

import json
import time

from fastapi.testclient import TestClient

from agentbridge.config import BridgeConfig, UpstreamConfig
from agentbridge.server import create_app
from agentbridge.telemetry.display import (
    log_request_metric,
    render_dashboard_panel,
    render_summary_panel,
)
from agentbridge.telemetry.metrics import RequestMetric, TelemetryManager


def make_config(**overrides) -> BridgeConfig:
    defaults = {
        "upstream": UpstreamConfig(
            name="test-upstream",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            default_model="gpt-4o",
        ),
        "model_mapping": {
            "claude-3-7-sonnet-20250219": "GLM-5.3-Flash",
        },
    }
    defaults.update(overrides)
    return BridgeConfig(**defaults)


# ---------------------------------------------------------------------------
# TelemetryManager unit tests
# ---------------------------------------------------------------------------


class TestTelemetryManager:
    def test_start_request_creates_metric(self):
        mgr = TelemetryManager()
        metric = mgr.start_request(
            "req-1", "claude-3-7-sonnet", "GLM-5.3-Flash", "claude-cli/1.0", True
        )
        assert isinstance(metric, RequestMetric)
        assert metric.request_id == "req-1"
        assert metric.model_requested == "claude-3-7-sonnet"
        assert metric.model_upstream == "GLM-5.3-Flash"
        assert metric.client_agent == "claude-cli/1.0"
        assert metric.is_stream is True
        assert metric.timestamp <= time.time()
        assert mgr.total_requests == 1

    def test_start_request_defaults_client_agent(self):
        mgr = TelemetryManager()
        metric = mgr.start_request("req-1", "m", "u", "", False)
        assert metric.client_agent == "Unknown"
        assert metric.is_stream is False

    def test_record_first_token_sets_ttft(self):
        mgr = TelemetryManager()
        metric = mgr.start_request("req-1", "m", "u", "agent", True)
        time.sleep(0.01)
        mgr.record_first_token(metric)
        assert metric.ttft_ms >= 10.0
        # A second first-token event must not overwrite the recorded TTFT.
        first = metric.ttft_ms
        mgr.record_first_token(metric)
        assert metric.ttft_ms == first

    def test_record_tokens_accumulates(self):
        mgr = TelemetryManager()
        metric = mgr.start_request("req-1", "m", "u", "agent", True)
        mgr.record_tokens(metric, prompt=100, completion=50, reasoning=20)
        mgr.record_tokens(metric, completion=10)
        assert metric.prompt_tokens == 100
        assert metric.completion_tokens == 60
        assert metric.reasoning_tokens == 20
        assert metric.total_tokens == 180
        assert mgr.total_prompt_tokens == 100
        assert mgr.total_completion_tokens == 60
        assert mgr.total_reasoning_tokens == 20

    def test_record_tool_call_deduplicates(self):
        mgr = TelemetryManager()
        metric = mgr.start_request("req-1", "m", "u", "agent", True)
        mgr.record_tool_call(metric, "mcp__bash")
        mgr.record_tool_call(metric, "mcp__bash")
        mgr.record_tool_call(metric, "web_search")
        mgr.record_tool_call(metric, "")
        assert metric.tools_called == ["mcp__bash", "web_search"]

    def test_finish_request_moves_to_history(self):
        mgr = TelemetryManager()
        metric = mgr.start_request("req-1", "m", "u", "agent", True)
        mgr.record_tokens(metric, prompt=10, completion=5)
        time.sleep(0.005)
        mgr.finish_request(metric, status_code=200)
        assert len(mgr._history) == 1
        assert mgr.successful_requests == 1
        assert mgr.failed_requests == 0
        assert metric.duration_ms > 0.0
        assert metric.total_tokens == 15

    def test_finish_request_failure_counts(self):
        mgr = TelemetryManager()
        metric = mgr.start_request("req-1", "m", "u", "agent", True)
        mgr.finish_request(metric, status_code=502, error_message="boom")
        assert mgr.failed_requests == 1
        assert mgr.successful_requests == 0
        assert metric.status_code == 502
        assert metric.error_message == "boom"

    def test_ring_buffer_keeps_recent(self):
        mgr = TelemetryManager(max_history=3)
        for i in range(5):
            metric = mgr.start_request(f"req-{i}", "m", "u", "agent", False)
            mgr.finish_request(metric)
        assert len(mgr._history) == 3
        assert [m.request_id for m in mgr._history] == ["req-2", "req-3", "req-4"]

    def test_get_summary_aggregates(self):
        mgr = TelemetryManager()
        m1 = mgr.start_request("req-1", "a", "b", "agent", True)
        mgr.record_tokens(m1, prompt=10, completion=20, reasoning=5)
        mgr.record_tool_call(m1, "mcp__bash")
        mgr.finish_request(m1, status_code=200)
        m2 = mgr.start_request("req-2", "a", "b", "agent", False)
        mgr.finish_request(m2, status_code=500, error_message="upstream down")

        summary = mgr.get_summary()
        assert summary["total_requests"] == 2
        assert summary["successful_requests"] == 1
        assert summary["failed_requests"] == 1
        assert summary["total_prompt_tokens"] == 10
        assert summary["total_completion_tokens"] == 20
        assert summary["total_reasoning_tokens"] == 5
        assert summary["total_tokens"] == 35
        assert summary["uptime_seconds"] >= 0
        assert summary["avg_duration_ms"] >= 0
        assert summary["tool_usage"] == {"mcp__bash": 1}
        assert len(summary["recent_requests"]) == 2
        assert summary["recent_requests"][1]["error_message"] == "upstream down"


# ---------------------------------------------------------------------------
# Display smoke tests
# ---------------------------------------------------------------------------


class TestDisplay:
    def test_log_request_metric_renders(self):
        from rich.console import Console

        console = Console(record=True, width=200)
        metric = RequestMetric(
            request_id="req-1",
            timestamp=time.time(),
            client_agent="claude-cli/1.0",
            model_requested="claude-3-7-sonnet",
            model_upstream="GLM-5.3-Flash",
            prompt_tokens=1240,
            completion_tokens=450,
            reasoning_tokens=180,
            duration_ms=1800.0,
            ttft_ms=120.0,
            tools_called=["mcp__bash"],
            status_code=200,
        )
        log_request_metric(metric, console=console)
        out = console.export_text()
        assert "200 OK" in out
        assert "claude-3-7-sonnet" in out
        assert "GLM-5.3-Flash" in out
        assert "mcp__bash" in out

    def test_log_request_metric_error_line(self):
        from rich.console import Console

        console = Console(record=True, width=200)
        metric = RequestMetric(
            request_id="req-1",
            timestamp=time.time(),
            status_code=502,
            error_message="upstream exploded",
        )
        log_request_metric(metric, console=console)
        out = console.export_text()
        assert "502 ERR" in out
        assert "upstream exploded" in out

    def test_render_panels_return_rich_panel(self):
        from rich.panel import Panel

        mgr = TelemetryManager()
        metric = mgr.start_request(
            "req-1", "claude-3-7-sonnet", "GLM-5.3-Flash", "agent", True
        )
        mgr.record_tokens(metric, prompt=10, completion=5, reasoning=2)
        mgr.finish_request(metric)

        assert isinstance(render_dashboard_panel(mgr), Panel)
        summary = mgr.get_summary()
        summary["recent_requests"][0]["model_requested"] = "claude-3-7-sonnet"
        summary["recent_requests"][0]["model_upstream"] = "GLM-5.3-Flash"
        panel = render_summary_panel(summary)
        assert isinstance(panel, Panel)


# ---------------------------------------------------------------------------
# Server integration tests
# ---------------------------------------------------------------------------


def fake_upstream_response(payload):
    return {
        "id": "chatcmpl-test",
        "model": payload.get("model", "unknown"),
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


class TestTelemetryEndpoint:
    def test_telemetry_endpoint_empty_state(self):
        app = create_app(make_config())
        with TestClient(app) as client:
            resp = client.get("/v1/telemetry")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_requests"] == 0
        assert body["successful_requests"] == 0
        assert body["failed_requests"] == 0
        assert body["total_tokens"] == 0
        assert body["recent_requests"] == []
        assert body["uptime_seconds"] >= 0


class TestNonStreamTracking:
    def test_non_stream_records_tokens_and_usage(self):
        app = create_app(make_config())

        async def fake_complete(payload):
            return fake_upstream_response(payload)

        with TestClient(app) as client:
            app.state.client.complete = fake_complete
            resp = client.post(
                "/v1/messages",
                headers={"User-Agent": "claude-cli/1.0"},
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            summary = client.get("/v1/telemetry").json()

        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["total_prompt_tokens"] == 12
        assert summary["total_completion_tokens"] == 7
        assert summary["total_tokens"] == 19
        recent = summary["recent_requests"][0]
        assert recent["client_agent"] == "claude-cli/1.0"
        assert recent["model_requested"] == "claude-3-7-sonnet-20250219"
        assert recent["model_upstream"] == "GLM-5.3-Flash"
        assert recent["is_stream"] is False
        assert recent["status_code"] == 200

    def test_non_stream_records_tool_calls(self):
        app = create_app(make_config())

        async def fake_complete_with_tools(payload):
            resp = fake_upstream_response(payload)
            resp["choices"][0]["message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "mcp__bash",
                            "arguments": '{"cmd": "ls"}',
                        },
                    }
                ],
            }
            return resp

        app = create_app(make_config())
        with TestClient(app) as client:
            app.state.client.complete = fake_complete_with_tools
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 200
            summary = client.get("/v1/telemetry").json()

        assert summary["tool_usage"] == {"mcp__bash": 1}
        assert summary["recent_requests"][0]["tools_called"] == ["mcp__bash"]

    def test_non_stream_upstream_error_recorded(self):
        app = create_app(make_config())

        async def failing_complete(payload):
            raise RuntimeError("connection refused")

        with TestClient(app) as client:
            app.state.client.complete = failing_complete
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 502
            summary = client.get("/v1/telemetry").json()

        assert summary["total_requests"] == 1
        assert summary["failed_requests"] == 1
        assert summary["successful_requests"] == 0
        assert "connection refused" in summary["recent_requests"][0]["error_message"]


def make_stream_chunks(with_usage=True):
    chunks = [
        {
            "id": "chatcmpl-stream",
            "model": "GLM-5.3-Flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "thinking hard ",
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "model": "GLM-5.3-Flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hello brave world"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "model": "GLM-5.3-Flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__weather__get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
    ]
    if with_usage:
        chunks.append(
            {
                "id": "chatcmpl-stream",
                "model": "GLM-5.3-Flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 13,
                    "total_tokens": 34,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            }
        )
    else:
        chunks.append(
            {
                "id": "chatcmpl-stream",
                "model": "GLM-5.3-Flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
    return chunks


class TestStreamTracking:
    def test_stream_records_usage_ttft_and_tools(self):
        app = create_app(make_config())

        async def fake_stream(payload):
            await __import__("asyncio").sleep(0.01)
            for chunk in make_stream_chunks(with_usage=True):
                yield chunk

        with TestClient(app) as client:
            app.state.client.stream_complete = fake_stream
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "event: message_stop" in resp.text
            summary = client.get("/v1/telemetry").json()

        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["total_prompt_tokens"] == 21
        assert summary["total_completion_tokens"] == 13
        assert summary["total_reasoning_tokens"] == 4
        assert summary["total_tokens"] == 38
        recent = summary["recent_requests"][0]
        assert recent["ttft_ms"] > 0.0
        assert recent["is_stream"] is True
        assert recent["tools_called"] == ["mcp__weather__get_weather"]

    def test_stream_without_usage_estimates_tokens(self):
        app = create_app(make_config())

        async def fake_stream(payload):
            for chunk in make_stream_chunks(with_usage=False):
                yield chunk

        with TestClient(app) as client:
            app.state.client.stream_complete = fake_stream
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            summary = client.get("/v1/telemetry").json()

        assert summary["total_prompt_tokens"] == 0
        # "thinking hard " (14) + "hello brave world" (17) = 31 chars -> ~8 tokens.
        assert summary["total_completion_tokens"] == (31 + 3) // 4
        assert summary["successful_requests"] == 1

    def test_stream_exception_recorded(self):
        app = create_app(make_config())

        async def failing_stream(payload):
            yield make_stream_chunks(with_usage=False)[0]
            raise RuntimeError("stream blew up")

        with TestClient(app) as client:
            app.state.client.stream_complete = failing_stream
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "event: error" in resp.text
            assert "stream blew up" in resp.text
            summary = client.get("/v1/telemetry").json()

        assert summary["total_requests"] == 1
        assert summary["failed_requests"] == 1
        assert summary["successful_requests"] == 0
        recent = summary["recent_requests"][0]
        assert recent["status_code"] == 500
        assert "stream blew up" in recent["error_message"]

    def test_stream_emits_valid_sse_events(self):
        app = create_app(make_config())

        async def fake_stream(payload):
            for chunk in make_stream_chunks(with_usage=True):
                yield chunk

        with TestClient(app) as client:
            app.state.client.stream_complete = fake_stream
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        event_types = []
        for line in resp.text.splitlines():
            if line.startswith("event: "):
                event_types.append(line[len("event: ") :])
        assert event_types[0] == "message_start"
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert event_types[-1] == "message_stop"
        # Every SSE event must carry a JSON payload.
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                json.loads(line[len("data: ") :])
