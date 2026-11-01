"""Unit tests for the multi-key failover pool (agentbridge.routing.pool)."""

import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import agentbridge.cli as cli
from agentbridge.config import BridgeConfig, UpstreamConfig
from agentbridge.routing.pool import (
    AllUpstreamsFailedError,
    FailoverRouter,
    UpstreamEndpoint,
)
from agentbridge.server import create_app

UPSTREAM_URL = "https://upstream.test/v1"


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------
class FakeUpstreamClient:
    """Scripted stand-in for :class:`UpstreamClient`."""

    def __init__(self, complete_script=None, stream_script=None, response=None):
        self.complete_script = list(complete_script or [])
        self.stream_script = list(stream_script or [])
        self.default_response = response or {"ok": True}
        self.complete_calls = 0
        self.stream_calls = 0
        self.closed = False

    async def complete(self, request):
        self.complete_calls += 1
        if self.complete_script:
            item = self.complete_script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return dict(self.default_response)

    async def stream_complete(self, request):
        self.stream_calls += 1
        for item in list(self.stream_script):
            if isinstance(item, Exception):
                raise item
            yield dict(item)

    async def close(self):
        self.closed = True


def status_error(code: int) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError carrying the given status code."""
    request = httpx.Request("POST", f"{UPSTREAM_URL}/chat/completions")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


def make_endpoint(name: str, client: FakeUpstreamClient, **kwargs) -> UpstreamEndpoint:
    return UpstreamEndpoint(name=name, client=client, **kwargs)


def make_router(clients, strategy="priority", **endpoint_kwargs) -> FailoverRouter:
    endpoints = [
        make_endpoint(f"ep-{i + 1}", client, **endpoint_kwargs)
        for i, client in enumerate(clients)
    ]
    return FailoverRouter(endpoints, strategy=strategy)


OPENAI_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
}

STREAM_CHUNKS = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
]


# ----------------------------------------------------------------------
# Endpoint selection strategies
# ----------------------------------------------------------------------
def test_round_robin_distributes_evenly():
    router = make_router(
        [FakeUpstreamClient() for _ in range(3)], strategy="round_robin"
    )
    picked = [router.get_next_endpoint().name for _ in range(6)]
    assert picked == ["ep-1", "ep-2", "ep-3", "ep-1", "ep-2", "ep-3"]


def test_round_robin_skips_unhealthy_endpoints():
    clients = [FakeUpstreamClient() for _ in range(3)]
    router = make_router(clients, strategy="round_robin")
    router.endpoints[1].is_healthy = False
    picked = [router.get_next_endpoint().name for _ in range(4)]
    assert picked == ["ep-1", "ep-3", "ep-1", "ep-3"]


def test_priority_always_returns_first_healthy():
    router = make_router([FakeUpstreamClient() for _ in range(3)])
    picked = [router.get_next_endpoint().name for _ in range(3)]
    assert picked == ["ep-1", "ep-1", "ep-1"]


def test_priority_skips_cooldown_endpoint():
    router = make_router([FakeUpstreamClient() for _ in range(2)])
    router.endpoints[0].is_healthy = False
    assert router.get_next_endpoint().name == "ep-2"


def test_invalid_strategy_raises_value_error():
    with pytest.raises(ValueError, match="strategy"):
        make_router([FakeUpstreamClient()], strategy="random")


def test_empty_pool_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        FailoverRouter([])


def test_get_next_endpoint_all_down_raises():
    router = make_router([FakeUpstreamClient(), FakeUpstreamClient()])
    for endpoint in router.endpoints:
        endpoint.is_healthy = False
    with pytest.raises(AllUpstreamsFailedError):
        router.get_next_endpoint()


def test_cooldown_expiry_restores_endpoint(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(FailoverRouter, "_now", lambda self: clock["t"])
    router = make_router([FakeUpstreamClient(), FakeUpstreamClient()])
    for endpoint in router.endpoints:
        endpoint.cooldown_seconds = 30.0
        router._mark_failure(endpoint, status_error(503))

    assert all(e.is_healthy is False for e in router.endpoints)
    clock["t"] = 120.0
    with pytest.raises(AllUpstreamsFailedError):
        router.get_next_endpoint()

    clock["t"] = 130.5
    assert router.get_next_endpoint() is router.endpoints[0]
    assert router.endpoints[0].is_healthy is True
    assert router.endpoints[0].cooldown_until == 0.0


def test_recovery_does_not_reset_failure_count():
    router = make_router([FakeUpstreamClient()])
    endpoint = router.endpoints[0]
    endpoint.is_healthy = False
    endpoint.failure_count = 2
    endpoint.cooldown_until = 1.0  # long expired
    router.get_next_endpoint()
    assert endpoint.is_healthy is True
    assert endpoint.failure_count == 2  # history is preserved


async def test_close_closes_all_clients():
    clients = [FakeUpstreamClient() for _ in range(2)]
    router = make_router(clients)

    await router.close()

    assert all(client.closed for client in clients)


# ----------------------------------------------------------------------
# complete(): non-streaming failover
# ----------------------------------------------------------------------
@pytest.mark.parametrize("code", [402, 429, 500, 502, 503, 504])
async def test_complete_fails_over_on_status_codes(code):
    primary = FakeUpstreamClient(complete_script=[status_error(code)])
    secondary = FakeUpstreamClient(response={"from": "secondary"})
    router = make_router([primary, secondary])

    response, name = await router.complete({"model": "gpt-4o"})

    assert response == {"from": "secondary"}
    assert name == "ep-2"
    assert primary.complete_calls == 1
    assert secondary.complete_calls == 1
    assert router.endpoints[0].failure_count == 1
    assert router.endpoints[0].is_healthy is False


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
    ],
)
async def test_complete_fails_over_on_network_errors(exc):
    primary = FakeUpstreamClient(complete_script=[exc])
    secondary = FakeUpstreamClient(response={"from": "secondary"})
    router = make_router([primary, secondary])

    response, name = await router.complete({"model": "gpt-4o"})

    assert response == {"from": "secondary"}
    assert name == "ep-2"


async def test_complete_returns_response_and_endpoint_name():
    router = make_router([FakeUpstreamClient(response={"answer": 42})])
    response, name = await router.complete({})
    assert response == {"answer": 42}
    assert name == "ep-1"


async def test_complete_all_failed_raises():
    primary = FakeUpstreamClient(complete_script=[status_error(429)])
    secondary = FakeUpstreamClient(complete_script=[httpx.ConnectError("down")])
    router = make_router([primary, secondary])

    with pytest.raises(AllUpstreamsFailedError, match="last error"):
        await router.complete({})

    assert primary.complete_calls == 1
    assert secondary.complete_calls == 1


async def test_complete_all_failed_marks_cooldown():
    clients = [
        FakeUpstreamClient(complete_script=[status_error(503)]) for _ in range(2)
    ]
    router = make_router(clients)
    with pytest.raises(AllUpstreamsFailedError):
        await router.complete({})
    assert all(e.is_healthy is False for e in router.endpoints)
    assert all(e.failure_count == 1 for e in router.endpoints)


async def test_complete_non_failover_error_propagates():
    primary = FakeUpstreamClient(complete_script=[status_error(401)])
    secondary = FakeUpstreamClient()
    router = make_router([primary, secondary])

    with pytest.raises(httpx.HTTPStatusError):
        await router.complete({})

    assert secondary.complete_calls == 0  # no failover on auth errors
    assert router.endpoints[0].failure_count == 0
    assert router.endpoints[0].is_healthy is True


async def test_complete_failure_sets_future_cooldown():
    primary = FakeUpstreamClient(complete_script=[status_error(429)])
    secondary = FakeUpstreamClient()
    router = make_router([primary, secondary])
    router.endpoints[0].cooldown_seconds = 30.0

    await router.complete({})

    endpoint = router.endpoints[0]
    assert endpoint.cooldown_until > router._now()
    assert endpoint.cooldown_until - router._now() <= 30.0


async def test_complete_retries_over_multiple_failures():
    clients = [
        FakeUpstreamClient(complete_script=[status_error(429)]),
        FakeUpstreamClient(complete_script=[status_error(503)]),
        FakeUpstreamClient(response={"from": "third"}),
    ]
    router = make_router(clients)

    response, name = await router.complete({})

    assert response == {"from": "third"}
    assert name == "ep-3"


async def test_complete_logs_failover_warning(caplog):
    primary = FakeUpstreamClient(complete_script=[status_error(429)])
    secondary = FakeUpstreamClient()
    router = make_router([primary, secondary])

    with caplog.at_level(logging.WARNING, logger="agentbridge.routing.pool"):
        await router.complete({})

    assert "Failover:" in caplog.text
    assert "ep-1" in caplog.text
    assert "failing over to next upstream" in caplog.text


# ----------------------------------------------------------------------
# stream_complete(): streaming failover
# ----------------------------------------------------------------------
async def test_stream_success_yields_all_chunks():
    router = make_router([FakeUpstreamClient(stream_script=STREAM_CHUNKS)])
    chunks = [chunk async for chunk in router.stream_complete({})]
    assert len(chunks) == 3
    assert chunks[1]["choices"][0]["delta"]["content"] == "hi"


async def test_stream_failover_before_first_chunk():
    primary = FakeUpstreamClient(stream_script=[httpx.ConnectError("refused")])
    secondary = FakeUpstreamClient(stream_script=STREAM_CHUNKS)
    router = make_router([primary, secondary])

    chunks = [chunk async for chunk in router.stream_complete({})]

    assert len(chunks) == 3
    assert primary.stream_calls == 1
    assert secondary.stream_calls == 1
    assert router.endpoints[0].failure_count == 1


async def test_stream_failover_on_429_before_first_chunk():
    primary = FakeUpstreamClient(stream_script=[status_error(429)])
    secondary = FakeUpstreamClient(stream_script=STREAM_CHUNKS)
    router = make_router([primary, secondary])

    chunks = [chunk async for chunk in router.stream_complete({})]

    assert len(chunks) == 3
    assert router.endpoints[0].is_healthy is False


async def test_stream_no_failover_after_first_chunk_sent():
    primary = FakeUpstreamClient(stream_script=[STREAM_CHUNKS[0], status_error(500)])
    secondary = FakeUpstreamClient(stream_script=STREAM_CHUNKS)
    router = make_router([primary, secondary])

    chunks = [chunk async for chunk in router.stream_complete({})]

    assert len(chunks) == 1  # first chunk delivered, then clean stop
    assert secondary.stream_calls == 0  # no silent retry mid-stream


async def test_stream_midflight_error_does_not_mark_failure():
    primary = FakeUpstreamClient(
        stream_script=[STREAM_CHUNKS[0], httpx.ReadTimeout("mid-stream")]
    )
    router = make_router([primary])

    chunks = [chunk async for chunk in router.stream_complete({})]

    assert len(chunks) == 1
    assert router.endpoints[0].failure_count == 0
    assert router.endpoints[0].is_healthy is True


async def test_stream_all_failed_before_first_chunk_raises():
    clients = [
        FakeUpstreamClient(stream_script=[status_error(503)]),
        FakeUpstreamClient(stream_script=[httpx.ConnectTimeout("timeout")]),
    ]
    router = make_router(clients)

    with pytest.raises(AllUpstreamsFailedError):
        [chunk async for chunk in router.stream_complete({})]


async def test_stream_failover_logs_warning(caplog):
    primary = FakeUpstreamClient(stream_script=[status_error(503)])
    secondary = FakeUpstreamClient(stream_script=STREAM_CHUNKS)
    router = make_router([primary, secondary])

    with caplog.at_level(logging.WARNING, logger="agentbridge.routing.pool"):
        [chunk async for chunk in router.stream_complete({})]

    assert "ep-1 failed" in caplog.text


async def test_stream_non_failover_error_before_chunk_propagates():
    primary = FakeUpstreamClient(stream_script=[status_error(401)])
    secondary = FakeUpstreamClient(stream_script=STREAM_CHUNKS)
    router = make_router([primary, secondary])

    with pytest.raises(httpx.HTTPStatusError):
        [chunk async for chunk in router.stream_complete({})]

    assert secondary.stream_calls == 0


# ----------------------------------------------------------------------
# get_pool_status()
# ----------------------------------------------------------------------
def test_get_pool_status_shape():
    router = make_router([FakeUpstreamClient(), FakeUpstreamClient()])
    status = router.get_pool_status()
    assert [entry["name"] for entry in status] == ["ep-1", "ep-2"]
    assert all(entry["is_healthy"] is True for entry in status)
    assert all(entry["failure_count"] == 0 for entry in status)
    assert all(entry["cooldown_remaining"] == 0.0 for entry in status)


def test_get_pool_status_reflects_cooldown(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(FailoverRouter, "_now", lambda self: clock["t"])
    router = make_router([FakeUpstreamClient()])
    router.endpoints[0].cooldown_seconds = 30.0
    router._mark_failure(router.endpoints[0], status_error(500))

    status = router.get_pool_status()

    assert status[0]["is_healthy"] is False
    assert status[0]["failure_count"] == 1
    assert 0.0 < status[0]["cooldown_remaining"] <= 30.0


def test_get_pool_status_recovers_expired_cooldown(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(FailoverRouter, "_now", lambda self: clock["t"])
    router = make_router([FakeUpstreamClient()])
    router.endpoints[0].cooldown_seconds = 30.0
    router._mark_failure(router.endpoints[0], status_error(500))

    clock["t"] = 200.0
    status = router.get_pool_status()

    assert status[0]["is_healthy"] is True
    assert status[0]["cooldown_remaining"] == 0.0


# ----------------------------------------------------------------------
# Server integration: /v1/failover/status and request routing
# ----------------------------------------------------------------------
def make_pool_config(**upstream_overrides) -> BridgeConfig:
    defaults = {
        "name": "test-upstream",
        "base_url": UPSTREAM_URL,
        "api_key": "test-key",
        "default_model": "gpt-4o",
    }
    defaults.update(upstream_overrides)
    return BridgeConfig(upstream=UpstreamConfig(**defaults))


def test_failover_status_endpoint_reports_pool():
    app = create_app(
        make_pool_config(api_keys=["k1", "k2"], fallback_url="https://fallback.test/v1")
    )
    with TestClient(app) as client:
        resp = client.get("/v1/failover/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["strategy"] == "priority"
    assert body["count"] == 3
    assert [e["name"] for e in body["endpoints"]] == ["key-1", "key-2", "fallback"]
    assert all(e["is_healthy"] is True for e in body["endpoints"])


def test_failover_status_endpoint_single_upstream():
    app = create_app(make_pool_config())
    with TestClient(app) as client:
        resp = client.get("/v1/failover/status")

    body = resp.json()
    assert body["count"] == 1
    assert body["endpoints"][0]["name"] == "test-upstream"


def test_server_builds_key_endpoints():
    app = create_app(make_pool_config(api_keys=["a", "b", "c"]))
    names = [e.name for e in app.state.endpoints]
    assert names == ["key-1", "key-2", "key-3"]


def test_server_uses_configured_strategy():
    app = create_app(
        make_pool_config(api_keys=["a", "b"], failover_strategy="round_robin")
    )
    assert app.state.router.strategy == "round_robin"


def test_server_nonstream_fails_over_to_next_key():
    app = create_app(make_pool_config(api_keys=["a", "b"]))
    app.state.endpoints[0].client = FakeUpstreamClient(
        complete_script=[status_error(429)]
    )
    app.state.endpoints[1].client = FakeUpstreamClient(response=OPENAI_RESPONSE)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "hello"


def test_server_stream_fails_over_before_first_chunk():
    app = create_app(make_pool_config(api_keys=["a", "b"]))
    app.state.endpoints[0].client = FakeUpstreamClient(
        stream_script=[status_error(503)]
    )
    app.state.endpoints[1].client = FakeUpstreamClient(stream_script=STREAM_CHUNKS)

    with TestClient(app) as client:
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 32,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert "hi" in resp.text
    assert '"type": "error"' not in resp.text


# ----------------------------------------------------------------------
# Config: multi-key & fallback settings
# ----------------------------------------------------------------------
def test_api_keys_comma_string_is_split():
    config = UpstreamConfig(api_keys="k1, k2,,k3,")
    assert config.api_keys == ["k1", "k2", "k3"]


def test_api_keys_list_passthrough():
    config = UpstreamConfig(api_keys=["a", "b"])
    assert config.api_keys == ["a", "b"]


def test_api_keys_default_empty():
    assert UpstreamConfig().api_keys == []


def test_invalid_failover_strategy_rejected():
    with pytest.raises(ValidationError):
        UpstreamConfig(failover_strategy="chaos")


def test_from_env_reads_upstream_keys(monkeypatch):
    monkeypatch.setenv("AGENTBRIDGE_UPSTREAM_KEYS", "alpha,beta,gamma")
    config = BridgeConfig.from_env()
    assert config.upstream.api_keys == ["alpha", "beta", "gamma"]


def test_from_env_reads_fallback_settings(monkeypatch):
    monkeypatch.setenv("AGENTBRIDGE_FALLBACK_URL", "https://fallback.test/v1")
    monkeypatch.setenv("AGENTBRIDGE_FALLBACK_KEY", "fb-key")
    monkeypatch.setenv("AGENTBRIDGE_FALLBACK_MODEL", "glm-4.6")
    config = BridgeConfig.from_env()
    assert config.upstream.fallback_url == "https://fallback.test/v1"
    assert config.upstream.fallback_key == "fb-key"
    assert config.upstream.fallback_model == "glm-4.6"


def test_from_env_reads_failover_strategy(monkeypatch):
    monkeypatch.setenv("AGENTBRIDGE_FAILOVER_STRATEGY", "round_robin")
    assert BridgeConfig.from_env().upstream.failover_strategy == "round_robin"


def test_from_env_failover_defaults(monkeypatch):
    for var in (
        "AGENTBRIDGE_UPSTREAM_KEYS",
        "AGENTBRIDGE_FALLBACK_URL",
        "AGENTBRIDGE_FALLBACK_KEY",
        "AGENTBRIDGE_FALLBACK_MODEL",
        "AGENTBRIDGE_FAILOVER_STRATEGY",
    ):
        monkeypatch.delenv(var, raising=False)
    config = BridgeConfig.from_env()
    assert config.upstream.api_keys == []
    assert config.upstream.fallback_url is None
    assert config.upstream.fallback_key is None
    assert config.upstream.fallback_model is None
    assert config.upstream.failover_strategy == "priority"


# ----------------------------------------------------------------------
# CLI: new flags and banner
# ----------------------------------------------------------------------
def parse(argv):
    return cli.create_parser().parse_args(argv)


def test_parser_accepts_upstream_keys():
    args = parse(["start", "--upstream-keys", "k1,k2,k3"])
    assert args.upstream_keys == "k1,k2,k3"


def test_parser_strategy_default_is_priority():
    assert parse(["start"]).strategy == "priority"


def test_parser_strategy_accepts_round_robin():
    assert parse(["start", "--strategy", "round_robin"]).strategy == "round_robin"


def test_parser_rejects_unknown_strategy():
    with pytest.raises(SystemExit):
        parse(["start", "--strategy", "chaos"])


def test_parser_fallback_flags():
    args = parse(
        [
            "start",
            "--fallback-url",
            "https://fallback.test/v1",
            "--fallback-key",
            "fb",
            "--fallback-model",
            "glm-4.6",
        ]
    )
    assert args.fallback_url == "https://fallback.test/v1"
    assert args.fallback_key == "fb"
    assert args.fallback_model == "glm-4.6"


def test_banner_shows_failover_pool(capsys):
    config = make_pool_config(
        api_keys=["k1", "k2"],
        fallback_url="https://fallback.test/v1",
        failover_strategy="round_robin",
    )
    cli.print_startup_banner(config)
    out = capsys.readouterr().out
    assert "Failover Pool:" in out
    assert "Active (Strategy: round_robin, Endpoints: 3)" in out


def test_banner_omits_failover_for_single_upstream(capsys):
    cli.print_startup_banner(make_pool_config())
    out = capsys.readouterr().out
    assert "Failover Pool:" not in out
