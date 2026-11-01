"""Integration tests for MCP auto-wiring in the gateway server."""

import json

import pytest
from fastapi.testclient import TestClient

import agentbridge.server as server_module
from agentbridge.config import BridgeConfig, UpstreamConfig
from agentbridge.mcp.proxy import DEFAULT_CONFIG_ENV_VAR, MCPProxy


class FakeTransport:
    """Replays canned MCP JSON-RPC responses without spawning processes."""

    def __init__(self, tools=None):
        self.tools = tools or {}
        self.calls = []

    def request(self, server_config, method, params=None):
        self.calls.append((server_config, method, params))
        if method == "tools/list":
            return {"tools": self.tools.get(server_config.get("_name", ""), [])}
        raise AssertionError(f"unexpected method: {method}")


WEATHER_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


def make_bridge_config(**overrides) -> BridgeConfig:
    defaults = {
        "upstream": UpstreamConfig(
            name="test-upstream",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            default_model="gpt-4o",
        ),
    }
    defaults.update(overrides)
    return BridgeConfig(**defaults)


@pytest.fixture
def mcp_env(monkeypatch):
    """Keep ambient MCP configuration out of the tests."""
    monkeypatch.delenv(DEFAULT_CONFIG_ENV_VAR, raising=False)
    return monkeypatch


@pytest.fixture
def enabled_app(mcp_env):
    """An app with MCP enabled, backed by a fake in-memory MCP server."""
    proxy = MCPProxy(
        config={"weather": {"command": "fake-weather-server", "_name": "weather"}},
        transport=FakeTransport({"weather": WEATHER_TOOLS}),
    )
    mcp_env.setattr(server_module, "MCPProxy", lambda config_path=None: proxy)
    app = server_module.create_app(make_bridge_config(enable_mcp=True))
    return app, proxy


def test_mcp_tools_disabled_by_default(mcp_env):
    app = server_module.create_app(make_bridge_config())
    with TestClient(app) as client:
        resp = client.get("/v1/mcp/tools")
    assert resp.status_code == 200
    assert resp.json() == {"status": "disabled", "tools": [], "count": 0}


def test_mcp_tools_endpoint_lists_active_tools(enabled_app):
    app, _proxy = enabled_app
    with TestClient(app) as client:
        resp = client.get("/v1/mcp/tools")
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert body["count"] == 1
    assert body["tools"] == [
        {
            "name": "mcp__weather__get_weather",
            "description": "Get current weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_mcp_proxy_starts_and_stops_with_lifespan(enabled_app):
    app, proxy = enabled_app
    with TestClient(app):
        # start_all() probed the fake server and cached its catalogue.
        assert [t.qualified_name for t in proxy.tools] == ["mcp__weather__get_weather"]
    # stop_all() dropped the cached catalogue on shutdown.
    assert proxy._tool_cache == {}


def _fake_upstream_response(payload):
    return {
        "id": "chatcmpl-test",
        "model": payload.get("model", "unknown"),
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_messages_injects_mcp_tools(enabled_app):
    app, _proxy = enabled_app
    captured = {}

    async def fake_complete(payload):
        captured.update(payload)
        return _fake_upstream_response(payload)

    with TestClient(app) as client:
        app.state.client.complete = fake_complete
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    openai_tool_names = [t["function"]["name"] for t in captured["tools"]]
    assert openai_tool_names == ["mcp__weather__get_weather"]


def test_messages_merge_skips_duplicate_tools(enabled_app):
    app, _proxy = enabled_app
    captured = {}

    async def fake_complete(payload):
        captured.update(payload)
        return _fake_upstream_response(payload)

    with TestClient(app) as client:
        app.state.client.complete = fake_complete
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "mcp__weather__get_weather",
                        "description": "Client-side override wins",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

    assert resp.status_code == 200
    openai_tool_names = [t["function"]["name"] for t in captured["tools"]]
    assert openai_tool_names.count("mcp__weather__get_weather") == 1
    client_tools = [
        t
        for t in captured["tools"]
        if t["function"]["name"] == "mcp__weather__get_weather"
    ]
    # The upstream declaration provided by the client is kept as-is.
    assert client_tools[0]["function"]["description"] == "Client-side override wins"


def test_messages_without_mcp_keeps_payload_tools(mcp_env):
    app = server_module.create_app(make_bridge_config())
    captured = {}

    async def fake_complete(payload):
        captured.update(payload)
        return _fake_upstream_response(payload)

    with TestClient(app) as client:
        app.state.client.complete = fake_complete
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "name": "local_tool",
                        "description": "A client tool",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert [t["function"]["name"] for t in captured["tools"]] == ["local_tool"]


def test_mcp_proxy_uses_explicit_config_path(tmp_path, mcp_env):
    config_file = tmp_path / "mcp.json"
    config_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "weather": {"command": "fake-weather-server", "_name": "weather"}
                }
            }
        ),
        encoding="utf-8",
    )
    proxy = MCPProxy(
        config_path=str(config_file),
        transport=FakeTransport({"weather": WEATHER_TOOLS}),
    )
    assert list(proxy.servers) == ["weather"]
    assert [t["name"] for t in proxy.get_anthropic_tools()] == [
        "mcp__weather__get_weather"
    ]
