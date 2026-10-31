"""Unit tests for the local MCP relay probe (agentbridge.mcp.proxy)."""

import json
import sys
import textwrap

import pytest

from agentbridge.mcp.proxy import (
    DEFAULT_CONFIG_ENV_VAR,
    MCPConfigError,
    MCPProxy,
    MCPTool,
    MCPTransportError,
    StdioMCPTransport,
    build_mcp_tool_name,
    is_mcp_tool,
    parse_mcp_tool_name,
    sanitize_tool_name,
)


class FakeTransport:
    """Records JSON-RPC calls and replays canned responses."""

    def __init__(self, tools=None, results=None, error=None):
        self.tools = tools or {}
        self.results = results or {}
        self.error = error
        self.calls = []

    def request(self, server_config, method, params=None):
        self.calls.append((server_config, method, params))
        if self.error is not None:
            raise self.error
        if method == "tools/list":
            return {"tools": self.tools.get(server_config.get("_name", ""), [])}
        if method == "tools/call":
            tool = (params or {}).get("name", "")
            if tool in self.results:
                return self.results[tool]
            return {
                "content": [{"type": "text", "text": f"ok:{tool}"}],
                "isError": False,
            }
        raise AssertionError(f"unexpected method: {method}")


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure no ambient MCP config leaks into the tests."""
    monkeypatch.delenv(DEFAULT_CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv("AGENTBRIDGE_MCP_SERVERS", raising=False)
    return monkeypatch


WEATHER_TOOLS = [
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for a city",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

SEARCH_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
]


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


class TestToolNameHelpers:
    def test_build_qualified_name(self):
        assert build_mcp_tool_name("weather", "get_forecast") == (
            "mcp__weather__get_forecast"
        )

    def test_sanitize_replaces_invalid_characters(self):
        assert sanitize_tool_name("My Server.X") == "My_Server_X"

    def test_sanitize_truncates_to_64_chars(self):
        assert len(sanitize_tool_name("x" * 200)) == 64

    def test_sanitize_empty_name_falls_back(self):
        assert sanitize_tool_name("") == "tool"

    def test_parse_qualified_name_roundtrip(self):
        assert parse_mcp_tool_name("mcp__weather__get_forecast") == (
            "weather",
            "get_forecast",
        )

    def test_parse_keeps_tool_underscores_intact(self):
        # Only the first separator after the prefix is split on.
        assert parse_mcp_tool_name("mcp__fs__read__file") == ("fs", "read__file")

    def test_parse_rejects_non_mcp_names(self):
        assert parse_mcp_tool_name("bash") is None
        assert parse_mcp_tool_name("mcp__noseparator") is None
        assert parse_mcp_tool_name("mcp____") is None

    def test_is_mcp_tool(self):
        assert is_mcp_tool("mcp__weather__get_forecast")
        assert not is_mcp_tool("get_forecast")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_inline_config_with_mcpServers_wrapper(self):
        proxy = MCPProxy(
            config={"mcpServers": {"weather": {"command": "uvx"}}},
            env_var=None,
            config_file=None,
        )
        assert proxy.servers == {"weather": {"command": "uvx"}}
        assert proxy.config_sources == ["inline:config"]

    def test_inline_config_bare_mapping(self):
        proxy = MCPProxy(
            config={"weather": {"command": "uvx", "args": ["mcp-server-weather"]}},
            env_var=None,
            config_file=None,
        )
        assert "weather" in proxy.servers

    def test_inline_config_ignores_non_dict_entries(self):
        proxy = MCPProxy(
            config={"weather": {"command": "uvx"}, "bogus": "not-a-dict"},
            env_var=None,
            config_file=None,
        )
        assert proxy.servers == {"weather": {"command": "uvx"}}

    def test_inline_config_rejects_non_dict_document(self):
        with pytest.raises(MCPConfigError):
            MCPProxy(config=["nope"], env_var=None, config_file=None)

    def test_load_from_config_file(self, clean_env, tmp_path):
        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            json.dumps({"mcpServers": {"weather": {"command": "uvx"}}}),
            encoding="utf-8",
        )
        proxy = MCPProxy(env_var=None, config_file=config_path)
        assert proxy.servers == {"weather": {"command": "uvx"}}

    def test_load_from_claude_default_path(self, clean_env, tmp_path, monkeypatch):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "mcp.json").write_text(
            json.dumps({"mcpServers": {"fs": {"command": "npx"}}}),
            encoding="utf-8",
        )
        fake_home = str(tmp_path)
        monkeypatch.setenv("USERPROFILE", fake_home)
        monkeypatch.setenv("HOME", fake_home)
        proxy = MCPProxy(env_var=None, config_file="~/.claude/mcp.json")
        assert proxy.servers == {"fs": {"command": "npx"}}
        assert proxy.config_sources[0].startswith("file:")

    def test_missing_default_config_file_yields_empty_proxy(
        self, clean_env, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        proxy = MCPProxy(env_var=None, config_file="~/.claude/mcp.json")
        assert proxy.servers == {}
        assert proxy.config_sources == ["none"]

    def test_missing_explicit_config_file_raises(self, clean_env, tmp_path):
        with pytest.raises(FileNotFoundError):
            MCPProxy(env_var=None, config_file=tmp_path / "absent.json")

    def test_invalid_json_config_file_raises(self, clean_env, tmp_path):
        bad = tmp_path / "mcp.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(MCPConfigError):
            MCPProxy(env_var=None, config_file=bad)

    def test_env_var_with_inline_json(self, clean_env, monkeypatch):
        monkeypatch.setenv(
            DEFAULT_CONFIG_ENV_VAR,
            json.dumps({"mcpServers": {"search": {"command": "npx"}}}),
        )
        proxy = MCPProxy(config_file=None)
        assert proxy.servers == {"search": {"command": "npx"}}
        assert proxy.config_sources == [f"env:{DEFAULT_CONFIG_ENV_VAR}"]

    def test_env_var_with_file_path(self, clean_env, monkeypatch, tmp_path):
        config_path = tmp_path / "mcp.json"
        config_path.write_text(
            json.dumps({"mcpServers": {"search": {"command": "npx"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv(DEFAULT_CONFIG_ENV_VAR, str(config_path))
        proxy = MCPProxy(config_file=None)
        assert proxy.servers == {"search": {"command": "npx"}}

    def test_env_var_missing_path_raises(self, clean_env, monkeypatch, tmp_path):
        monkeypatch.setenv(DEFAULT_CONFIG_ENV_VAR, str(tmp_path / "gone.json"))
        with pytest.raises(FileNotFoundError):
            MCPProxy(config_file=None)

    def test_env_var_takes_precedence_over_default_file(
        self, clean_env, monkeypatch, tmp_path
    ):
        env_file = tmp_path / "env.json"
        env_file.write_text(
            json.dumps({"mcpServers": {"from-env": {"command": "env"}}}),
            encoding="utf-8",
        )
        home_file = tmp_path / "home.json"
        home_file.write_text(
            json.dumps({"mcpServers": {"from-file": {"command": "file"}}}),
            encoding="utf-8",
        )
        monkeypatch.setenv(DEFAULT_CONFIG_ENV_VAR, str(env_file))
        proxy = MCPProxy(config_file=home_file)
        assert list(proxy.servers) == ["from-env"]

    def test_inline_config_takes_precedence_over_env_var(self, clean_env, monkeypatch):
        monkeypatch.setenv(
            DEFAULT_CONFIG_ENV_VAR,
            json.dumps({"mcpServers": {"from-env": {"command": "env"}}}),
        )
        proxy = MCPProxy(config={"from-inline": {"command": "inline"}}, config_file=None)
        assert list(proxy.servers) == ["from-inline"]

    def test_env_placeholders_are_expanded(self, clean_env, monkeypatch):
        monkeypatch.setenv("MCP_TEST_TOKEN", "secret-token")
        proxy = MCPProxy(
            config={
                "search": {
                    "command": "npx",
                    "args": ["-y", "${MCP_TEST_TOKEN}"],
                    "env": {"API_KEY": "${MCP_TEST_TOKEN}"},
                }
            },
            env_var=None,
            config_file=None,
        )
        server = proxy.servers["search"]
        assert server["args"] == ["-y", "secret-token"]
        assert server["env"]["API_KEY"] == "secret-token"


# ---------------------------------------------------------------------------
# Tool conversion & registration
# ---------------------------------------------------------------------------


def make_proxy(transport, servers):
    # Tag each server config with its name so FakeTransport can key responses;
    # MCPProxy treats server configs as opaque dicts, so this is transparent.
    tagged = {name: {**cfg, "_name": name} for name, cfg in servers.items()}
    return MCPProxy(
        config=tagged, env_var=None, config_file=None, transport=transport
    )


class TestToolConversion:
    def test_tools_are_converted_to_anthropic_format(self):
        proxy = make_proxy(
            FakeTransport(tools={"weather": WEATHER_TOOLS}),
            {"weather": {"command": "uvx"}},
        )
        tools = proxy.get_anthropic_tools()
        assert tools == [
            {
                "name": "mcp__weather__get_forecast",
                "description": "Get the weather forecast for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    def test_input_schema_defaults_to_empty_object(self):
        proxy = make_proxy(
            FakeTransport(tools={"weather": [{"name": "ping"}]}),
            {"weather": {"command": "uvx"}},
        )
        (tool,) = proxy.get_anthropic_tools()
        assert tool["input_schema"] == {"type": "object", "properties": {}}

    def test_description_falls_back_when_missing(self):
        proxy = make_proxy(
            FakeTransport(tools={"weather": [{"name": "ping"}]}),
            {"weather": {"command": "uvx"}},
        )
        (tool,) = proxy.get_anthropic_tools()
        assert tool["description"] == (
            "MCP tool 'ping' provided by server 'weather'."
        )

    def test_server_and_tool_names_are_sanitized(self):
        proxy = make_proxy(
            FakeTransport(tools={"My Server.X": [{"name": "do thing"}]}),
            {"My Server.X": {"command": "uvx"}},
        )
        (tool,) = proxy.get_anthropic_tools()
        assert tool["name"] == "mcp__My_Server_X__do_thing"

    def test_mcp_tool_dataclass_to_anthropic(self):
        tool = MCPTool(server="weather", name="get_forecast", description="d")
        assert tool.qualified_name == "mcp__weather__get_forecast"
        assert tool.to_anthropic()["name"] == tool.qualified_name

    def test_inject_tools_appends_to_existing_declaration(self):
        proxy = make_proxy(
            FakeTransport(tools={"weather": WEATHER_TOOLS}),
            {"weather": {"command": "uvx"}},
        )
        existing = [
            {
                "name": "bash",
                "description": "Run a shell command",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        merged = proxy.inject_tools(existing)
        assert [tool["name"] for tool in merged] == [
            "bash",
            "mcp__weather__get_forecast",
        ]
        # The input list must not be mutated.
        assert existing == [merged[0]]

    def test_inject_tools_skips_duplicate_names(self):
        proxy = make_proxy(
            FakeTransport(tools={"weather": WEATHER_TOOLS}),
            {"weather": {"command": "uvx"}},
        )
        existing = [
            {
                "name": "mcp__weather__get_forecast",
                "description": "upstream declaration wins",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        merged = proxy.inject_tools(existing)
        assert len(merged) == 1
        assert merged[0]["description"] == "upstream declaration wins"

    def test_list_tools_from_multiple_servers(self):
        proxy = make_proxy(
            FakeTransport(tools={"weather": WEATHER_TOOLS, "search": SEARCH_TOOLS}),
            {
                "weather": {"command": "uvx"},
                "search": {"command": "npx"},
            },
        )
        names = [tool.qualified_name for tool in proxy.list_tools()]
        assert names == [
            "mcp__weather__get_forecast",
            "mcp__search__web_search",
        ]

    def test_list_tools_single_unknown_server_raises(self):
        proxy = make_proxy(FakeTransport(), {"weather": {"command": "uvx"}})
        with pytest.raises(KeyError):
            proxy.list_tools(server="nope")

    def test_list_tools_with_empty_config_returns_nothing(self, clean_env):
        proxy = MCPProxy(env_var=None, config_file=None)
        assert proxy.servers == {}
        assert proxy.list_tools() == []

    def test_list_tools_caches_per_server(self):
        transport = FakeTransport(tools={"weather": WEATHER_TOOLS})
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        proxy.list_tools()
        proxy.list_tools()
        assert len(transport.calls) == 1
        proxy.list_tools(refresh=True)
        assert len(transport.calls) == 2
        proxy.refresh()
        proxy.list_tools()
        assert len(transport.calls) == 3

    def test_list_tools_skips_failing_server_but_records_error(self):
        class FlakyTransport(FakeTransport):
            def request(self, server_config, method, params=None):
                if server_config.get("_name") == "broken":
                    raise RuntimeError("boom")
                return super().request(server_config, method, params)

        proxy = make_proxy(
            FlakyTransport(tools={"weather": WEATHER_TOOLS}),
            {"broken": {"command": "x"}, "weather": {"command": "uvx"}},
        )
        tools = proxy.list_tools()
        assert [tool.qualified_name for tool in tools] == [
            "mcp__weather__get_forecast"
        ]
        assert "broken" in proxy.last_errors
        assert "boom" in proxy.last_errors["broken"]

    def test_list_tools_can_raise_on_error(self):
        transport = FakeTransport(error=RuntimeError("boom"))
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        with pytest.raises(RuntimeError, match="boom"):
            proxy.list_tools(raise_on_error=True)


# ---------------------------------------------------------------------------
# Execution proxying
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_execute_tool_routes_to_correct_server(self):
        transport = FakeTransport(tools={"weather": WEATHER_TOOLS})
        proxy = make_proxy(transport, {"weather": {"command": "uvx", "args": ["w"]}})
        proxy.list_tools()
        result = proxy.execute_tool(
            "mcp__weather__get_forecast", {"city": "Beijing"}
        )
        assert result == {
            "content": [{"type": "text", "text": "ok:get_forecast"}],
            "isError": False,
        }
        server_config, method, params = transport.calls[-1]
        assert server_config == {"command": "uvx", "args": ["w"], "_name": "weather"}
        assert method == "tools/call"
        assert params == {
            "name": "get_forecast",
            "arguments": {"city": "Beijing"},
        }

    def test_execute_tool_defaults_arguments_to_empty_dict(self):
        transport = FakeTransport()
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        proxy.execute_tool("mcp__weather__ping")
        _, _, params = transport.calls[-1]
        assert params == {"name": "ping", "arguments": {}}

    def test_execute_tool_rejects_non_mcp_name(self):
        proxy = make_proxy(FakeTransport(), {"weather": {"command": "uvx"}})
        with pytest.raises(ValueError, match="Not an MCP tool name"):
            proxy.execute_tool("bash", {})

    def test_execute_tool_rejects_unknown_server(self):
        proxy = make_proxy(FakeTransport(), {"weather": {"command": "uvx"}})
        with pytest.raises(KeyError, match="Unknown MCP server"):
            proxy.execute_tool("mcp__ghost__tool", {})

    def test_call_tool_wraps_result_as_anthropic_tool_result(self):
        transport = FakeTransport(
            results={
                "get_forecast": {
                    "content": [{"type": "text", "text": "Sunny, 22C"}],
                    "isError": False,
                }
            }
        )
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        result = proxy.call_tool(
            "mcp__weather__get_forecast", {"city": "Beijing"}, tool_use_id="toolu_1"
        )
        assert result == {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [{"type": "text", "text": "Sunny, 22C"}],
            "is_error": False,
        }

    def test_call_tool_propagates_mcp_is_error(self):
        transport = FakeTransport(
            results={
                "get_forecast": {
                    "content": [{"type": "text", "text": "city not found"}],
                    "isError": True,
                }
            }
        )
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        result = proxy.call_tool("mcp__weather__get_forecast", {})
        assert result["is_error"] is True

    def test_call_tool_converts_transport_failure_into_error_block(self):
        transport = FakeTransport(error=RuntimeError("server exploded"))
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        result = proxy.call_tool(
            "mcp__weather__get_forecast", {}, tool_use_id="toolu_2"
        )
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_2"
        assert result["is_error"] is True
        assert "server exploded" in result["content"][0]["text"]

    def test_call_tool_serializes_non_text_content_blocks(self):
        transport = FakeTransport(
            results={
                "get_forecast": {
                    "content": [
                        {"type": "text", "text": "Sunny"},
                        {"type": "resource", "uri": "file:///x.json"},
                    ]
                }
            }
        )
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        result = proxy.call_tool("mcp__weather__get_forecast", {})
        assert result["content"][0] == {"type": "text", "text": "Sunny"}
        assert result["content"][1]["type"] == "text"
        assert "file:///x.json" in result["content"][1]["text"]

    def test_call_tool_serializes_empty_result(self):
        transport = FakeTransport(results={"get_forecast": {}})
        proxy = make_proxy(transport, {"weather": {"command": "uvx"}})
        result = proxy.call_tool("mcp__weather__get_forecast", {})
        assert result["content"][0]["type"] == "text"
        assert json.loads(result["content"][0]["text"]) == {}


# ---------------------------------------------------------------------------
# Stdio JSON-RPC transport (real subprocess integration)
# ---------------------------------------------------------------------------

FAKE_MCP_SERVER_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        method = message.get("method")
        if method == "initialize":
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "fake", "version": "0.0.1"},
                },
            }
        elif method == "tools/list":
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo the given text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        }
                    ]
                },
            }
        elif method == "tools/call":
            text = message["params"]["arguments"]["text"]
            reply = {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "content": [{"type": "text", "text": "echo:" + text}],
                    "isError": False,
                },
            }
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32601, "message": "method not found"},
            }
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
    """
)


class TestStdioTransport:
    @pytest.fixture
    def server_script(self, tmp_path):
        script = tmp_path / "fake_mcp_server.py"
        script.write_text(FAKE_MCP_SERVER_SCRIPT, encoding="utf-8")
        return script

    def test_transport_rejects_config_without_command(self):
        transport = StdioMCPTransport()
        with pytest.raises(MCPConfigError):
            transport.request({"url": "https://example.com/mcp"}, "tools/list")

    def test_end_to_end_stdio_roundtrip(self, server_script):
        proxy = MCPProxy(
            config={
                "local": {"command": sys.executable, "args": [str(server_script)]}
            },
            env_var=None,
            config_file=None,
        )
        tools = proxy.list_tools()
        assert [tool.qualified_name for tool in tools] == ["mcp__local__echo"]

        result = proxy.call_tool(
            "mcp__local__echo", {"text": "hello"}, tool_use_id="toolu_stdio"
        )
        assert result["is_error"] is False
        assert result["content"] == [{"type": "text", "text": "echo:hello"}]

    def test_stdio_transport_surfaces_jsonrpc_error(self, server_script):
        transport = StdioMCPTransport()
        server_config = {"command": sys.executable, "args": [str(server_script)]}
        result = transport.request(server_config, "tools/list")
        assert [tool["name"] for tool in result["tools"]] == ["echo"]
        with pytest.raises(MCPTransportError, match="method not found"):
            transport.request(server_config, "bogus/method")
