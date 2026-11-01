"""Local MCP (Model Context Protocol) relay probe.

AgentBridge discovers the MCP servers configured on the local machine
(``~/.claude/mcp.json`` or the ``AGENTBRIDGE_MCP_CONFIG`` environment
variable), probes them for their tool catalogues, and dynamically re-publishes
those tools as Anthropic Tools declarations so that *any* upstream model can
invoke them through the gateway. Tool execution calls coming back from the
model are routed to the originating MCP server via a stdio JSON-RPC transport.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

__all__ = [
    "DEFAULT_CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_FILE",
    "MCP_PROTOCOL_VERSION",
    "MCPConfigError",
    "MCPProxy",
    "MCPTool",
    "MCPTransportError",
    "StdioMCPTransport",
    "build_mcp_tool_name",
    "is_mcp_tool",
    "parse_mcp_tool_name",
    "sanitize_tool_name",
]

#: Wire protocol version advertised during the MCP ``initialize`` handshake.
MCP_PROTOCOL_VERSION = "2024-11-05"

#: Client identity sent to MCP servers.
CLIENT_INFO = {"name": "agentbridge", "version": "0.1.0"}

#: Default on-disk location of the Claude Code MCP configuration.
DEFAULT_CONFIG_FILE = "~/.claude/mcp.json"

#: Environment variable holding MCP server configuration (inline JSON or a
#: path to a JSON file). Takes precedence over :data:`DEFAULT_CONFIG_FILE`.
DEFAULT_CONFIG_ENV_VAR = "AGENTBRIDGE_MCP_CONFIG"

#: Prefix given to every re-published Anthropic tool name.
MCP_TOOL_NAME_PREFIX = "mcp__"

#: Anthropic tool names are limited to 64 characters of [a-zA-Z0-9_-].
TOOL_NAME_MAX_LEN = 64

_INVALID_TOOL_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}


class MCPConfigError(ValueError):
    """Raised when the MCP server configuration cannot be parsed."""


class MCPTransportError(RuntimeError):
    """Raised when an MCP JSON-RPC exchange fails at the transport level."""


class MCPTransport(Protocol):
    """Minimal transport protocol used by :class:`MCPProxy`.

    Implementations receive the (already env-expanded) server configuration
    block and a JSON-RPC ``method``/``params`` pair, and must return the
    ``result`` object of the response. Errors are signalled by raising.
    """

    def request(
        self,
        server_config: Dict[str, Any],
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:  # pragma: no cover - protocol definition
        ...


def sanitize_tool_name(name: str) -> str:
    """Coerce ``name`` into a valid Anthropic tool name segment.

    Characters outside ``[a-zA-Z0-9_-]`` are replaced with underscores and the
    result is truncated to 64 characters.
    """
    cleaned = _INVALID_TOOL_CHARS.sub("_", name)[:TOOL_NAME_MAX_LEN]
    return cleaned or "tool"


def build_mcp_tool_name(server: str, tool: str) -> str:
    """Build the fully-qualified Anthropic tool name for an MCP tool.

    Follows the ``mcp__<server>__<tool>`` convention used by Claude Code.
    """
    return f"{MCP_TOOL_NAME_PREFIX}{sanitize_tool_name(server)}__{sanitize_tool_name(tool)}"


def parse_mcp_tool_name(name: str) -> Optional[Tuple[str, str]]:
    """Split a qualified ``mcp__<server>__<tool>`` name into its parts.

    Returns ``None`` when ``name`` is not an MCP tool name.
    """
    if not name.startswith(MCP_TOOL_NAME_PREFIX):
        return None
    remainder = name[len(MCP_TOOL_NAME_PREFIX) :]
    if "__" not in remainder:
        return None
    server, tool = remainder.split("__", 1)
    if not server or not tool:
        return None
    return server, tool


def is_mcp_tool(name: str) -> bool:
    """Return ``True`` when ``name`` refers to a proxied MCP tool."""
    return parse_mcp_tool_name(name) is not None


def _expand_env_placeholders(value: Any) -> Any:
    """Recursively expand ``${VAR}`` placeholders in a server configuration."""
    if isinstance(value, str):
        return _ENV_PLACEHOLDER.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    return value


def _parse_config_document(document: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a raw config document into a ``{name: server_config}`` dict.

    Accepts both the Claude Code wrapping ``{"mcpServers": {...}}`` and a bare
    ``{name: {...}}`` mapping. Entries that are not dicts are ignored.
    """
    if not isinstance(document, dict):
        raise MCPConfigError(
            f"MCP configuration must be a JSON object, got {type(document).__name__}"
        )
    servers = document.get("mcpServers", document)
    if not isinstance(servers, dict):
        raise MCPConfigError(
            "'mcpServers' must be an object mapping server names to configs"
        )
    return {str(name): cfg for name, cfg in servers.items() if isinstance(cfg, dict)}


@dataclass
class MCPTool:
    """A single tool exposed by an MCP server."""

    server: str
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=lambda: dict(_EMPTY_SCHEMA))

    @property
    def qualified_name(self) -> str:
        """Fully-qualified Anthropic tool name (``mcp__<server>__<tool>``)."""
        return build_mcp_tool_name(self.server, self.name)

    def to_anthropic(self) -> Dict[str, Any]:
        """Render the tool as an Anthropic Tools declaration."""
        return {
            "name": self.qualified_name,
            "description": self.description
            or f"MCP tool '{self.name}' provided by server '{self.server}'.",
            "input_schema": self.input_schema or dict(_EMPTY_SCHEMA),
        }


class StdioMCPTransport:
    """JSON-RPC transport that talks to a local MCP server over stdio.

    Each request spawns the configured ``command`` (with ``args``/``env`` from
    the server configuration), performs the MCP ``initialize`` handshake,
    issues the JSON-RPC call, and terminates the child process.
    """

    def __init__(
        self,
        protocol_version: str = MCP_PROTOCOL_VERSION,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self.protocol_version = protocol_version
        self.shutdown_timeout = shutdown_timeout

    def request(
        self,
        server_config: Dict[str, Any],
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one JSON-RPC request against the MCP server."""
        command = server_config.get("command")
        if not command:
            raise MCPConfigError(
                "MCP server config must define a local 'command' for the "
                f"stdio transport, got: {server_config!r}"
            )
        args = [str(arg) for arg in server_config.get("args", [])]
        env = os.environ.copy()
        for key, value in (server_config.get("env") or {}).items():
            env[str(key)] = str(_expand_env_placeholders(value))

        process = subprocess.Popen(
            [str(command), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=server_config.get("cwd") or None,
        )
        stderr_tail: List[str] = []
        drain = threading.Thread(
            target=self._drain_stderr, args=(process, stderr_tail), daemon=True
        )
        drain.start()
        try:
            self._send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": self.protocol_version,
                        "capabilities": {},
                        "clientInfo": CLIENT_INFO,
                    },
                },
            )
            init_response = self._receive(process, 1, stderr_tail)
            if "error" in init_response:
                raise MCPTransportError(
                    f"MCP initialize failed: {init_response['error']!r}"
                )
            # Notifications carry no id and expect no response.
            self._send(
                process, {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            self._send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": method,
                    "params": params if params is not None else {},
                },
            )
            response = self._receive(process, 2, stderr_tail)
        finally:
            self._shutdown(process)

        if "error" in response:
            raise MCPTransportError(
                f"MCP request '{method}' failed: {response['error']!r}"
            )
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    @staticmethod
    def _send(process: subprocess.Popen, message: Dict[str, Any]) -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPTransportError(
                f"MCP server process closed its stdin: {exc}"
            ) from exc

    @staticmethod
    def _receive(
        process: subprocess.Popen,
        request_id: int,
        stderr_tail: List[str],
    ) -> Dict[str, Any]:
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if not line:
                detail = "".join(stderr_tail).strip()[-500:]
                raise MCPTransportError(
                    "MCP server closed stdout before answering request "
                    f"id={request_id}. stderr: {detail or '<empty>'}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("id") == request_id:
                return message

    @staticmethod
    def _drain_stderr(process: subprocess.Popen, tail: List[str]) -> None:
        assert process.stderr is not None
        try:
            for line in iter(process.stderr.readline, ""):
                tail.append(line)
                del tail[:-50]
        except (OSError, ValueError):  # pragma: no cover - stream torn down
            pass

    def _shutdown(self, process: subprocess.Popen) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:  # pragma: no cover - already closed
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                process.kill()
                process.wait(timeout=self.shutdown_timeout)


class MCPProxy:
    """Relay between Anthropic Tools and locally configured MCP servers.

    Configuration resolution order:

    1. ``config`` — an inline ``{name: server_config}`` dict or a
       ``{"mcpServers": {...}}`` document;
    2. the environment variable ``env_var`` — either inline JSON or a path to
       a JSON file;
    3. ``config_file`` — defaults to ``~/.claude/mcp.json``. A missing default
       file yields an empty (but functional) proxy.

    Example:
        >>> proxy = MCPProxy(config={"weather": {"command": "uvx",
        ...     "args": ["mcp-server-weather"]}})
        >>> tools = proxy.get_anthropic_tools()      # doctest: +SKIP
        >>> result = proxy.call_tool("mcp__weather__get_forecast",  # doctest: +SKIP
        ...     {"city": "Beijing"})
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        env_var: Optional[str] = DEFAULT_CONFIG_ENV_VAR,
        config_file: Union[str, Path, None] = DEFAULT_CONFIG_FILE,
        transport: Optional[MCPTransport] = None,
        config_path: Union[str, Path, None] = None,
    ) -> None:
        """Build a relay.

        Args:
            config: Inline server configuration (highest priority).
            env_var: Environment variable holding inline JSON or a file path.
            config_file: JSON file to fall back to (default
                ``~/.claude/mcp.json``; missing default file is tolerated).
            transport: Optional transport override (used by tests).
            config_path: Explicit path to an ``mcp.json`` file. When given it
                takes precedence over ``config_file``.
        """
        if config_path is not None:
            config_file = config_path
        self.transport: MCPTransport = transport or StdioMCPTransport()
        self.last_errors: Dict[str, str] = {}
        self.config_sources: List[str] = []
        self._servers: Dict[str, Dict[str, Any]] = {}
        self._tool_cache: Dict[str, List[MCPTool]] = {}

        self._servers = self._resolve_config(config, env_var, config_file)

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------

    def _resolve_config(
        self,
        config: Optional[Dict[str, Any]],
        env_var: Optional[str],
        config_file: Union[str, Path, None],
    ) -> Dict[str, Dict[str, Any]]:
        if config is not None:
            self.config_sources.append("inline:config")
            return {
                name: _expand_env_placeholders(server)
                for name, server in _parse_config_document(config).items()
            }

        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                self.config_sources.append(f"env:{env_var}")
                return self._parse_env_value(env_value)

        if config_file is not None:
            path = Path(config_file).expanduser()
            is_default = str(config_file) == DEFAULT_CONFIG_FILE
            if path.is_file():
                self.config_sources.append(f"file:{path}")
                return self._load_file(path)
            if not is_default:
                raise FileNotFoundError(f"MCP config file not found: {path}")

        # No configuration anywhere: an empty relay is still operational.
        self.config_sources.append("none")
        return {}

    @staticmethod
    def _parse_env_value(env_value: str) -> Dict[str, Dict[str, Any]]:
        stripped = env_value.strip()
        if stripped.startswith("{"):
            return {
                name: _expand_env_placeholders(server)
                for name, server in _parse_config_document(json.loads(stripped)).items()
            }
        path = Path(stripped).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{DEFAULT_CONFIG_ENV_VAR} points to a missing file: {path}"
            )
        return MCPProxy._load_file(path)

    @staticmethod
    def _load_file(path: Path) -> Dict[str, Dict[str, Any]]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MCPConfigError(f"Invalid JSON in MCP config {path}: {exc}") from exc
        return {
            name: _expand_env_placeholders(server)
            for name, server in _parse_config_document(document).items()
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def servers(self) -> Dict[str, Dict[str, Any]]:
        """The resolved ``{name: server_config}`` mapping."""
        return dict(self._servers)

    @property
    def tools(self) -> List[MCPTool]:
        """Every currently active MCP tool (served from the probe cache)."""
        return self.list_tools()

    def list_tools(
        self,
        server: Optional[str] = None,
        refresh: bool = False,
        raise_on_error: bool = False,
    ) -> List[MCPTool]:
        """Probe MCP servers and return their tools as :class:`MCPTool` items.

        Args:
            server: Restrict probing to a single server. ``None`` probes all.
            refresh: Bypass the per-server tool cache and re-probe.
            raise_on_error: Re-raise transport failures instead of recording
                them in :attr:`last_errors` and skipping the server.

        Raises:
            KeyError: If ``server`` is not part of the configuration.
        """
        if server is not None and server not in self._servers:
            raise KeyError(f"Unknown MCP server: {server!r}")
        targets = [server] if server is not None else list(self._servers)

        tools: List[MCPTool] = []
        self.last_errors = {}
        for name in targets:
            if not refresh and name in self._tool_cache:
                tools.extend(self._tool_cache[name])
                continue
            try:
                response = self._rpc(name, "tools/list", {})
                raw_tools = (
                    response.get("tools", []) if isinstance(response, dict) else []
                )
                parsed = [
                    MCPTool(
                        server=name,
                        name=str(raw["name"]),
                        description=str(raw.get("description", "")),
                        input_schema=raw.get("inputSchema") or dict(_EMPTY_SCHEMA),
                    )
                    for raw in raw_tools
                    if isinstance(raw, dict) and raw.get("name")
                ]
                self._tool_cache[name] = parsed
                tools.extend(parsed)
            except Exception as exc:  # noqa: BLE001 - probe must be resilient
                self.last_errors[name] = str(exc)
                if raise_on_error:
                    raise
        return tools

    def get_anthropic_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Return every MCP tool rendered as an Anthropic Tools declaration."""
        return [tool.to_anthropic() for tool in self.list_tools(refresh=refresh)]

    def inject_tools(
        self, anthropic_tools: List[Dict[str, Any]], refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Append MCP tool declarations to an existing Anthropic ``tools`` list.

        The input list is not mutated. MCP tools whose qualified name is
        already present are skipped so upstream declarations win.
        """
        existing = {
            tool.get("name") for tool in anthropic_tools if isinstance(tool, dict)
        }
        merged = list(anthropic_tools)
        for tool in self.list_tools(refresh=refresh):
            if tool.qualified_name not in existing:
                merged.append(tool.to_anthropic())
        return merged

    def refresh(self) -> None:
        """Drop the tool cache so the next probe re-queries every server."""
        self._tool_cache.clear()

    # ------------------------------------------------------------------
    # Lifecycle (async interface for the gateway server)
    # ------------------------------------------------------------------

    async def start_all(self) -> None:
        """Start the relay: probe every configured server for its tools.

        Transport failures are recorded in :attr:`last_errors` and skipped so
        one broken server never blocks gateway startup.
        """
        self.list_tools()

    async def stop_all(self) -> None:
        """Stop the relay: drop cached tool catalogues and probe errors."""
        self._tool_cache.clear()
        self.last_errors = {}

    # ------------------------------------------------------------------
    # Execution proxying
    # ------------------------------------------------------------------

    def execute_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Invoke an MCP tool by its qualified name and return the raw result.

        Raises:
            ValueError: If ``name`` is not a qualified MCP tool name.
            KeyError: If the target server is not configured.
        """
        parsed = parse_mcp_tool_name(name)
        if parsed is None:
            raise ValueError(f"Not an MCP tool name: {name!r}")
        server, tool = parsed
        if server not in self._servers:
            raise KeyError(f"Unknown MCP server: {server!r}")
        return self._rpc(
            server, "tools/call", {"name": tool, "arguments": arguments or {}}
        )

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        tool_use_id: str = "",
    ) -> Dict[str, Any]:
        """Invoke an MCP tool and wrap the outcome as an Anthropic ``tool_result``.

        Transport failures and MCP-level ``isError`` results are surfaced via
        ``is_error: true`` rather than exceptions, matching what the Anthropic
        Messages API expects the client to report back to the model.
        """
        try:
            result = self.execute_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 - errors become tool_result blocks
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [{"type": "text", "text": f"MCP tool call failed: {exc}"}],
                "is_error": True,
            }
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": self._convert_content(result),
            "is_error": bool(result.get("isError", False)),
        }

    @staticmethod
    def _convert_content(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Map an MCP ``tools/call`` result onto Anthropic content blocks."""
        content = result.get("content")
        if not isinstance(content, list) or not content:
            text = json.dumps(result, ensure_ascii=False)
            return [{"type": "text", "text": text}]
        blocks: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append({"type": "text", "text": str(block.get("text", ""))})
            else:
                blocks.append(
                    {"type": "text", "text": json.dumps(block, ensure_ascii=False)}
                )
        return blocks

    def _rpc(
        self, server: str, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.transport.request(self._servers[server], method, params)
