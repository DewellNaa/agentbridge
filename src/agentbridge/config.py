"""Configuration management for AgentBridge."""

import os
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from agentbridge.discovery.ollama import parse_ollama_host

_TRUTHY = ("true", "1", "yes")

_FAILOVER_STRATEGIES = ("priority", "round_robin")


class UpstreamConfig(BaseModel):
    """Configuration for an upstream LLM provider."""

    name: str = "default"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    default_model: str = "gpt-4o"
    timeout: float = 120.0
    extra_headers: Dict[str, str] = Field(default_factory=dict)

    # Ollama local mode: target the zero-config local Ollama endpoint.
    ollama_mode: bool = False

    # Multi-key rotation & failover pool.
    # ``api_keys`` accepts a comma-separated string (e.g. "key1,key2").
    api_keys: List[str] = Field(default_factory=list)
    fallback_url: str | None = None
    fallback_key: str | None = None
    fallback_model: str | None = None
    failover_strategy: str = "priority"  # "priority" 或 "round_robin"

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, value):
        """Allow ``api_keys="k1,k2"`` by splitting the comma-joined string."""
        if isinstance(value, str):
            return [key.strip() for key in value.split(",") if key.strip()]
        return value

    @field_validator("failover_strategy")
    @classmethod
    def _validate_failover_strategy(cls, value: str) -> str:
        if value not in _FAILOVER_STRATEGIES:
            raise ValueError(
                f"failover_strategy must be one of {_FAILOVER_STRATEGIES}, "
                f"got {value!r}"
            )
        return value


def ollama_base_url() -> str:
    """Build the Ollama OpenAI-compatible base URL from ``OLLAMA_HOST``.

    Falls back to the local default (``http://127.0.0.1:11434/v1``) when
    ``OLLAMA_HOST`` is unset or unparsable.
    """
    host, port = parse_ollama_host(os.getenv("OLLAMA_HOST", ""))
    return f"http://{host}:{port}/v1"


class BridgeConfig(BaseModel):
    """Global configuration for AgentBridge."""

    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    log_level: str = "INFO"

    # Default upstream provider
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)

    # Model mapping: e.g. {"claude-3-5-sonnet-20241022": "deepseek-reasoner"}
    model_mapping: Dict[str, str] = Field(default_factory=dict)

    # Thinking token behavior: "passthrough", "thinking_block", "strip"
    thinking_mode: str = "thinking_block"

    # Local MCP relay: auto-discover and re-publish MCP tools to upstreams
    enable_mcp: bool = False
    mcp_config_path: str | None = None

    # Console logging of per-request metrics (disable via --no-telemetry-log)
    telemetry_log: bool = True

    # Ollama auto-discovery mode (enabled via --ollama / AGENTBRIDGE_OLLAMA)
    ollama_mode: bool = False

    # Number of local models found during Ollama discovery (for the banner)
    ollama_model_count: int = 0

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        """Load configuration from environment variables with sensible defaults."""
        enable_mcp = os.getenv("AGENTBRIDGE_ENABLE_MCP", "").strip().lower() in _TRUTHY
        mcp_config_path = os.getenv("AGENTBRIDGE_MCP_CONFIG") or None

        # Multi-key rotation & failover pool settings.
        api_keys = [
            key.strip()
            for key in os.getenv("AGENTBRIDGE_UPSTREAM_KEYS", "").split(",")
            if key.strip()
        ]
        fallback_url = os.getenv("AGENTBRIDGE_FALLBACK_URL") or None
        fallback_key = os.getenv("AGENTBRIDGE_FALLBACK_KEY") or None
        fallback_model = os.getenv("AGENTBRIDGE_FALLBACK_MODEL") or None
        failover_strategy = (
            os.getenv("AGENTBRIDGE_FAILOVER_STRATEGY", "priority").strip().lower()
            or "priority"
        )

        # Ollama mode is opt-in via AGENTBRIDGE_OLLAMA; OLLAMA_HOST only
        # supplies the host:port defaults used by that mode.
        ollama_mode = os.getenv("AGENTBRIDGE_OLLAMA", "").strip().lower() in _TRUTHY
        pool_kwargs = {
            "api_keys": api_keys,
            "fallback_url": fallback_url,
            "fallback_key": fallback_key,
            "fallback_model": fallback_model,
            "failover_strategy": failover_strategy,
        }
        if ollama_mode:
            upstream = UpstreamConfig(
                name="ollama",
                base_url=ollama_base_url(),
                api_key="ollama",
                default_model=os.getenv("AGENTBRIDGE_DEFAULT_MODEL", "qwen2.5-coder"),
                ollama_mode=True,
                **pool_kwargs,
            )
        else:
            upstream = UpstreamConfig(
                name=os.getenv("AGENTBRIDGE_UPSTREAM_NAME", "default"),
                base_url=os.getenv(
                    "AGENTBRIDGE_UPSTREAM_URL",
                    os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                ),
                api_key=os.getenv(
                    "AGENTBRIDGE_UPSTREAM_KEY",
                    os.getenv("OPENAI_API_KEY", ""),
                ),
                default_model=os.getenv("AGENTBRIDGE_DEFAULT_MODEL", "gpt-4o"),
                **pool_kwargs,
            )

        return cls(
            host=os.getenv("AGENTBRIDGE_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENTBRIDGE_PORT", "8080")),
            debug=os.getenv("AGENTBRIDGE_DEBUG", "false").lower() == "true",
            upstream=upstream,
            enable_mcp=enable_mcp,
            mcp_config_path=mcp_config_path,
            ollama_mode=ollama_mode,
        )
