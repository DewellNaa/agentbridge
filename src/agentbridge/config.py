"""Configuration management for AgentBridge."""

import os
from typing import Dict

from pydantic import BaseModel, Field


class UpstreamConfig(BaseModel):
    """Configuration for an upstream LLM provider."""

    name: str = "default"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    default_model: str = "gpt-4o"
    timeout: float = 120.0
    extra_headers: Dict[str, str] = Field(default_factory=dict)


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

    @classmethod
    def from_env(cls) -> "BridgeConfig":
        """Load configuration from environment variables with sensible defaults."""
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
        )

        enable_mcp = os.getenv("AGENTBRIDGE_ENABLE_MCP", "").strip().lower() in (
            "true",
            "1",
            "yes",
        )
        mcp_config_path = os.getenv("AGENTBRIDGE_MCP_CONFIG") or None

        return cls(
            host=os.getenv("AGENTBRIDGE_HOST", "127.0.0.1"),
            port=int(os.getenv("AGENTBRIDGE_PORT", "8080")),
            debug=os.getenv("AGENTBRIDGE_DEBUG", "false").lower() == "true",
            upstream=upstream,
            enable_mcp=enable_mcp,
            mcp_config_path=mcp_config_path,
        )
