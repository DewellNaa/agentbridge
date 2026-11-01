"""Providers package for AgentBridge."""

from agentbridge.providers.client import UpstreamClient
from agentbridge.providers.factory import (
    create_upstream_client,
    detect_provider,
    is_glm_upstream,
    is_ollama_upstream,
)
from agentbridge.providers.glm import GLMProvider
from agentbridge.providers.ollama import OllamaProvider

__all__ = [
    "GLMProvider",
    "OllamaProvider",
    "UpstreamClient",
    "create_upstream_client",
    "detect_provider",
    "is_glm_upstream",
    "is_ollama_upstream",
]
