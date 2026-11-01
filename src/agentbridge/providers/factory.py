"""Upstream provider factory with automatic GLM / Ollama routing.

Selects the concrete :class:`UpstreamClient` implementation for an
:class:`UpstreamConfig` based on the provider name and endpoint, so callers
never need to know which dialect the upstream speaks.
"""

from agentbridge.config import UpstreamConfig
from agentbridge.providers.client import UpstreamClient
from agentbridge.providers.glm import GLMProvider
from agentbridge.providers.ollama import OllamaProvider

__all__ = [
    "GLM_HOST_KEYWORDS",
    "OLLAMA_HOST_KEYWORDS",
    "create_upstream_client",
    "detect_provider",
    "is_glm_upstream",
    "is_ollama_upstream",
]

#: Substrings in ``base_url`` that identify a Zhipu BigModel / Z.ai endpoint.
GLM_HOST_KEYWORDS = ("bigmodel.cn", "z.ai")

#: Substrings in ``base_url`` that identify a local Ollama endpoint.
OLLAMA_HOST_KEYWORDS = (":11434", "ollama")


def is_glm_upstream(config: UpstreamConfig) -> bool:
    """Return ``True`` when ``config`` targets the GLM native provider."""
    if config.name.lower() == "glm":
        return True
    return any(keyword in config.base_url for keyword in GLM_HOST_KEYWORDS)


def is_ollama_upstream(config: UpstreamConfig) -> bool:
    """Return ``True`` when ``config`` targets a local Ollama instance."""
    if config.name.lower() == "ollama":
        return True
    base_url = config.base_url.lower()
    return any(keyword in base_url for keyword in OLLAMA_HOST_KEYWORDS)


def detect_provider(config: UpstreamConfig) -> str:
    """Public name of the provider the factory would select.

    Returns ``"ollama"``, ``"glm"`` or ``"openai"``; used by the CLI
    startup banner.
    """
    if is_ollama_upstream(config):
        return "ollama"
    if is_glm_upstream(config):
        return "glm"
    return "openai"


def create_upstream_client(config: UpstreamConfig) -> UpstreamClient:
    """Build the upstream client matching ``config``.

    Routes to :class:`OllamaProvider` when the provider is named
    ``"ollama"`` (case insensitive) or the base URL points at an Ollama
    endpoint, and to :class:`GLMProvider` for BigModel/Z.ai; everything
    else falls back to the standard OpenAI-compatible
    :class:`UpstreamClient`.
    """
    if is_ollama_upstream(config):
        return OllamaProvider(config)
    if is_glm_upstream(config):
        return GLMProvider(config)
    return UpstreamClient(config)
