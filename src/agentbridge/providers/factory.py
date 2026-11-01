"""Upstream provider factory with automatic GLM routing.

Selects the concrete :class:`UpstreamClient` implementation for an
:class:`UpstreamConfig` based on the provider name and endpoint, so callers
never need to know which dialect the upstream speaks.
"""

from agentbridge.config import UpstreamConfig
from agentbridge.providers.client import UpstreamClient
from agentbridge.providers.glm import GLMProvider

__all__ = [
    "GLM_HOST_KEYWORDS",
    "create_upstream_client",
    "detect_provider",
    "is_glm_upstream",
]

#: Substrings in ``base_url`` that identify a Zhipu BigModel / Z.ai endpoint.
GLM_HOST_KEYWORDS = ("bigmodel.cn", "z.ai")


def is_glm_upstream(config: UpstreamConfig) -> bool:
    """Return ``True`` when ``config`` targets the GLM native provider."""
    if config.name.lower() == "glm":
        return True
    return any(keyword in config.base_url for keyword in GLM_HOST_KEYWORDS)


def detect_provider(config: UpstreamConfig) -> str:
    """Public name of the provider the factory would select.

    Returns ``"glm"`` or ``"openai"``; used by the CLI startup banner.
    """
    return "glm" if is_glm_upstream(config) else "openai"


def create_upstream_client(config: UpstreamConfig) -> UpstreamClient:
    """Build the upstream client matching ``config``.

    Routes to :class:`GLMProvider` when the provider is named ``"glm"`` (case
    insensitive) or the base URL points at BigModel/Z.ai; everything else
    falls back to the standard OpenAI-compatible :class:`UpstreamClient`.
    """
    if is_glm_upstream(config):
        return GLMProvider(config)
    return UpstreamClient(config)
