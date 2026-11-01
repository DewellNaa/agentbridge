"""Unit tests for the upstream provider factory (agentbridge.providers.factory)."""

import pytest

from agentbridge.config import UpstreamConfig
from agentbridge.providers.client import UpstreamClient
from agentbridge.providers.factory import (
    create_upstream_client,
    detect_provider,
    is_glm_upstream,
)
from agentbridge.providers.glm import GLMProvider


def make_config(**overrides) -> UpstreamConfig:
    """Build an UpstreamConfig with safe defaults for factory tests."""
    defaults = {
        "name": "default",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
    }
    defaults.update(overrides)
    return UpstreamConfig(**defaults)


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "glm"},
        {"name": "GLM"},
        {"name": "Glm", "base_url": "https://api.openai.com/v1"},
        {"base_url": "https://open.bigmodel.cn/api/paas/v4"},
        {"base_url": "https://api.z.ai/api/paas/v4"},
        {"name": "deepseek", "base_url": "https://proxy.example.com/z.ai/v1"},
    ],
)
def test_glm_inputs_route_to_glm_provider(overrides):
    client = create_upstream_client(make_config(**overrides))
    assert isinstance(client, GLMProvider)


@pytest.mark.asyncio
async def test_glm_clients_close_cleanly():
    client = create_upstream_client(make_config(name="glm"))
    await client.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "openai"},
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1"},
        {"name": "glm-proxy", "base_url": "https://api.openai.com/v1"},
    ],
)
def test_non_glm_inputs_fall_back_to_standard_client(overrides):
    client = create_upstream_client(make_config(**overrides))
    assert type(client) is UpstreamClient


@pytest.mark.asyncio
async def test_standard_clients_close_cleanly():
    client = create_upstream_client(make_config(name="deepseek"))
    await client.close()


def test_is_glm_upstream_detection():
    assert is_glm_upstream(make_config(name="glm"))
    assert is_glm_upstream(make_config(base_url="https://open.bigmodel.cn/api/paas/v4"))
    assert is_glm_upstream(make_config(base_url="https://api.z.ai/api/paas/v4"))
    assert not is_glm_upstream(make_config(name="openai"))
    assert not is_glm_upstream(make_config(base_url="https://api.deepseek.com/v1"))


def test_detect_provider_labels():
    assert detect_provider(make_config(name="glm")) == "glm"
    assert (
        detect_provider(make_config(base_url="https://api.z.ai/api/paas/v4")) == "glm"
    )
    assert detect_provider(make_config(name="openai")) == "openai"
