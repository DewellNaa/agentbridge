"""Pytest fixtures for AgentBridge tests."""

import pytest

from agentbridge.adapters.anthropic_to_openai import AnthropicToOpenAIAdapter
from agentbridge.config import BridgeConfig, UpstreamConfig


@pytest.fixture
def mock_config():
    """Provides a sample BridgeConfig."""
    return BridgeConfig(
        host="127.0.0.1",
        port=8080,
        upstream=UpstreamConfig(
            name="test-upstream",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            default_model="gpt-4o",
        ),
        model_mapping={
            "claude-3-5-sonnet-20241022": "deepseek-chat",
            "claude-3-7-sonnet-20250219": "deepseek-reasoner",
        },
    )


@pytest.fixture
def adapter(mock_config):
    """Provides an initialized AnthropicToOpenAIAdapter."""
    return AnthropicToOpenAIAdapter(model_map=mock_config.model_mapping)
