"""Unit tests for BridgeConfig MCP-related environment parsing."""

import pytest

from agentbridge.config import BridgeConfig


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "AGENTBRIDGE_ENABLE_MCP",
        "AGENTBRIDGE_MCP_CONFIG",
        "AGENTBRIDGE_UPSTREAM_NAME",
        "AGENTBRIDGE_UPSTREAM_URL",
        "AGENTBRIDGE_UPSTREAM_KEY",
        "AGENTBRIDGE_DEFAULT_MODEL",
        "AGENTBRIDGE_HOST",
        "AGENTBRIDGE_PORT",
        "AGENTBRIDGE_DEBUG",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_mcp_disabled_by_default(clean_env):
    config = BridgeConfig.from_env()
    assert config.enable_mcp is False
    assert config.mcp_config_path is None


@pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", "Yes"])
def test_enable_mcp_truthy_values(clean_env, value):
    clean_env.setenv("AGENTBRIDGE_ENABLE_MCP", value)
    assert BridgeConfig.from_env().enable_mcp is True


@pytest.mark.parametrize("value", ["false", "0", "no", "", "off"])
def test_enable_mcp_falsy_values(clean_env, value):
    clean_env.setenv("AGENTBRIDGE_ENABLE_MCP", value)
    assert BridgeConfig.from_env().enable_mcp is False


def test_mcp_config_path_from_env(clean_env):
    clean_env.setenv("AGENTBRIDGE_MCP_CONFIG", "/tmp/mcp.json")
    config = BridgeConfig.from_env()
    assert config.mcp_config_path == "/tmp/mcp.json"


def test_mcp_config_path_empty_env_is_none(clean_env):
    clean_env.setenv("AGENTBRIDGE_MCP_CONFIG", "")
    assert BridgeConfig.from_env().mcp_config_path is None
