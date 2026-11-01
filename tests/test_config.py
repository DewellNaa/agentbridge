"""Unit tests for BridgeConfig environment parsing."""

import pytest

from agentbridge.config import BridgeConfig, UpstreamConfig


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
        "AGENTBRIDGE_OLLAMA",
        "OLLAMA_HOST",
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


# ---------------------------------------------------------------------------
# Ollama environment support
# ---------------------------------------------------------------------------


class TestOllamaEnv:
    def test_ollama_mode_disabled_by_default(self, clean_env):
        config = BridgeConfig.from_env()
        assert config.ollama_mode is False
        assert config.upstream.ollama_mode is False

    def test_upstream_ollama_mode_defaults_false(self):
        assert UpstreamConfig().ollama_mode is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
    def test_agentbridge_ollama_truthy_enables_mode(self, clean_env, value):
        clean_env.setenv("AGENTBRIDGE_OLLAMA", value)
        config = BridgeConfig.from_env()
        assert config.ollama_mode is True
        assert config.upstream.ollama_mode is True

    def test_ollama_mode_applies_ollama_defaults(self, clean_env):
        clean_env.setenv("AGENTBRIDGE_OLLAMA", "true")
        config = BridgeConfig.from_env()
        assert config.upstream.name == "ollama"
        assert config.upstream.base_url == "http://127.0.0.1:11434/v1"
        assert config.upstream.api_key == "ollama"
        assert config.upstream.default_model == "qwen2.5-coder"

    def test_ollama_host_supplies_endpoint(self, clean_env):
        clean_env.setenv("AGENTBRIDGE_OLLAMA", "true")
        clean_env.setenv("OLLAMA_HOST", "192.168.1.5:11500")
        config = BridgeConfig.from_env()
        assert config.upstream.base_url == "http://192.168.1.5:11500/v1"

    def test_ollama_host_url_form(self, clean_env):
        clean_env.setenv("AGENTBRIDGE_OLLAMA", "true")
        clean_env.setenv("OLLAMA_HOST", "http://localhost:11434")
        config = BridgeConfig.from_env()
        assert config.upstream.base_url == "http://localhost:11434/v1"

    def test_ollama_host_alone_does_not_enable_mode(self, clean_env):
        clean_env.setenv("OLLAMA_HOST", "192.168.1.5:11500")
        config = BridgeConfig.from_env()
        assert config.ollama_mode is False
        assert config.upstream.name == "default"
        assert config.upstream.base_url == "https://api.openai.com/v1"
