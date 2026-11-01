"""Tests for IDE configuration template generation."""

import json

from agentbridge.templates import (
    SUPPORTED_TARGETS,
    generate_ide_config,
)


class TestCursorTemplate:
    def test_contains_gateway_url(self):
        out = generate_ide_config("cursor", host="127.0.0.1", port=8080)
        assert "http://127.0.0.1:8080" in out
        assert "http://127.0.0.1:8080/v1/messages" in out

    def test_contains_claude_model_mappings(self):
        out = generate_ide_config("cursor")
        assert "claude-3-7-sonnet-20250219" in out
        assert "claude-3-5-sonnet-20241022" in out

    def test_contains_openai_compatible_endpoint(self):
        out = generate_ide_config("cursor", host="10.0.0.5", port=9999)
        assert "http://10.0.0.5:9999/v1" in out

    def test_contains_cursorrules_example(self):
        out = generate_ide_config("cursor")
        assert ".cursorrules" in out
        assert "```" in out

    def test_contains_api_key_hint(self):
        out = generate_ide_config("cursor")
        assert "agentbridge" in out


class TestClineTemplate:
    def test_output_is_valid_json(self):
        out = generate_ide_config("cline")
        assert json.loads(out) == json.loads(out)  # parseable, deterministic

    def test_json_has_anthropic_provider(self):
        cfg = json.loads(generate_ide_config("cline", host="127.0.0.1", port=8080))
        assert isinstance(cfg, dict)
        assert cfg["apiProvider"] == "anthropic"
        assert cfg["anthropicBaseUrl"] == "http://127.0.0.1:8080"
        assert cfg["anthropicApiKey"] == "agentbridge"

    def test_json_includes_openai_compatible_fallback(self):
        cfg = json.loads(generate_ide_config("cline", host="localhost", port=9000))
        assert cfg["openAiBaseUrl"] == "http://localhost:9000/v1"


class TestContinueTemplate:
    def test_output_is_valid_json(self):
        cfg = json.loads(generate_ide_config("continue"))
        assert isinstance(cfg, dict)

    def test_models_list_shape(self):
        cfg = json.loads(generate_ide_config("continue"))
        models = cfg["models"]
        assert isinstance(models, list)
        assert len(models) >= 1
        for model in models:
            assert {"title", "provider", "model", "apiKey", "apiBase"} <= set(model)

    def test_primary_model_points_at_gateway(self):
        cfg = json.loads(generate_ide_config("continue", host="127.0.0.1", port=8080))
        primary = cfg["models"][0]
        assert primary["apiBase"] == "http://127.0.0.1:8080"
        assert primary["apiKey"] == "agentbridge"

    def test_secondary_model_is_openai_compatible(self):
        cfg = json.loads(generate_ide_config("continue", host="h", port=1))
        assert cfg["models"][1]["apiBase"] == "http://h:1/v1"


class TestTargetDispatch:
    def test_all_supported_targets_produce_content(self):
        for target in SUPPORTED_TARGETS:
            out = generate_ide_config(target)
            assert out.strip()
            assert "Unsupported" not in out

    def test_unknown_target_lists_supported_targets(self):
        out = generate_ide_config("windsurf")
        for target in SUPPORTED_TARGETS:
            assert target in out
        assert "windsurf" in out

    def test_target_matching_is_case_insensitive(self):
        assert generate_ide_config("CURSOR") == generate_ide_config("cursor")
        assert generate_ide_config(" Cline ") == generate_ide_config("cline")

    def test_host_and_port_are_reflected_in_every_target(self):
        for target in SUPPORTED_TARGETS:
            out = generate_ide_config(target, host="192.168.1.5", port=7777)
            assert "192.168.1.5:7777" in out
