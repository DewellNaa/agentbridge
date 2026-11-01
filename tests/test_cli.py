"""Tests for the CLI: run/config/ollama subcommands and argument parsing."""

import sys

import pytest

from agentbridge import cli
from agentbridge import launcher as launcher_module
from agentbridge.config import BridgeConfig
from agentbridge.discovery.ollama import OllamaModelInfo

# ---------------------------------------------------------------------------
# `run` argument parsing
# ---------------------------------------------------------------------------


class TestRunParser:
    def parse(self, argv):
        return cli.create_parser().parse_args(argv)

    def test_agent_only(self):
        args = self.parse(["run", "claude"])
        assert args.agent == "claude"
        assert args.args == []
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.no_auto_start is False

    def test_remainder_args_are_collected(self):
        args = self.parse(["run", "claude", "-p", "explain main.py"])
        assert args.args == ["-p", "explain main.py"]

    def test_double_dash_separator_is_consumed(self):
        args = self.parse(["run", "claude", "--", "-p", "explain main.py"])
        assert args.args == ["-p", "explain main.py"]

    def test_aider_model_flag(self):
        args = self.parse(
            ["run", "aider", "--", "--model", "claude-3-5-sonnet-20241022"]
        )
        assert args.agent == "aider"
        assert args.args == ["--model", "claude-3-5-sonnet-20241022"]

    def test_gateway_options_before_agent(self):
        args = self.parse(
            ["run", "--host", "0.0.0.0", "--port", "9000", "aider", "-m", "x"]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.args == ["-m", "x"]

    def test_no_auto_start_flag(self):
        args = self.parse(["run", "--no-auto-start", "claude"])
        assert args.no_auto_start is True

    def test_upstream_arg_is_repeatable(self):
        args = self.parse(
            [
                "run",
                "--upstream-arg=--model=glm-4.6",
                "--upstream-arg=--enable-mcp",
                "claude",
            ]
        )
        assert args.upstream_arg == ["--model=glm-4.6", "--enable-mcp"]

    def test_agent_is_required(self):
        with pytest.raises(SystemExit) as exc:
            self.parse(["run"])
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# `start --ollama` argument parsing
# ---------------------------------------------------------------------------


class TestOllamaParser:
    def parse(self, argv):
        return cli.create_parser().parse_args(argv)

    def test_ollama_flag_defaults_to_auto(self):
        args = self.parse(["start", "--ollama"])
        assert args.ollama == "auto"

    def test_ollama_with_model_name(self):
        args = self.parse(["start", "--ollama", "qwen2.5-coder:14b"])
        assert args.ollama == "qwen2.5-coder:14b"

    def test_ollama_absent_is_none(self):
        args = self.parse(["start"])
        assert args.ollama is None


# ---------------------------------------------------------------------------
# `start --ollama` wiring
# ---------------------------------------------------------------------------


def patch_discovery(monkeypatch, *, running=True, models=None):
    """Patch the discovery functions used by the CLI."""
    import agentbridge.discovery.ollama as discovery

    monkeypatch.setattr(
        discovery, "check_ollama_running", lambda *a, **k: running, raising=True
    )
    monkeypatch.setattr(
        discovery,
        "discover_ollama_models",
        lambda *a, **k: models if models is not None else [],
        raising=True,
    )


class TestApplyOllamaConfig:
    def test_auto_selects_best_model(self, monkeypatch):
        patch_discovery(
            monkeypatch,
            models=[
                OllamaModelInfo(name="llama3:8b", score=40),
                OllamaModelInfo(name="qwen2.5-coder:14b", score=185),
            ],
        )
        config = BridgeConfig()
        assert cli.apply_ollama_config(config, "auto") == 0
        assert config.ollama_mode is True
        assert config.upstream.name == "ollama"
        assert config.upstream.base_url == "http://127.0.0.1:11434/v1"
        assert config.upstream.default_model == "qwen2.5-coder:14b"
        assert config.upstream.api_key == "ollama"
        assert config.upstream.ollama_mode is True
        assert config.ollama_model_count == 2

    def test_explicit_model_is_used(self, monkeypatch):
        patch_discovery(monkeypatch, models=[])
        config = BridgeConfig()
        assert cli.apply_ollama_config(config, "deepseek-r1:14b") == 0
        assert config.upstream.default_model == "deepseek-r1:14b"
        assert config.upstream.base_url == "http://127.0.0.1:11434/v1"

    def test_not_running_returns_exit_code_one(self, monkeypatch, capsys):
        patch_discovery(monkeypatch, running=False)
        config = BridgeConfig()
        assert cli.apply_ollama_config(config, "auto") == 1
        out = capsys.readouterr().out
        assert "Ollama is not running at http://127.0.0.1:11434" in out
        assert "https://ollama.com" in out
        assert config.ollama_mode is False

    def test_auto_without_models_returns_exit_code_one(self, monkeypatch, capsys):
        patch_discovery(monkeypatch, models=[])
        config = BridgeConfig()
        assert cli.apply_ollama_config(config, "auto") == 1
        out = capsys.readouterr().out
        assert "ollama pull qwen2.5-coder" in out


class TestOllamaCommand:
    def test_main_lists_models_and_highlights_best(self, monkeypatch, capsys):
        patch_discovery(
            monkeypatch,
            models=[
                OllamaModelInfo(
                    name="qwen2.5-coder:14b",
                    score=185,
                    parameter_size="14.8B",
                    quantization_level="Q4_K_M",
                    size_bytes=8_989_452_339,
                ),
                OllamaModelInfo(name="llama3:8b", score=40, parameter_size="8B"),
            ],
        )
        monkeypatch.setattr(sys, "argv", ["agentbridge", "ollama"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "qwen2.5-coder:14b" in out
        assert "llama3:8b" in out
        assert "14.8B" in out
        assert "Q4_K_M" in out
        assert "8.4 GB" in out
        assert "185" in out
        assert "★" in out
        assert "Recommended" in out

    def test_main_exits_one_when_not_running(self, monkeypatch, capsys):
        patch_discovery(monkeypatch, running=False)
        monkeypatch.setattr(sys, "argv", ["agentbridge", "ollama"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "Ollama is not running" in capsys.readouterr().out

    def test_main_empty_library_suggests_pull(self, monkeypatch, capsys):
        patch_discovery(monkeypatch, models=[])
        monkeypatch.setattr(sys, "argv", ["agentbridge", "ollama"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert "ollama pull qwen2.5-coder" in capsys.readouterr().out

    def test_main_start_ollama_not_running_exits_one(self, monkeypatch, capsys):
        patch_discovery(monkeypatch, running=False)
        monkeypatch.setattr(sys, "argv", ["agentbridge", "start", "--ollama"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Ollama is not running" in out


# ---------------------------------------------------------------------------
# Ollama startup banner
# ---------------------------------------------------------------------------


class TestOllamaBanner:
    def make_ollama_config(self):
        config = BridgeConfig(ollama_mode=True, ollama_model_count=3)
        config.upstream.name = "ollama"
        config.upstream.base_url = "http://127.0.0.1:11434/v1"
        config.upstream.default_model = "qwen2.5-coder:14b"
        config.upstream.api_key = "ollama"
        return config

    def test_banner_shows_ollama_discovery(self, capsys):
        cli.print_startup_banner(self.make_ollama_config())
        out = capsys.readouterr().out
        assert "Ollama Auto-Discovery" in out
        assert "Active (Target: qwen2.5-coder:14b)" in out
        assert "3" in out
        assert "ollama" in out

    def test_banner_omits_ollama_rows_in_normal_mode(self, capsys):
        cli.print_startup_banner(BridgeConfig())
        out = capsys.readouterr().out
        assert "Ollama Auto-Discovery" not in out

    def test_start_with_ollama_prints_banner_and_wires_upstream(
        self, monkeypatch, capsys
    ):
        patch_discovery(
            monkeypatch,
            models=[OllamaModelInfo(name="qwen2.5-coder:14b", score=185)],
        )
        ran = {}
        monkeypatch.setattr(
            cli.uvicorn, "run", lambda app, **kw: ran.setdefault("called", True)
        )
        monkeypatch.setattr(sys, "argv", ["agentbridge", "start", "--ollama"])
        cli.main()
        assert ran.get("called") is True
        out = capsys.readouterr().out
        assert "Ollama Auto-Discovery" in out
        assert "Active (Target: qwen2.5-coder:14b)" in out
        assert "Provider:" in out


# ---------------------------------------------------------------------------
# `config` argument parsing
# ---------------------------------------------------------------------------


class TestConfigParser:
    def parse(self, argv):
        return cli.create_parser().parse_args(argv)

    def test_target_parsed(self):
        args = self.parse(["config", "cursor"])
        assert args.target == "cursor"
        assert args.host == "127.0.0.1"
        assert args.port == 8080

    def test_custom_host_and_port(self):
        args = self.parse(["config", "continue", "--host", "0.0.0.0", "-p", "1234"])
        assert args.host == "0.0.0.0"
        assert args.port == 1234

    def test_target_is_required(self):
        with pytest.raises(SystemExit) as exc:
            self.parse(["config"])
        assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


class TestRunDispatch:
    def test_main_delegates_to_run_agent(self, monkeypatch):
        captured = {}

        def fake_run_agent(agent_args, **kwargs):
            captured["agent_args"] = agent_args
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(launcher_module, "run_agent", fake_run_agent)
        monkeypatch.setattr(sys, "argv", ["agentbridge", "run", "claude", "-p", "hi"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert captured["agent_args"] == ["claude", "-p", "hi"]
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 8080
        assert captured["auto_start"] is True
        assert captured["upstream_args"] is None

    def test_main_passes_no_auto_start(self, monkeypatch):
        captured = {}

        def fake_run_agent(agent_args, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(launcher_module, "run_agent", fake_run_agent)
        monkeypatch.setattr(
            sys, "argv", ["agentbridge", "run", "--no-auto-start", "claude"]
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert captured["auto_start"] is False

    def test_main_propagates_nonzero_exit_code(self, monkeypatch):
        monkeypatch.setattr(launcher_module, "run_agent", lambda *a, **k: 127)
        monkeypatch.setattr(sys, "argv", ["agentbridge", "run", "nosuchagent"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 127

    def test_main_forwards_upstream_args(self, monkeypatch):
        captured = {}

        def fake_run_agent(agent_args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(launcher_module, "run_agent", fake_run_agent)
        monkeypatch.setattr(
            sys,
            "argv",
            ["agentbridge", "run", "--upstream-arg=--model=glm-4.6", "claude"],
        )
        with pytest.raises(SystemExit):
            cli.main()
        assert captured["upstream_args"] == ["--model=glm-4.6"]


class TestConfigDispatch:
    def test_main_prints_cursor_config(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["agentbridge", "config", "cursor"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert ".cursorrules" in out
        assert "http://127.0.0.1:8080" in out

    def test_main_prints_json_config(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["agentbridge", "config", "continue"])
        with pytest.raises(SystemExit):
            cli.main()
        out = capsys.readouterr().out
        assert '"models"' in out
        assert '"apiBase"' in out

    def test_main_unknown_target_lists_alternatives(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["agentbridge", "config", "windsurf"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "cursor" in out
        assert "cline" in out
        assert "continue" in out
