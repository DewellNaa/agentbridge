"""Tests for the CLI: run/config subcommands and argument parsing."""

import sys

import pytest

from agentbridge import cli
from agentbridge import launcher as launcher_module

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
