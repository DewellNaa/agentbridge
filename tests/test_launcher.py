"""Tests for the one-shot agent launcher (gateway probe, auto-start, env)."""

import os
import subprocess
import sys
from unittest import mock

import pytest

from agentbridge import launcher
from agentbridge.launcher import (
    check_server_healthy,
    get_agent_env,
    resolve_executable,
    run_agent,
    wait_for_gateway,
)


def make_response(status_code: int):
    return mock.Mock(status_code=status_code)


# ---------------------------------------------------------------------------
# check_server_healthy
# ---------------------------------------------------------------------------


class TestCheckServerHealthy:
    def test_returns_true_on_200(self):
        with mock.patch.object(launcher.httpx, "get", return_value=make_response(200)):
            assert check_server_healthy("127.0.0.1", 8080) is True

    def test_requests_expected_url(self):
        with mock.patch.object(
            launcher.httpx, "get", return_value=make_response(200)
        ) as get:
            check_server_healthy("localhost", 9090, timeout=0.3)
        get.assert_called_once_with("http://localhost:9090/health", timeout=0.3)

    def test_returns_false_on_non_200(self):
        with mock.patch.object(launcher.httpx, "get", return_value=make_response(503)):
            assert check_server_healthy("127.0.0.1", 8080) is False

    def test_returns_false_on_timeout(self):
        with mock.patch.object(
            launcher.httpx, "get", side_effect=__import__("httpx").Timeout("timed out")
        ):
            assert check_server_healthy("127.0.0.1", 8080) is False

    def test_returns_false_on_connection_error(self):
        with mock.patch.object(
            launcher.httpx, "get", side_effect=ConnectionError("refused")
        ):
            assert check_server_healthy("127.0.0.1", 8080) is False

    def test_returns_false_when_nothing_listens(self):
        # Port 1 on loopback is never our gateway; the real httpx call must
        # fail fast and be reported as unhealthy.
        assert check_server_healthy("127.0.0.1", 1, timeout=0.2) is False


# ---------------------------------------------------------------------------
# get_agent_env
# ---------------------------------------------------------------------------


class TestGetAgentEnv:
    def test_claude_gets_plain_base_url(self):
        env = get_agent_env("claude", host="127.0.0.1", port=8080, existing_env={})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"

    def test_aider_gets_v1_suffix(self):
        env = get_agent_env("aider", host="127.0.0.1", port=8080, existing_env={})
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080/v1"

    def test_custom_agent_gets_default_base_url(self):
        env = get_agent_env("my-agent", host="localhost", port=9000, existing_env={})
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:9000"

    def test_api_key_is_injected(self):
        for agent in ("claude", "aider", "my-agent"):
            env = get_agent_env(agent, existing_env={})
            assert env["ANTHROPIC_API_KEY"] == "agentbridge"

    def test_inherits_system_environ_by_default(self):
        env = get_agent_env("claude", existing_env=None)
        assert env["PATH"] == os.environ.get("PATH")
        assert env["ANTHROPIC_API_KEY"] == "agentbridge"

    def test_existing_env_is_preserved_and_not_mutated(self):
        original = {"MY_VAR": "hello", "ANTHROPIC_API_KEY": "secret"}
        env = get_agent_env("claude", existing_env=original)
        assert env["MY_VAR"] == "hello"
        assert env["ANTHROPIC_API_KEY"] == "agentbridge"
        # The caller's dict must not be modified in place.
        assert original["ANTHROPIC_API_KEY"] == "secret"
        assert "ANTHROPIC_BASE_URL" not in original

    def test_returns_string_values(self):
        env = get_agent_env("claude", existing_env={})
        assert all(isinstance(v, str) for v in env.values())


# ---------------------------------------------------------------------------
# wait_for_gateway
# ---------------------------------------------------------------------------


class TestWaitForGateway:
    def test_returns_true_when_immediately_ready(self):
        with mock.patch.object(
            launcher, "check_server_healthy", return_value=True
        ) as check:
            assert wait_for_gateway(timeout=1.0, interval=0.05) is True
        assert check.call_count == 1

    def test_returns_true_after_retries(self):
        with mock.patch.object(
            launcher, "check_server_healthy", side_effect=[False, False, True]
        ):
            assert wait_for_gateway(timeout=5.0, interval=0.01) is True

    def test_returns_false_on_timeout(self):
        with mock.patch.object(launcher, "check_server_healthy", return_value=False):
            assert wait_for_gateway(timeout=0.3, interval=0.05) is False

    def test_timeout_zero_returns_false_without_probe(self):
        with mock.patch.object(
            launcher, "check_server_healthy", return_value=True
        ) as check:
            assert wait_for_gateway(timeout=0, interval=0.01) is False
        assert check.call_count == 0

    def test_polls_expected_host_and_port(self):
        with mock.patch.object(
            launcher, "check_server_healthy", return_value=True
        ) as check:
            wait_for_gateway(host="0.0.0.0", port=1234, timeout=1.0, interval=0.05)
        check.assert_called_with("0.0.0.0", 1234, timeout=0.05)


# ---------------------------------------------------------------------------
# resolve_executable
# ---------------------------------------------------------------------------


class TestResolveExecutable:
    def test_returns_none_for_unknown_command(self):
        assert resolve_executable("definitely-not-a-real-agent-xyz") is None

    def test_resolves_existing_executable(self):
        name = os.path.basename(sys.executable)
        resolved = resolve_executable(name)
        assert resolved is not None
        assert name.lower() in resolved.lower()

    def test_delegates_to_shutil_which(self):
        with mock.patch.object(
            launcher.shutil, "which", return_value="/usr/bin/claude"
        ) as which:
            assert resolve_executable("claude") == "/usr/bin/claude"
        which.assert_called_once_with("claude")


# ---------------------------------------------------------------------------
# run_agent
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_env():
    return {"EXISTING": "1"}


class TestRunAgentExistingGateway:
    def test_runs_agent_without_spawning_gateway(self, fake_env):
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=True),
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env) as ge,
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
            mock.patch.object(launcher.subprocess, "Popen") as popen,
        ):
            code = run_agent(["claude", "-p", "hi"], host="h", port=1)
        assert code == 0
        popen.assert_not_called()
        run.assert_called_once_with(["/bin/claude", "-p", "hi"], env=fake_env)
        ge.assert_called_once_with("claude", "h", 1)

    def test_propagates_agent_exit_code(self, fake_env):
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=True),
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 42),
            ),
        ):
            assert run_agent(["claude"]) == 42

    def test_stdio_is_not_captured(self, fake_env):
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=True),
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
        ):
            run_agent(["claude"])
        assert run.call_args.kwargs == {"env": fake_env}


class TestRunAgentAutoStart:
    def test_spawns_gateway_when_unhealthy(self, fake_env):
        proc = mock.Mock()
        proc.poll.return_value = None
        with (
            mock.patch.object(
                launcher, "check_server_healthy", side_effect=[False, True]
            ),
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc) as popen,
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            code = run_agent(["claude"], host="127.0.0.1", port=8080)
        assert code == 0
        cmd = popen.call_args.args[0]
        assert cmd[:3] == [sys.executable, "-m", "agentbridge.cli"]
        assert cmd[3:] == ["start", "--host", "127.0.0.1", "--port", "8080"]
        # The spawned gateway must be shut down afterwards.
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()

    def test_forwards_upstream_args_to_gateway(self, fake_env):
        proc = mock.Mock()
        proc.poll.return_value = None
        with (
            mock.patch.object(
                launcher, "check_server_healthy", side_effect=[False, True]
            ),
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc) as popen,
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            run_agent(["claude"], upstream_args=["--model", "glm-4.6"])
        cmd = popen.call_args.args[0]
        assert cmd[-2:] == ["--model", "glm-4.6"]

    def test_returns_1_when_gateway_never_becomes_healthy(self, fake_env):
        proc = mock.Mock()
        proc.poll.return_value = None
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=False),
            mock.patch.object(launcher, "wait_for_gateway", return_value=False),
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc) as popen,
            mock.patch.object(launcher.subprocess, "run") as run,
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
        ):
            code = run_agent(["claude"])
        assert code == 1
        popen.assert_called_once()
        run.assert_not_called()
        proc.terminate.assert_called_once()

    def test_kills_gateway_that_ignores_terminate(self, fake_env):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("gw", 5), 0]
        with (
            mock.patch.object(
                launcher, "check_server_healthy", side_effect=[False, True]
            ),
            mock.patch.object(launcher.subprocess, "Popen", return_value=proc),
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            assert run_agent(["claude"]) == 0
        proc.kill.assert_called_once()

    def test_does_not_touch_preexisting_gateway(self, fake_env):
        # Gateway already healthy: nothing should be terminated even though
        # the gateway lifecycle code path is armed.
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=True),
            mock.patch.object(launcher.subprocess, "Popen") as popen,
            mock.patch.object(
                launcher, "resolve_executable", return_value="/bin/claude"
            ),
            mock.patch.object(launcher, "get_agent_env", return_value=fake_env),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
        ):
            assert run_agent(["claude"]) == 0
        popen.assert_not_called()


class TestRunAgentFailures:
    def test_returns_1_when_auto_start_disabled_and_no_gateway(self):
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=False),
            mock.patch.object(launcher.subprocess, "Popen") as popen,
        ):
            assert run_agent(["claude"], auto_start=False) == 1
        popen.assert_not_called()

    def test_returns_127_when_agent_not_installed(self):
        with (
            mock.patch.object(launcher, "check_server_healthy", return_value=True),
            mock.patch.object(launcher, "resolve_executable", return_value=None),
            mock.patch.object(launcher.subprocess, "run") as run,
        ):
            assert run_agent(["nosuchagent"]) == 127
        run.assert_not_called()

    def test_returns_1_for_empty_agent_args(self):
        with mock.patch.object(launcher.subprocess, "Popen") as popen:
            assert run_agent([]) == 1
        popen.assert_not_called()
