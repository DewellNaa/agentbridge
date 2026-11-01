"""Agent one-shot launcher: gateway probing, auto-start, and env injection.

Provides `agentbridge run <agent>`: transparently ensure a local AgentBridge
gateway is running, then spawn the target coding agent (claude, aider, ...)
with environment variables pointing it at the gateway.
"""

import os
import shutil
import subprocess
import sys
import time

import httpx
from rich.console import Console

console = Console()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_API_KEY = "agentbridge"

INSTALL_HINT = (
    "未找到该命令，请先安装该智能体"
    "（如 npm install -g @anthropic-ai/claude-code 或 pip install aider-chat）"
)


def check_server_healthy(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.8
) -> bool:
    """Return True if a gateway is already serving /health on host:port."""
    try:
        resp = httpx.get(f"http://{host}:{port}/health", timeout=timeout)
    except Exception:
        return False
    return resp.status_code == 200


def wait_for_gateway(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = 8.0,
    interval: float = 0.25,
) -> bool:
    """Poll /health until the gateway is ready; return False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_server_healthy(host, port, timeout=min(interval, 1.0)):
            return True
        time.sleep(interval)
    return False


def get_agent_env(
    agent: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    existing_env: dict | None = None,
) -> dict[str, str]:
    """Build the environment dict for the agent subprocess.

    Injects ANTHROPIC_BASE_URL (with a /v1 suffix for aider) and
    ANTHROPIC_API_KEY, on top of the inherited system environment.
    """
    if agent == "aider":
        base_url = f"http://{host}:{port}/v1"
    else:
        base_url = f"http://{host}:{port}"
    env = dict(os.environ if existing_env is None else existing_env)
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_API_KEY"] = DEFAULT_API_KEY
    return env


def resolve_executable(cmd: str) -> str | None:
    """Resolve a command to a full executable path (handles .cmd/.exe on Windows)."""
    return shutil.which(cmd)


def _stop_gateway(proc: subprocess.Popen) -> None:
    """Gracefully terminate a gateway spawned by this launcher."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_agent(
    agent_args: list[str],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auto_start: bool = True,
    upstream_args: list[str] | None = None,
) -> int:
    """Run the target agent wired to the gateway; return its exit code.

    If no healthy gateway is found and `auto_start` is True, a background
    gateway is spawned first and shut down again once the agent exits.
    """
    if not agent_args:
        console.print("[bold red]Error:[/bold red] no agent command provided.")
        return 1

    spawned_gateway: subprocess.Popen | None = None
    try:
        if not check_server_healthy(host, port):
            if not auto_start:
                console.print(
                    f"[bold red]Error:[/bold red] no gateway is running at "
                    f"http://{host}:{port} and auto-start is disabled "
                    "(drop --no-auto-start or start one with `agentbridge start`)."
                )
                return 1
            console.print(
                f"[yellow]No gateway detected at http://{host}:{port}, "
                "starting one in the background...[/yellow]"
            )
            cmd = [
                sys.executable,
                "-m",
                "agentbridge.cli",
                "start",
                "--host",
                host,
                "--port",
                str(port),
            ] + list(upstream_args or [])
            spawned_gateway = subprocess.Popen(cmd)
            if not wait_for_gateway(host, port):
                console.print(
                    "[bold red]Error:[/bold red] gateway did not become healthy "
                    f"at http://{host}:{port} in time. Check its output above."
                )
                return 1
            console.print("[green]Gateway is ready.[/green]")

        agent = agent_args[0]
        resolved = resolve_executable(agent)
        if resolved is None:
            console.print(f"[bold red]{agent}: {INSTALL_HINT}[/bold red]")
            return 127

        resolved_args = [resolved, *agent_args[1:]]
        env = get_agent_env(agent, host, port)
        completed = subprocess.run(resolved_args, env=env)
        return completed.returncode
    finally:
        if spawned_gateway is not None:
            _stop_gateway(spawned_gateway)
