"""Command-line interface for AgentBridge."""

import argparse
import os

import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentbridge import __version__
from agentbridge.config import BridgeConfig
from agentbridge.providers.factory import detect_provider
from agentbridge.server import create_app

console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agentbridge",
        description="Ultra-lightweight local AI coding agent gateway & protocol bridge",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"agentbridge {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # start command
    start_parser = subparsers.add_parser("start", help="Start the bridge server")
    start_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    start_parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Port to bind (default: 8080)"
    )
    start_parser.add_argument(
        "--upstream-url",
        default=None,
        help="Upstream OpenAI-compatible API base URL (e.g. https://api.openai.com/v1)",
    )
    start_parser.add_argument(
        "--upstream-key",
        default=None,
        help="Upstream API key (defaults to OPENAI_API_KEY env var)",
    )
    start_parser.add_argument(
        "--model",
        default=None,
        help="Target upstream model name (e.g. deepseek-chat, glm-4-plus)",
    )
    start_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    start_parser.add_argument(
        "--enable-mcp",
        action="store_true",
        help="Automatically discover and relay local MCP servers",
    )
    start_parser.add_argument(
        "--mcp-config",
        default=None,
        help="Path to custom mcp.json config file",
    )
    start_parser.add_argument(
        "--provider",
        choices=["auto", "openai", "glm"],
        default="auto",
        help="Upstream provider type (default: auto)",
    )
    start_parser.add_argument(
        "--no-telemetry-log",
        action="store_true",
        help="Disable per-request telemetry logs in the console",
    )
    start_parser.add_argument(
        "--ollama",
        nargs="?",
        const="auto",
        default=None,
        help="Connect to local Ollama with auto-discovery "
        "(optionally specify model name, e.g. --ollama qwen2.5-coder:14b)",
    )

    # ollama command
    subparsers.add_parser(
        "ollama",
        help="List local Ollama models with coding-fit scores and a recommendation",
    )

    # metrics command
    metrics_parser = subparsers.add_parser(
        "metrics",
        aliases=["status"],
        help="Show telemetry dashboard of a running gateway",
    )
    metrics_parser.add_argument(
        "--host", default="127.0.0.1", help="Gateway host (default: 127.0.0.1)"
    )
    metrics_parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Gateway port (default: 8080)"
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Launch a coding agent wired to the gateway (auto-starts one if needed)",
    )
    run_parser.add_argument(
        "agent",
        help="Target agent command (e.g. claude, aider) or any custom command",
    )
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to the agent (use -- before agent flags)",
    )
    run_parser.add_argument(
        "--host", default="127.0.0.1", help="Gateway host (default: 127.0.0.1)"
    )
    run_parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Gateway port (default: 8080)"
    )
    run_parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Never spawn a background gateway; only connect to an existing one",
    )
    run_parser.add_argument(
        "--upstream-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="Extra argument passed to the auto-started gateway "
        "(repeatable, e.g. --upstream-arg=--model=glm-4.6)",
    )

    # config command
    config_parser = subparsers.add_parser(
        "config",
        help="Print a one-click IDE configuration snippet (cursor, cline, continue)",
    )
    config_parser.add_argument("target", help="IDE target: cursor, cline, or continue")
    config_parser.add_argument(
        "--host", default="127.0.0.1", help="Gateway host (default: 127.0.0.1)"
    )
    config_parser.add_argument(
        "-p", "--port", type=int, default=8080, help="Gateway port (default: 8080)"
    )

    return parser


def print_startup_banner(config: BridgeConfig):
    """Print a visually striking startup banner and instructions for Claude Code."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row(
        "[bold cyan]Gateway URL:[/bold cyan]", f"http://{config.host}:{config.port}"
    )
    table.add_row(
        "[bold cyan]Anthropic Endpoint:[/bold cyan]",
        f"http://{config.host}:{config.port}/v1/messages",
    )
    table.add_row("[bold cyan]Upstream Target:[/bold cyan]", config.upstream.base_url)
    table.add_row(
        "[bold cyan]Default Model:[/bold cyan]", config.upstream.default_model
    )
    provider_type = detect_provider(config.upstream)
    table.add_row(
        "[bold cyan]Provider:[/bold cyan]",
        f"{provider_type} ({config.upstream.name})",
    )
    if config.ollama_mode:
        table.add_row(
            "[bold cyan]Ollama Auto-Discovery:[/bold cyan]",
            f"Active (Target: {config.upstream.default_model})",
        )
        table.add_row(
            "[bold cyan]Local Models:[/bold cyan]",
            str(config.ollama_model_count),
        )
    if config.enable_mcp:
        mcp_status = "[green]enabled[/green]"
        if config.mcp_config_path:
            mcp_status += f" [dim]({config.mcp_config_path})[/dim]"
    else:
        mcp_status = "[dim]disabled[/dim]"
    table.add_row("[bold cyan]MCP Relay:[/bold cyan]", mcp_status)

    panel = Panel(
        table,
        title=f"[bold green]🚀 AgentBridge v{__version__} Ready[/bold green]",
        subtitle="[dim]Zero-overhead AI Coding Agent Protocol Bridge[/dim]",
        border_style="green",
    )
    console.print(panel)

    instructions = f"""
[bold yellow]To connect Claude Code:[/bold yellow]
  [green]export ANTHROPIC_BASE_URL="http://{config.host}:{config.port}"[/green]
  [green]export ANTHROPIC_API_KEY="agentbridge"[/green]
  [green]claude[/green]

[bold yellow]To connect Aider:[/bold yellow]
  [green]aider --anthropic-api-base "http://{config.host}:{config.port}/v1" --model claude-3-5-sonnet-20241022[/green]
"""
    console.print(
        Panel(instructions.strip(), title="Client Integration", border_style="cyan")
    )


def print_metrics_dashboard(host: str, port: int) -> int:
    """Fetch /v1/telemetry from a running gateway and render the dashboard."""
    import httpx

    from agentbridge.telemetry.display import render_summary_panel

    url = f"http://{host}:{port}/v1/telemetry"
    try:
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        summary = resp.json()
    except Exception as exc:
        console.print(f"[bold red]Failed to query gateway at {url}:[/bold red] {exc}")
        return 1
    console.print(render_summary_panel(summary))
    return 0


def run_command(args: argparse.Namespace) -> int:
    """Handle `agentbridge run`: delegate to the launcher."""
    from agentbridge.launcher import run_agent

    agent_args = [args.agent, *args.args]
    while agent_args and agent_args[0] == "--":
        agent_args.pop(0)
    return run_agent(
        agent_args,
        host=args.host,
        port=args.port,
        auto_start=not args.no_auto_start,
        upstream_args=args.upstream_arg,
    )


def config_command(args: argparse.Namespace) -> int:
    """Handle `agentbridge config`: print the IDE configuration snippet."""
    from agentbridge.templates import generate_ide_config

    console.print(generate_ide_config(args.target, host=args.host, port=args.port))
    return 0


def apply_ollama_config(config: BridgeConfig, ollama_arg: str) -> int:
    """Wire ``config`` to a local Ollama instance via auto-discovery.

    ``ollama_arg`` is ``"auto"`` (plain ``--ollama``) for best-model
    selection, or an explicit model name. Returns a process exit code
    (0 on success, 1 when Ollama is unreachable or has no models).
    """
    from agentbridge.discovery.ollama import (
        check_ollama_running,
        discover_ollama_models,
        parse_ollama_host,
        select_best_ollama_model,
    )

    host, port = parse_ollama_host(os.getenv("OLLAMA_HOST", ""))
    endpoint = f"http://{host}:{port}"

    if not check_ollama_running(host, port):
        console.print(
            f"[red]Error:[/red] Ollama is not running at {endpoint}. "
            "Please start Ollama or download it from https://ollama.com"
        )
        return 1

    if ollama_arg == "auto":
        models = discover_ollama_models(host, port)
        config.ollama_model_count = len(models)
        best = select_best_ollama_model(models)
        if best is None:
            console.print(
                "[yellow]No Ollama models installed.[/yellow] "
                "Pull a coding model first:\n"
                "  [green]ollama pull qwen2.5-coder[/green]"
            )
            return 1
        model = best.name
        console.print(
            f"[bold cyan]Ollama Auto-Discovery:[/bold cyan] selected "
            f"[green]{model}[/green] (score {best.score}) from "
            f"{len(models)} local model(s)"
        )
    else:
        model = ollama_arg
        try:
            config.ollama_model_count = len(discover_ollama_models(host, port))
        except Exception:
            config.ollama_model_count = 0

    config.ollama_mode = True
    config.upstream.name = "ollama"
    config.upstream.base_url = f"{endpoint}/v1"
    config.upstream.default_model = model
    config.upstream.api_key = "ollama"
    config.upstream.ollama_mode = True
    return 0


def ollama_command(args: argparse.Namespace) -> int:
    """Handle `agentbridge ollama`: list local models with coding scores."""
    from agentbridge.discovery.ollama import (
        check_ollama_running,
        discover_ollama_models,
        parse_ollama_host,
        select_best_ollama_model,
    )

    host, port = parse_ollama_host(os.getenv("OLLAMA_HOST", ""))
    if not check_ollama_running(host, port):
        console.print(
            f"[red]Error:[/red] Ollama is not running at http://{host}:{port}. "
            "Please start Ollama or download it from https://ollama.com"
        )
        return 1
    try:
        models = discover_ollama_models(host, port)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Failed to query Ollama: {exc}")
        return 1

    if not models:
        console.print(
            "[yellow]No Ollama models installed.[/yellow] Pull a coding model:\n"
            "  [green]ollama pull qwen2.5-coder[/green]"
        )
        return 0

    best = select_best_ollama_model(models)
    table = Table(title=f"Ollama Local Models ({len(models)})")
    table.add_column("Model", style="cyan")
    table.add_column("Params")
    table.add_column("Quantization")
    table.add_column("Size", justify="right")
    table.add_column("Coding Score", justify="right")
    for model in models:
        is_best = best is not None and model.name == best.name
        if is_best:
            name = f"[bold green]★ {model.name}[/bold green]"
            score = f"[bold green]{model.score}[/bold green]"
        else:
            name = model.name
            score = str(model.score)
        table.add_row(
            name,
            model.parameter_size or "-",
            model.quantization_level or "-",
            _human_size(model.size_bytes),
            score,
        )
    console.print(table)
    if best is not None:
        console.print(
            f"Recommended for coding: [bold green]{best.name}[/bold green] "
            f"(score {best.score})"
        )
    return 0


def _human_size(size_bytes: int) -> str:
    """Format a byte count as a compact human-readable string."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command in (None, "start"):
        config = BridgeConfig.from_env()

        if hasattr(args, "host") and args.host:
            config.host = args.host
        if hasattr(args, "port") and args.port:
            config.port = args.port
        if hasattr(args, "upstream_url") and args.upstream_url:
            config.upstream.base_url = args.upstream_url
        if hasattr(args, "upstream_key") and args.upstream_key:
            config.upstream.api_key = args.upstream_key
        if hasattr(args, "model") and args.model:
            config.upstream.default_model = args.model
        if hasattr(args, "debug") and args.debug:
            config.debug = args.debug
        if hasattr(args, "enable_mcp") and args.enable_mcp:
            config.enable_mcp = True
        if hasattr(args, "mcp_config") and args.mcp_config:
            config.mcp_config_path = args.mcp_config
        if getattr(args, "provider", "auto") == "glm":
            config.upstream.name = "glm"
        if getattr(args, "no_telemetry_log", False):
            config.telemetry_log = False

        ollama_arg = getattr(args, "ollama", None)
        if ollama_arg is not None:
            exit_code = apply_ollama_config(config, ollama_arg)
            if exit_code != 0:
                raise SystemExit(exit_code)

        print_startup_banner(config)
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    elif args.command in ("metrics", "status"):
        raise SystemExit(print_metrics_dashboard(args.host, args.port))
    elif args.command == "run":
        raise SystemExit(run_command(args))
    elif args.command == "config":
        raise SystemExit(config_command(args))
    elif args.command == "ollama":
        raise SystemExit(ollama_command(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
