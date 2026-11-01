"""Command-line interface for AgentBridge."""

import argparse

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


def main():
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "start":
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

        print_startup_banner(config)
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
