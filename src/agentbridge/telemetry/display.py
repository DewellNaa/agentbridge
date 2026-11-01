"""Rich-based terminal visualization for request telemetry."""

import time
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from agentbridge.telemetry.metrics import RequestMetric, TelemetryManager


def log_request_metric(
    metric: RequestMetric, console: Optional[Console] = None
) -> None:
    """Print one compact Rich-formatted log line for a finished request."""
    console = console or Console()
    ts = time.strftime("%H:%M:%S", time.localtime(metric.timestamp))

    if metric.status_code < 400:
        status = f"[green]{metric.status_code} OK[/green]"
    else:
        status = f"[bold red]{metric.status_code} ERR[/bold red]"

    elapsed_s = metric.duration_ms / 1000.0
    tok_per_s = (
        metric.completion_tokens / elapsed_s
        if elapsed_s > 0
        else metric.completion_tokens
    )

    parts = [
        f"[bold cyan]{ts}[/bold cyan]",
        status,
        f"[yellow]{metric.model_requested or '?'}[/yellow] -> "
        f"[bold magenta]{metric.model_upstream or '?'}[/bold magenta]",
        f"in: {metric.prompt_tokens:,}, out: {metric.completion_tokens:,} "
        f"([blue]think: {metric.reasoning_tokens}[/blue])",
        f"[dim]TTFT: {metric.ttft_ms:.0f}ms, Tot: {elapsed_s:.1f}s "
        f"({tok_per_s:.0f} tok/s)[/dim]",
        # Escape the bracketed tool list so Rich markup keeps it literal.
        f"tools: \\[{', '.join(metric.tools_called)}]" if metric.tools_called else None,
    ]
    if metric.saved_tokens > 0:
        parts.append(f"[dim](saved {metric.saved_tokens} ctx tokens)[/dim]")
    line = " | ".join(p for p in parts if p is not None)
    if metric.error_message:
        line += f" | [bold red]{metric.error_message}[/bold red]"
    console.print(line)


def render_summary_panel(summary: dict) -> Panel:
    """Build a Rich dashboard panel from a telemetry summary dict."""
    tokens = Table(show_header=False, box=None, padding=(0, 2))
    tokens.add_row(
        "[bold cyan]Requests[/bold cyan]",
        (
            f"total: {summary.get('total_requests', 0)} | "
            f"[green]ok: {summary.get('successful_requests', 0)}[/green] | "
            f"[red]fail: {summary.get('failed_requests', 0)}[/red] | "
            f"active: {summary.get('active_requests', 0)}"
        ),
    )
    tokens.add_row(
        "[bold cyan]Tokens[/bold cyan]",
        (
            f"in: {summary.get('total_prompt_tokens', 0):,} | "
            f"out: {summary.get('total_completion_tokens', 0):,} | "
            f"reasoning: [blue]{summary.get('total_reasoning_tokens', 0):,}[/blue] | "
            f"total: {summary.get('total_tokens', 0):,}"
        ),
    )
    tokens.add_row(
        "[bold cyan]Latency[/bold cyan]",
        (
            f"avg duration: {summary.get('avg_duration_ms', 0.0):.0f}ms | "
            f"avg TTFT: {summary.get('avg_ttft_ms', 0.0):.0f}ms"
        ),
    )

    tool_usage = summary.get("tool_usage") or {}
    if tool_usage:
        tools_str = ", ".join(
            f"{name} x{count}" for name, count in sorted(tool_usage.items())
        )
        tokens.add_row("[bold cyan]Tools[/bold cyan]", tools_str)

    elements = [tokens]

    recent = summary.get("recent_requests") or []
    if recent:
        recent_table = Table(
            title="Recent Requests",
            show_lines=False,
            expand=True,
            header_style="bold",
        )
        recent_table.add_column("Time", style="cyan", no_wrap=True)
        recent_table.add_column("Status", justify="right")
        recent_table.add_column("Model", overflow="fold")
        recent_table.add_column("In", justify="right")
        recent_table.add_column("Out", justify="right")
        recent_table.add_column("Think", justify="right", style="blue")
        recent_table.add_column("Duration", justify="right")
        recent_table.add_column("Stream", justify="center")
        for r in recent:
            status_style = "green" if r.get("status_code", 500) < 400 else "red"
            recent_table.add_row(
                str(r.get("time", "")),
                f"[{status_style}]{r.get('status_code', '?')}[/{status_style}]",
                f"{r.get('model_requested', '?')} -> {r.get('model_upstream', '?')}",
                f"{r.get('prompt_tokens', 0):,}",
                f"{r.get('completion_tokens', 0):,}",
                f"{r.get('reasoning_tokens', 0):,}",
                f"{r.get('duration_ms', 0.0):.0f}ms",
                "yes" if r.get("is_stream") else "no",
            )
        elements.append(recent_table)

    return Panel(
        Group(*elements),
        title="[bold green]📊 AgentBridge Telemetry[/bold green]",
        subtitle=f"[dim]uptime: {summary.get('uptime_seconds', 0.0):.0f}s[/dim]",
        border_style="green",
    )


def render_dashboard_panel(manager: TelemetryManager) -> Panel:
    """Build a Rich dashboard panel for the current aggregated metrics."""
    return render_summary_panel(manager.get_summary())
