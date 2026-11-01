"""FastAPI ASGI server implementation."""

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agentbridge.adapters.anthropic_to_openai import AnthropicToOpenAIAdapter
from agentbridge.adapters.thinking import ThinkingTransformer
from agentbridge.config import BridgeConfig, UpstreamConfig
from agentbridge.mcp.proxy import MCPProxy
from agentbridge.providers.factory import create_upstream_client
from agentbridge.routing.pool import FailoverRouter, UpstreamEndpoint
from agentbridge.streaming.sse import format_sse
from agentbridge.telemetry.display import log_request_metric
from agentbridge.telemetry.metrics import RequestMetric, TelemetryManager


def _extract_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    """Normalize an OpenAI usage object into (prompt, completion, reasoning)."""
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt": int(usage.get("prompt_tokens", 0) or 0),
        "completion": int(usage.get("completion_tokens", 0) or 0),
        "reasoning": int(details.get("reasoning_tokens", 0) or 0),
    }


def _build_endpoints(config: BridgeConfig) -> list:
    """Build the failover pool from the upstream configuration.

    One endpoint is created per configured API key (``key-1``, ``key-2``,
    ...), plus a ``fallback`` endpoint when ``fallback_url`` is set.
    """
    upstream: UpstreamConfig = config.upstream
    endpoints: list[UpstreamEndpoint] = []

    if upstream.api_keys:
        for index, key in enumerate(upstream.api_keys):
            endpoint_config = upstream.model_copy(
                update={"name": f"key-{index + 1}", "api_key": key}
            )
            endpoints.append(
                UpstreamEndpoint(
                    name=f"key-{index + 1}",
                    client=create_upstream_client(endpoint_config),
                )
            )
    else:
        endpoints.append(
            UpstreamEndpoint(
                name=upstream.name, client=create_upstream_client(upstream)
            )
        )

    if upstream.fallback_url:
        fallback_config = upstream.model_copy(
            update={
                "name": "fallback",
                "base_url": upstream.fallback_url,
                "api_key": upstream.fallback_key or upstream.api_key,
                "default_model": upstream.fallback_model or upstream.default_model,
            }
        )
        endpoints.append(
            UpstreamEndpoint(
                name="fallback", client=create_upstream_client(fallback_config)
            )
        )

    return endpoints


def create_app(config: BridgeConfig) -> FastAPI:
    """Create and configure the FastAPI application."""
    adapter = AnthropicToOpenAIAdapter(model_map=config.model_mapping)
    endpoints = _build_endpoints(config)
    router = FailoverRouter(endpoints, strategy=config.upstream.failover_strategy)
    mcp_proxy = (
        MCPProxy(config_path=config.mcp_config_path) if config.enable_mcp else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if mcp_proxy is not None:
            await mcp_proxy.start_all()
        yield
        if mcp_proxy is not None:
            await mcp_proxy.stop_all()
        await router.close()

    app = FastAPI(
        title="AgentBridge",
        description="Local AI coding agent gateway & protocol bridge",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ``client`` is kept as the primary endpoint's client so existing
    # integrations (and tests) that patch it keep working; the router
    # holds the same instances.
    app.state.client = endpoints[0].client
    app.state.router = router
    app.state.endpoints = endpoints
    app.state.mcp_proxy = mcp_proxy
    telemetry = TelemetryManager()
    app.state.telemetry = telemetry

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/telemetry")
    async def telemetry_overview():
        """Expose aggregated traffic / token / latency metrics."""
        return telemetry.get_summary()

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [
                {
                    "id": "claude-3-7-sonnet-20250219",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "agentbridge",
                },
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "agentbridge",
                },
                {
                    "id": "claude-3-5-haiku-20241022",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "agentbridge",
                },
            ]
        }

    @app.get("/v1/mcp/tools")
    async def list_mcp_tools():
        """Expose the active MCP tool catalogue to management clients."""
        if mcp_proxy is None:
            return {"status": "disabled", "tools": [], "count": 0}
        tools = mcp_proxy.get_anthropic_tools()
        return {"status": "ok", "tools": tools, "count": len(mcp_proxy.tools)}

    @app.get("/v1/failover/status")
    async def failover_status():
        """Report the health of every endpoint in the failover pool."""
        return {
            "status": "ok",
            "strategy": router.strategy,
            "count": len(endpoints),
            "endpoints": router.get_pool_status(),
        }

    @app.post("/v1/messages")
    async def create_message(
        request: Request,
        anthropic_version: str = Header(
            default="2023-06-01", alias="anthropic-version"
        ),
    ):
        """Handle Anthropic Messages API requests and forward to upstream."""
        try:
            payload: Dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        is_stream = payload.get("stream", False)
        model_requested = payload.get("model", "unknown")
        metric: RequestMetric = telemetry.start_request(
            req_id=f"req_{uuid.uuid4().hex[:12]}",
            model_req=model_requested,
            model_upstream="",
            client_agent=request.headers.get("user-agent") or "Unknown",
            is_stream=bool(is_stream),
        )

        if mcp_proxy is not None:
            # Auto-inject non-duplicate MCP tool declarations so the upstream
            # model becomes aware of the locally relayed MCP tools.
            payload["tools"] = mcp_proxy.inject_tools(payload.get("tools") or [])

        openai_req = adapter.convert_request(payload)
        metric.model_upstream = openai_req.get("model", "")

        if not is_stream:
            try:
                upstream_resp, _endpoint_name = await router.complete(openai_req)
                anthropic_resp = adapter.convert_response(upstream_resp)
            except Exception as e:
                telemetry.finish_request(metric, status_code=502, error_message=str(e))
                if config.telemetry_log:
                    log_request_metric(metric)
                raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

            usage = _extract_usage(upstream_resp.get("usage"))
            telemetry.record_tokens(metric, **usage)
            for block in anthropic_resp.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    telemetry.record_tool_call(metric, block.get("name", ""))
            telemetry.finish_request(metric, status_code=200)
            if config.telemetry_log:
                log_request_metric(metric)
            return JSONResponse(content=anthropic_resp)

        async def stream_generator() -> AsyncIterator[str]:
            context: Dict[str, Any] = {}
            text_chars = 0
            reasoning_chars = 0
            first_chunk_seen = False
            stream_usage: Dict[str, Any] | None = None

            def track_chunk(chunk: Dict[str, Any]) -> None:
                """Fold one upstream chunk into the running metric."""
                nonlocal text_chars, reasoning_chars, first_chunk_seen, stream_usage
                if not first_chunk_seen:
                    first_chunk_seen = True
                    telemetry.record_first_token(metric)
                if chunk.get("usage"):
                    stream_usage = chunk["usage"]
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    reasoning, content = (
                        ThinkingTransformer.extract_reasoning_and_content(chunk)
                    )
                    if reasoning:
                        reasoning_chars += len(reasoning)
                    if content:
                        text_chars += len(content)
                    for tc in delta.get("tool_calls", []) or []:
                        name = (tc.get("function") or {}).get("name")
                        if name:
                            telemetry.record_tool_call(metric, name)

            try:
                async for chunk in router.stream_complete(openai_req):
                    track_chunk(chunk)
                    events = adapter.convert_stream_chunk(chunk, context)
                    for ev in events:
                        ev_type = ev.get("type", "message_delta")
                        yield format_sse(ev_type, ev)
            except Exception as exc:
                telemetry.finish_request(
                    metric, status_code=500, error_message=str(exc)
                )
                if config.telemetry_log:
                    log_request_metric(metric)
                err_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                }
                yield format_sse("error", err_event)
            else:
                if stream_usage is not None:
                    telemetry.record_tokens(metric, **_extract_usage(stream_usage))
                elif text_chars or reasoning_chars:
                    # No usage chunk from upstream: rough estimate (~4 chars/token).
                    telemetry.record_tokens(
                        metric, completion=(text_chars + reasoning_chars + 3) // 4
                    )
                telemetry.finish_request(metric, status_code=200)
                if config.telemetry_log:
                    log_request_metric(metric)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    return app
