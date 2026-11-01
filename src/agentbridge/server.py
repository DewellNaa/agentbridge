"""FastAPI ASGI server implementation."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agentbridge.adapters.anthropic_to_openai import AnthropicToOpenAIAdapter
from agentbridge.config import BridgeConfig
from agentbridge.mcp.proxy import MCPProxy
from agentbridge.providers.factory import create_upstream_client
from agentbridge.streaming.sse import format_sse


def create_app(config: BridgeConfig) -> FastAPI:
    """Create and configure the FastAPI application."""
    adapter = AnthropicToOpenAIAdapter(model_map=config.model_mapping)
    client = create_upstream_client(config.upstream)
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
        await client.close()

    app = FastAPI(
        title="AgentBridge",
        description="Local AI coding agent gateway & protocol bridge",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.client = client
    app.state.mcp_proxy = mcp_proxy

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "0.1.0"}

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

        if mcp_proxy is not None:
            # Auto-inject non-duplicate MCP tool declarations so the upstream
            # model becomes aware of the locally relayed MCP tools.
            payload["tools"] = mcp_proxy.inject_tools(payload.get("tools") or [])

        openai_req = adapter.convert_request(payload)

        if not is_stream:
            try:
                upstream_resp = await client.complete(openai_req)
                anthropic_resp = adapter.convert_response(upstream_resp)
                return JSONResponse(content=anthropic_resp)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Upstream error: {str(e)}")

        async def stream_generator() -> AsyncIterator[str]:
            context: Dict[str, Any] = {}
            try:
                async for chunk in client.stream_complete(openai_req):
                    events = adapter.convert_stream_chunk(chunk, context)
                    for ev in events:
                        ev_type = ev.get("type", "message_delta")
                        yield format_sse(ev_type, ev)
            except Exception as exc:
                err_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                }
                yield format_sse("error", err_event)

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
