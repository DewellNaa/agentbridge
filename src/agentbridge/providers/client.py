"""Upstream provider HTTP client."""

import json
from typing import Any, AsyncIterator, Dict

import httpx

from agentbridge.config import UpstreamConfig
from agentbridge.streaming.buffer import StreamBuffer


class UpstreamClient:
    """Async HTTP client for upstream OpenAI-compatible providers."""

    def __init__(self, config: UpstreamConfig):
        self.config = config
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}" if config.api_key else "",
            **config.extra_headers,
        }
        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout,
        )

    async def close(self):
        """Close client session."""
        await self.client.aclose()

    async def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a non-streaming completion request."""
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def stream_complete(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Send a streaming completion request and yield parsed chunks."""
        payload["stream"] = True

        buffer = StreamBuffer()
        async with self.client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()

            async for raw_bytes in response.aiter_text():
                lines = buffer.feed(raw_bytes)
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
