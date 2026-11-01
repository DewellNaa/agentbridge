"""Ollama local provider adapter (OpenAI-compatible endpoint).

Talks to the OpenAI-compatible API Ollama exposes at
``http://127.0.0.1:11434/v1``, applying Ollama-specific customizations on
behalf of coding agents:

- Defaults to the local Ollama endpoint and the placeholder ``"ollama"``
  API key so no credential setup is needed.
- Strips OpenAI-only request parameters that make Ollama fail with 400.
- Normalizes DeepSeek-R1 style reasoning output: streaming ``<think>``
  markers embedded in ``content`` (and native ``reasoning_content``
  deltas) are surfaced as ``reasoning_content`` so downstream thinking
  adapters see a consistent shape.
"""

import copy
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from agentbridge.config import UpstreamConfig
from agentbridge.providers.client import UpstreamClient

_UPSTREAM_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OllamaProvider(UpstreamClient):
    """Async HTTP client for a local Ollama instance."""

    DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
    DEFAULT_API_KEY = "ollama"

    CHAT_COMPLETIONS_PATH = "/chat/completions"

    # OpenAI-specific parameters Ollama rejects with 400; dropped before
    # the payload is forwarded.
    UNSUPPORTED_PARAMS = frozenset(
        {
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "user",
            "n",
            "best_of",
            "echo",
            "functions",
            "function_call",
            "parallel_tool_calls",
        }
    )

    THINK_OPEN_TAG = "<think>"
    THINK_CLOSE_TAG = "</think>"

    def __init__(
        self,
        config: UpstreamConfig,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """Build an Ollama provider.

        Falls back to the local Ollama endpoint and the placeholder
        ``"ollama"`` API key when the config still carries the OpenAI
        defaults, so ``--ollama`` needs no further configuration.
        """
        self.config = config
        base_url = self._resolve_base_url(config.base_url)
        api_key = config.api_key or self.DEFAULT_API_KEY

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **config.extra_headers,
        }
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout,
            transport=transport,
        )

    # ------------------------------------------------------------------
    # Endpoint resolution
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_base_url(cls, base_url: str) -> str:
        """Fall back to the local Ollama endpoint for non-Ollama URLs."""
        if not base_url or base_url.rstrip("/") == _UPSTREAM_OPENAI_DEFAULT_BASE_URL:
            return cls.DEFAULT_BASE_URL
        return base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Request normalization
    # ------------------------------------------------------------------

    def prepare_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Turn an incoming chat payload into an Ollama-compatible one.

        - Fills in the configured default model when none is requested.
        - Drops OpenAI-only parameters Ollama answers with 400.
        """
        prepared = copy.deepcopy(payload)
        if not prepared.get("model") and self.config.default_model:
            prepared["model"] = self.config.default_model
        for key in self.UNSUPPORTED_PARAMS:
            prepared.pop(key, None)
        return prepared

    # ------------------------------------------------------------------
    # Reasoning normalization (<think> tags / reasoning_content)
    # ------------------------------------------------------------------

    def normalize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Split ``<think>...</think>`` in a non-streaming message into
        ``reasoning_content`` + plain ``content``."""
        choices = response.get("choices") or []
        if not choices:
            return response
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if self.THINK_OPEN_TAG not in content:
            return response
        before, _, rest = content.partition(self.THINK_OPEN_TAG)
        thinking, _, after = rest.partition(self.THINK_CLOSE_TAG)
        message["content"] = (before + after).strip()
        if thinking.strip() and "reasoning_content" not in message:
            message["reasoning_content"] = thinking.strip()
        return response

    # ------------------------------------------------------------------
    # HTTP surface
    # ------------------------------------------------------------------

    async def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a non-streaming chat completion to Ollama."""
        response = await self.client.post(
            self.CHAT_COMPLETIONS_PATH, json=self.prepare_payload(payload)
        )
        response.raise_for_status()
        return self.normalize_response(response.json())

    async def stream_complete(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a chat completion, normalizing reasoning into deltas.

        Ollama emits DeepSeek-R1 reasoning either as native
        ``delta.reasoning_content`` (newer builds) or as ``<think>``
        markers inside ``delta.content``. Both are folded into
        ``reasoning_content`` so downstream thinking adapters see one
        consistent shape.
        """
        prepared = self.prepare_payload(payload)
        splitter = _ThinkStreamSplitter(self.THINK_OPEN_TAG, self.THINK_CLOSE_TAG)
        template_chunk: Optional[Dict[str, Any]] = None

        async for chunk in super().stream_complete(prepared):
            normalized = splitter.normalize_chunk(chunk)
            if normalized is not None:
                template_chunk = normalized
                yield normalized

        # Flush any text held back while waiting for a possibly split tag.
        if template_chunk is not None:
            reasoning, content = splitter.flush()
            if reasoning or content:
                final = copy.deepcopy(template_chunk)
                delta = final["choices"][0]["delta"]
                if content:
                    delta["content"] = delta.get("content", "") + content
                if reasoning:
                    delta["reasoning_content"] = (
                        delta.get("reasoning_content", "") + reasoning
                    )
                yield final


class _ThinkStreamSplitter:
    """Incrementally split streamed ``content`` on ``<think>`` markers.

    Tags may arrive split across chunks; a small carry buffer holds back
    a trailing partial marker until the next chunk disambiguates it.
    """

    def __init__(self, open_tag: str, close_tag: str):
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.in_thinking = False
        self._carry = ""

    def normalize_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rewrite one streamed chunk, or return it unchanged."""
        choices = chunk.get("choices") or []
        if not choices:
            return chunk
        delta = choices[0].get("delta")
        if not isinstance(delta, dict) or "content" not in delta:
            return chunk

        reasoning, content = self.feed(delta.get("content") or "")
        native_reasoning = delta.get("reasoning_content") or ""
        reasoning = native_reasoning + reasoning

        if reasoning:
            delta["reasoning_content"] = reasoning
        elif "reasoning_content" in delta:
            del delta["reasoning_content"]
        delta["content"] = content
        return chunk

    def feed(self, text: str) -> Tuple[str, str]:
        """Consume raw content text; return ``(reasoning, content)``."""
        self._carry += text
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        while self._carry:
            marker = self.close_tag if self.in_thinking else self.open_tag
            idx = self._carry.find(marker)
            if idx >= 0:
                before = self._carry[:idx]
                self._carry = self._carry[idx + len(marker) :]
                self._emit(before, reasoning_parts, content_parts)
                self.in_thinking = not self.in_thinking
                continue
            # No complete marker: emit everything except a potential
            # partial marker at the tail.
            holdback = self._partial_marker_len(marker)
            emit_len = len(self._carry) - holdback
            if emit_len > 0:
                self._emit(self._carry[:emit_len], reasoning_parts, content_parts)
                self._carry = self._carry[emit_len:]
            break
        return "".join(reasoning_parts), "".join(content_parts)

    def flush(self) -> Tuple[str, str]:
        """Emit any held-back tail as-is (stream is over)."""
        text, self._carry = self._carry, ""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        if text:
            self._emit(text, reasoning_parts, content_parts)
        return "".join(reasoning_parts), "".join(content_parts)

    def _emit(self, text: str, reasoning_parts: list[str], content_parts: list[str]):
        if not text:
            return
        if self.in_thinking:
            reasoning_parts.append(text)
        else:
            content_parts.append(text)

    def _partial_marker_len(self, marker: str) -> int:
        """Length of the longest suffix of the carry that is a proper
        prefix of ``marker``."""
        limit = min(len(marker) - 1, len(self._carry))
        for length in range(limit, 0, -1):
            if self._carry.endswith(marker[:length]):
                return length
        return 0
