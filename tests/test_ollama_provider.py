"""Tests for the Ollama provider adapter and factory routing."""

import json

import httpx
import pytest

from agentbridge.config import UpstreamConfig
from agentbridge.providers.client import UpstreamClient
from agentbridge.providers.factory import (
    create_upstream_client,
    detect_provider,
    is_ollama_upstream,
)
from agentbridge.providers.ollama import OllamaProvider


def make_config(**overrides) -> UpstreamConfig:
    defaults = {
        "name": "ollama",
        "base_url": "https://api.openai.com/v1",  # non-Ollama default on purpose
        "api_key": "",
        "default_model": "qwen2.5-coder:14b",
    }
    defaults.update(overrides)
    return UpstreamConfig(**defaults)


def make_provider(config=None, **kwargs) -> OllamaProvider:
    return OllamaProvider(config or make_config(), **kwargs)


def sse_response(*chunks, done=True):
    """Build an SSE body from OpenAI-style delta chunks."""
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return httpx.Response(200, content="".join(lines).encode("utf-8"))


def delta_chunk(content=None, reasoning=None, **extra):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    delta.update(extra)
    return {
        "id": "chatcmpl-1",
        "model": "deepseek-r1:14b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


# ---------------------------------------------------------------------------
# Defaults & endpoint resolution
# ---------------------------------------------------------------------------


def test_default_base_url_and_api_key():
    provider = make_provider()
    assert str(provider.client.base_url).rstrip("/") == "http://127.0.0.1:11434/v1"
    assert provider.client.headers["Authorization"] == "Bearer ollama"


def test_explicit_base_url_preserved():
    provider = make_provider(make_config(base_url="http://192.168.1.5:11434/v1"))
    assert str(provider.client.base_url).rstrip("/") == "http://192.168.1.5:11434/v1"


def test_explicit_api_key_preserved():
    provider = make_provider(make_config(api_key="secret"))
    assert provider.client.headers["Authorization"] == "Bearer secret"


# ---------------------------------------------------------------------------
# Payload normalization
# ---------------------------------------------------------------------------


def test_prepare_payload_strips_unsupported_params():
    provider = make_provider()
    payload = provider.prepare_payload(
        {
            "model": "deepseek-r1:14b",
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": {"50256": -100},
            "logprobs": True,
            "top_logprobs": 5,
            "user": "agent-1",
            "n": 2,
            "parallel_tool_calls": True,
        }
    )
    for key in (
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "n",
        "parallel_tool_calls",
    ):
        assert key not in payload
    assert payload["model"] == "deepseek-r1:14b"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_prepare_payload_fills_default_model():
    provider = make_provider()
    payload = provider.prepare_payload(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert payload["model"] == "qwen2.5-coder:14b"


def test_prepare_payload_does_not_mutate_input():
    provider = make_provider()
    original = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True,
    }
    snapshot = json.dumps(original)
    provider.prepare_payload(original)
    assert json.dumps(original) == snapshot


# ---------------------------------------------------------------------------
# Non-streaming reasoning normalization
# ---------------------------------------------------------------------------


async def test_complete_extracts_think_block():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "deepseek-r1:14b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<think>Let me reason step by step</think>"
                                "The answer is 42."
                            ),
                        },
                    }
                ],
            },
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    response = await provider.complete(
        {"model": "deepseek-r1:14b", "messages": [{"role": "user", "content": "?"}]}
    )
    message = response["choices"][0]["message"]
    assert message["content"] == "The answer is 42."
    assert message["reasoning_content"] == "Let me reason step by step"


async def test_complete_passthrough_without_think_tags():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello!"}}
                ]
            },
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    response = await provider.complete({"messages": []})
    message = response["choices"][0]["message"]
    assert message["content"] == "Hello!"
    assert "reasoning_content" not in message


# ---------------------------------------------------------------------------
# Streaming reasoning normalization
# ---------------------------------------------------------------------------


async def collect_stream(provider, payload):
    return [chunk async for chunk in provider.stream_complete(payload)]


async def test_stream_native_reasoning_content_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            delta_chunk(reasoning="thinking hard"),
            delta_chunk(content="final answer"),
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = await collect_stream(provider, {"model": "deepseek-r1:14b"})
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0]["reasoning_content"] == "thinking hard"
    assert deltas[1]["content"] == "final answer"


async def test_stream_embedded_think_tags_split_into_reasoning():
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            delta_chunk(content="<think>Plan "),
            delta_chunk(content="the fix"),
            delta_chunk(content="</think>Here is"),
            delta_chunk(content=" the code"),
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = await collect_stream(provider, {"model": "deepseek-r1:14b"})
    deltas = [c["choices"][0]["delta"] for c in chunks]

    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert reasoning == "Plan the fix"
    assert content == "Here is the code"


async def test_stream_think_tag_split_across_chunk_boundary():
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            delta_chunk(content="<thi"),
            delta_chunk(content="nk>Pl"),
            delta_chunk(content="an step</think>F"),
            delta_chunk(content="inal answer"),
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = await collect_stream(provider, {"model": "deepseek-r1:14b"})
    deltas = [c["choices"][0]["delta"] for c in chunks]

    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert reasoning == "Plan step"
    assert content == "Final answer"


async def test_stream_unclosed_think_block_flushes_reasoning():
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(delta_chunk(content="<think>still reasoning"))

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = await collect_stream(provider, {"model": "deepseek-r1:14b"})
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "still reasoning"
    assert "".join(d.get("content", "") for d in deltas) == ""


async def test_stream_chunk_without_content_passes_through():
    role_chunk = {
        "id": "chatcmpl-1",
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(role_chunk, delta_chunk(content="ok"))

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = await collect_stream(provider, {"model": "m"})
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}


async def test_stream_request_sends_cleaned_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return sse_response(delta_chunk(content="ok"))

    provider = make_provider(transport=httpx.MockTransport(handler))
    await collect_stream(
        provider,
        {"model": "deepseek-r1:14b", "messages": [], "logprobs": True},
    )
    assert captured["payload"]["stream"] is True
    assert "logprobs" not in captured["payload"]
    assert captured["auth"] == "Bearer ollama"


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "ollama"},
        {"name": "Ollama"},
        {"name": "OLLAMA"},
        {"base_url": "http://127.0.0.1:11434/v1"},
        {"base_url": "http://192.168.1.5:11434/v1", "name": "local"},
        {"base_url": "https://ollama.example.com/v1"},
    ],
)
def test_ollama_inputs_route_to_ollama_provider(overrides):
    client = create_upstream_client(make_config(**overrides))
    assert isinstance(client, OllamaProvider)


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "openai"},
        {"name": "deepseek", "base_url": "https://api.deepseek.com/v1"},
        {
            "name": "glm",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "api_key": "zk-test",
        },
    ],
)
def test_non_ollama_inputs_do_not_route_to_ollama(overrides):
    client = create_upstream_client(make_config(**overrides))
    assert not isinstance(client, OllamaProvider)


def test_is_ollama_upstream_detection():
    assert is_ollama_upstream(make_config(name="ollama"))
    assert is_ollama_upstream(make_config(base_url="http://127.0.0.1:11434/v1"))
    assert is_ollama_upstream(make_config(base_url="https://ollama.lan/v1"))
    assert not is_ollama_upstream(make_config(name="openai"))
    assert not is_ollama_upstream(
        make_config(name="default", base_url="https://api.openai.com/v1")
    )


def test_detect_provider_labels_ollama():
    assert detect_provider(make_config(name="ollama")) == "ollama"
    assert detect_provider(make_config(base_url="http://127.0.0.1:11434/v1")) == (
        "ollama"
    )


@pytest.mark.asyncio
async def test_ollama_client_closes_cleanly():
    client = create_upstream_client(make_config())
    await client.close()


def test_standard_client_still_fallback():
    config = make_config(name="openai", base_url="https://api.openai.com/v1")
    assert type(create_upstream_client(config)) is UpstreamClient
