"""Tests for the BigModel / Z.ai GLM-5.3 provider adapter."""

import json

import httpx
import pytest

from agentbridge.config import UpstreamConfig
from agentbridge.providers.glm import GLMProvider


def make_config(**overrides) -> UpstreamConfig:
    defaults = {
        "name": "glm",
        "base_url": "https://api.openai.com/v1",  # non-GLM default on purpose
        "api_key": "",
        "default_model": "gpt-4o",
    }
    defaults.update(overrides)
    return UpstreamConfig(**defaults)


def make_provider(config=None, **kwargs) -> GLMProvider:
    return GLMProvider(config or make_config(api_key="zk-test-key"), **kwargs)


# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------


def test_auth_header_from_config_api_key():
    provider = make_provider()
    assert provider.client.headers["Authorization"] == "Bearer zk-test-key"


def test_auth_header_derived_from_client_token():
    provider = make_provider(make_config(api_key=""), client_token="zk-client-token")
    assert provider.api_key == "zk-client-token"
    assert provider.client.headers["Authorization"] == "Bearer zk-client-token"


def test_client_token_bearer_prefix_stripped():
    provider = make_provider(
        make_config(api_key=""), client_token="Bearer zk-client-token"
    )
    assert provider.client.headers["Authorization"] == "Bearer zk-client-token"


def test_config_api_key_wins_over_client_token():
    provider = make_provider(client_token="zk-client-token")
    assert provider.api_key == "zk-test-key"


def test_missing_credentials_raises():
    with pytest.raises(ValueError, match="API key"):
        GLMProvider(make_config(api_key=""))


# ----------------------------------------------------------------------
# Endpoint resolution
# ----------------------------------------------------------------------


def test_default_base_url_is_bigmodel_cn():
    provider = make_provider()
    assert str(provider.client.base_url).rstrip("/") == GLMProvider.BIGMODEL_BASE_URL


def test_zai_region_base_url():
    provider = make_provider(region="zai")
    assert str(provider.client.base_url).rstrip("/") == GLMProvider.ZAI_BASE_URL


def test_unknown_region_raises():
    with pytest.raises(ValueError, match="region"):
        make_provider(region="mars")


def test_explicit_base_url_preserved():
    provider = make_provider(
        make_config(
            api_key="zk-test-key", base_url="https://glm.example.com/api/paas/v4"
        )
    )
    assert (
        str(provider.client.base_url).rstrip("/")
        == "https://glm.example.com/api/paas/v4"
    )


# ----------------------------------------------------------------------
# Model mapping
# ----------------------------------------------------------------------


def test_sonnet_maps_to_glm_53():
    provider = make_provider()
    assert provider.resolve_model("claude-3-7-sonnet-20250219") == "glm-5.3"


def test_haiku_maps_to_glm_53_flash():
    provider = make_provider()
    assert provider.resolve_model("claude-3-5-haiku-20241022") == "glm-5.3-flash"


def test_glm_model_passthrough():
    provider = make_provider()
    assert provider.resolve_model("glm-5.3-flash") == "glm-5.3-flash"


def test_unknown_model_falls_back_to_default():
    provider = make_provider()
    assert provider.resolve_model("gpt-4o") == "glm-5.3"


def test_unknown_model_uses_glm_default_model_from_config():
    provider = make_provider(make_config(api_key="k", default_model="glm-5.3-flash"))
    assert provider.resolve_model("some-other-model") == "glm-5.3-flash"


def test_missing_model_uses_default():
    provider = make_provider()
    assert provider.resolve_model(None) == "glm-5.3"


# ----------------------------------------------------------------------
# thinking / do_sample parameter handling
# ----------------------------------------------------------------------


def test_thinking_dict_passthrough():
    provider = make_provider()
    payload = provider.prepare_payload(
        {"model": "glm-5.3", "messages": [], "thinking": {"type": "enabled"}}
    )
    assert payload["thinking"] == {"type": "enabled"}


def test_thinking_bool_normalized():
    provider = make_provider()
    assert provider.prepare_payload({"thinking": True})["thinking"] == {
        "type": "enabled"
    }
    assert provider.prepare_payload({"thinking": False})["thinking"] == {
        "type": "disabled"
    }


def test_thinking_string_normalized():
    provider = make_provider()
    assert provider.prepare_payload({"thinking": "enabled"})["thinking"] == {
        "type": "enabled"
    }


def test_thinking_absent_not_injected():
    provider = make_provider()
    assert "thinking" not in provider.prepare_payload({"messages": []})


def test_invalid_thinking_raises():
    provider = make_provider()
    with pytest.raises(ValueError, match="thinking"):
        provider.prepare_payload({"thinking": "maybe"})
    with pytest.raises(ValueError, match="thinking"):
        provider.prepare_payload({"thinking": 42})


def test_do_sample_false_drops_sampling_params():
    provider = make_provider()
    payload = provider.prepare_payload(
        {
            "messages": [],
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    )
    assert payload["do_sample"] is False
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_do_sample_true_keeps_sampling_params():
    provider = make_provider()
    payload = provider.prepare_payload(
        {"messages": [], "do_sample": True, "temperature": 0.6, "top_p": 0.95}
    )
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95


def test_invalid_do_sample_raises():
    provider = make_provider()
    with pytest.raises(ValueError, match="do_sample"):
        provider.prepare_payload({"do_sample": "yes"})


# ----------------------------------------------------------------------
# cache_control sanitation
# ----------------------------------------------------------------------


def test_cache_control_stripped_from_payload():
    provider = make_provider()
    payload = provider.prepare_payload(
        {
            "model": "claude-3-7-sonnet-20250219",
            "system": [
                {
                    "type": "text",
                    "text": "system prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "name": "read_file",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    )
    assert "cache_control" not in json.dumps(payload)
    assert payload["model"] == "glm-5.3"


def test_prepare_payload_does_not_mutate_input():
    provider = make_provider()
    original = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "hi"}],
    }
    snapshot = json.dumps(original)
    provider.prepare_payload(original)
    assert json.dumps(original) == snapshot


# ----------------------------------------------------------------------
# HTTP behavior (mocked transport)
# ----------------------------------------------------------------------


async def test_complete_request_and_response():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-glm-1",
                "model": "glm-5.3",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello!"}}
                ],
            },
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    response = await provider.complete(
        {
            "model": "claude-3-7-sonnet-20250219",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert response["choices"][0]["message"]["content"] == "Hello!"
    assert captured["url"] == GLMProvider.BIGMODEL_BASE_URL + "/chat/completions"
    assert captured["auth"] == "Bearer zk-test-key"
    assert captured["payload"]["model"] == "glm-5.3"
    assert captured["payload"].get("stream") is not True


async def test_complete_async_payload_mapping():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": []})

    provider = make_provider(transport=httpx.MockTransport(handler))
    await provider.complete(
        {
            "model": "claude-3-5-haiku-20241022",
            "messages": [],
            "thinking": {"type": "enabled"},
        }
    )
    assert captured["payload"]["model"] == "glm-5.3-flash"
    assert captured["payload"]["thinking"] == {"type": "enabled"}


async def test_stream_parses_sse_chunks():
    sse_body = "\n".join(
        [
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-glm-2",
                    "model": "glm-5.3",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "thinking..."},
                        }
                    ],
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-glm-2",
                    "model": "glm-5.3",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}}],
                }
            ),
            "",
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-glm-2",
                    "model": "glm-5.3",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                }
            ),
            "",
            "data: [DONE]",
            "",
            "",
        ]
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=sse_body.encode("utf-8"),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = [
        chunk
        async for chunk in provider.stream_complete(
            {
                "model": "claude-3-7-sonnet-20250219",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    ]

    assert len(chunks) == 3
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "thinking..."
    assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert "usage" in chunks[2]

    # The request must be converted and forced into streaming mode.
    assert captured["payload"]["model"] == "glm-5.3"
    assert captured["payload"]["stream"] is True


async def test_stream_survives_malformed_lines():
    sse_body = "\n".join(
        [
            "data: not-json",
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body.encode("utf-8"))

    provider = make_provider(transport=httpx.MockTransport(handler))
    chunks = [chunk async for chunk in provider.stream_complete({"messages": []})]
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["delta"]["content"] == "ok"
