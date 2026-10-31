"""Tests for AnthropicToOpenAIAdapter."""

import json


def test_convert_simple_request(adapter):
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "system": "You are a helpful coding assistant.",
        "messages": [
            {"role": "user", "content": "Hello, write a binary search function."}
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    req = adapter.convert_request(payload)

    assert req["model"] == "deepseek-chat"
    assert req["max_tokens"] == 1024
    assert req["temperature"] == 0.2
    assert len(req["messages"]) == 2
    assert req["messages"][0] == {
        "role": "system",
        "content": "You are a helpful coding assistant.",
    }
    assert req["messages"][1] == {
        "role": "user",
        "content": "Hello, write a binary search function.",
    }


def test_convert_complex_block_messages(adapter):
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Run this tool"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "read_file",
                        "input": {"path": "main.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "file contents",
                    }
                ],
            },
        ],
    }

    req = adapter.convert_request(payload)

    assert len(req["messages"]) == 3
    # user text
    assert req["messages"][0] == {"role": "user", "content": "Run this tool"}
    # assistant tool_calls
    assert req["messages"][1]["role"] == "assistant"
    assert len(req["messages"][1]["tool_calls"]) == 1
    tc = req["messages"][1]["tool_calls"][0]
    assert tc["id"] == "toolu_123"
    assert tc["function"]["name"] == "read_file"
    assert json.loads(tc["function"]["arguments"]) == {"path": "main.py"}
    # tool result
    assert req["messages"][2] == {
        "role": "tool",
        "tool_call_id": "toolu_123",
        "content": "file contents",
    }


def test_convert_response(adapter):
    openai_resp = {
        "id": "chatcmpl-test",
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Here is your code.",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    anthropic_resp = adapter.convert_response(openai_resp)

    assert anthropic_resp["id"] == "chatcmpl-test"
    assert anthropic_resp["role"] == "assistant"
    assert anthropic_resp["stop_reason"] == "end_turn"
    assert anthropic_resp["content"] == [{"type": "text", "text": "Here is your code."}]
    assert anthropic_resp["usage"]["input_tokens"] == 10
    assert anthropic_resp["usage"]["output_tokens"] == 20
