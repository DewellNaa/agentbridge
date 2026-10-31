"""Tests for tool conversion and streaming deltas."""

from agentbridge.adapters.tools import ToolAdapter


def test_tool_definitions_translation():
    anthropic_tools = [
        {
            "name": "search_code",
            "description": "Searches codebase for pattern",
            "input_schema": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        }
    ]

    openai_tools = ToolAdapter.anthropic_tools_to_openai(anthropic_tools)
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "search_code"
    assert openai_tools[0]["function"]["description"] == "Searches codebase for pattern"
    assert (
        openai_tools[0]["function"]["parameters"] == anthropic_tools[0]["input_schema"]
    )


def test_streaming_tool_call_chunks(adapter):
    """Test tool call streaming chunks conversion."""
    context = {}

    # Chunk 1: Tool call initiation
    chunk1 = {
        "id": "chatcmpl-tc-1",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc123",
                            "function": {
                                "name": "run_command",
                                "arguments": '{"cmd": ',
                            },
                        }
                    ]
                }
            }
        ],
    }
    events1 = adapter.convert_stream_chunk(chunk1, context)
    assert any(
        e.get("type") == "content_block_start"
        and e["content_block"]["type"] == "tool_use"
        for e in events1
    )
    assert any(
        e.get("type") == "content_block_delta"
        and e["delta"]["type"] == "input_json_delta"
        for e in events1
    )

    # Chunk 2: Tool argument delta
    chunk2 = {
        "id": "chatcmpl-tc-1",
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '"pytest"}'},
                        }
                    ]
                }
            }
        ],
    }
    events2 = adapter.convert_stream_chunk(chunk2, context)
    assert any(
        e.get("type") == "content_block_delta"
        and e["delta"]["partial_json"] == '"pytest"}'
        for e in events2
    )

    # Chunk 3: Finished
    chunk3 = {
        "id": "chatcmpl-tc-1",
        "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
    }
    events3 = adapter.convert_stream_chunk(chunk3, context)
    assert any(e.get("type") == "content_block_stop" for e in events3)
    assert any(
        e.get("type") == "message_delta" and e["delta"]["stop_reason"] == "tool_use"
        for e in events3
    )
