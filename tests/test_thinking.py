"""Tests for reasoning tokens and thinking block stream conversion."""

from agentbridge.adapters.thinking import ThinkingTransformer


def test_thinking_transformer_extract():
    chunk = {
        "choices": [
            {
                "delta": {
                    "reasoning_content": "Let me think about this step by step...",
                    "content": None,
                }
            }
        ]
    }

    reasoning, content = ThinkingTransformer.extract_reasoning_and_content(chunk)
    assert reasoning == "Let me think about this step by step..."
    assert content is None


def test_thinking_stream_transition(adapter):
    """Test full stream lifecycle: thinking block -> text block -> stop."""
    context = {}

    # Chunk 1: Thinking delta
    chunk1 = {
        "id": "chatcmpl-stream-1",
        "model": "deepseek-reasoner",
        "choices": [{"delta": {"reasoning_content": "Thinking..."}}],
    }
    events1 = adapter.convert_stream_chunk(chunk1, context)
    assert any(e.get("type") == "message_start" for e in events1)
    assert any(
        e.get("type") == "content_block_start"
        and e["content_block"]["type"] == "thinking"
        for e in events1
    )
    assert any(
        e.get("type") == "content_block_delta"
        and e["delta"]["type"] == "thinking_delta"
        for e in events1
    )

    # Chunk 2: Switch to content
    chunk2 = {
        "id": "chatcmpl-stream-1",
        "choices": [{"delta": {"content": "Here is the answer."}}],
    }
    events2 = adapter.convert_stream_chunk(chunk2, context)
    # Must stop thinking block, start text block, and emit text_delta
    assert any(e.get("type") == "content_block_stop" for e in events2)
    assert any(
        e.get("type") == "content_block_start" and e["content_block"]["type"] == "text"
        for e in events2
    )
    assert any(
        e.get("type") == "content_block_delta" and e["delta"]["type"] == "text_delta"
        for e in events2
    )

    # Chunk 3: Finished
    chunk3 = {
        "id": "chatcmpl-stream-1",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    events3 = adapter.convert_stream_chunk(chunk3, context)
    assert any(e.get("type") == "content_block_stop" for e in events3)
    assert any(e.get("type") == "message_delta" for e in events3)
    assert any(e.get("type") == "message_stop" for e in events3)
