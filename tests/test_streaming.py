"""Tests for SSE formatting and StreamBuffer."""

from agentbridge.streaming.buffer import StreamBuffer
from agentbridge.streaming.sse import format_sse, parse_sse_line


def test_format_sse():
    payload = {"type": "content_block_delta", "index": 0}
    formatted = format_sse("content_block_delta", payload)

    assert formatted.startswith("event: content_block_delta\n")
    assert 'data: {"type": "content_block_delta", "index": 0}' in formatted
    assert formatted.endswith("\n\n")


def test_parse_sse_line():
    field, val = parse_sse_line("event: message_start")
    assert field == "event"
    assert val == "message_start"

    field, val = parse_sse_line('data: {"type": "test"}')
    assert field == "data"
    assert val == '{"type": "test"}'


def test_stream_buffer_line_splitting():
    buffer = StreamBuffer()

    # Partial chunk 1
    lines1 = buffer.feed("data: chunk1\r\ndata: ch")
    assert lines1 == ["data: chunk1"]

    # Partial chunk 2 completing the line
    lines2 = buffer.feed("unk2\ndata: chunk3")
    assert lines2 == ["data: chunk2"]

    # Flush remaining
    lines3 = buffer.flush()
    assert lines3 == ["data: chunk3"]
