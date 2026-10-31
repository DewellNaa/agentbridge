"""Tests for agentbridge.adapters.cache_control.sanitize_cache_control."""

import copy

import pytest

from agentbridge.adapters.cache_control import sanitize_cache_control

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_removes_cache_control_from_text_blocks():
    payload = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }

    clean = sanitize_cache_control(payload)

    assert clean["messages"][0]["content"][0] == {"type": "text", "text": "hello"}


def test_removes_cache_control_from_string_content_message():
    """String content carries no marker; the message must round-trip intact."""
    payload = {
        "messages": [{"role": "user", "content": "plain string"}],
    }

    assert sanitize_cache_control(payload) == payload


def test_removes_cache_control_from_tool_use_and_tool_result_blocks():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "main.py"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {
                                "type": "text",
                                "text": "file body",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ],
            },
        ],
    }

    clean = sanitize_cache_control(payload)

    assert "cache_control" not in clean["messages"][0]["content"][0]
    assert "cache_control" not in clean["messages"][1]["content"][0]["content"][0]
    # Sibling fields survive untouched.
    assert clean["messages"][0]["content"][0]["input"] == {"path": "main.py"}
    assert clean["messages"][1]["content"][0]["tool_use_id"] == "toolu_1"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_removes_cache_control_from_system_block_list():
    payload = {
        "system": [
            {
                "type": "text",
                "text": "You are a coding assistant.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [],
    }

    clean = sanitize_cache_control(payload)

    assert clean["system"] == [{"type": "text", "text": "You are a coding assistant."}]


def test_keeps_string_system_prompt():
    payload = {"system": "You are a coding assistant.", "messages": []}

    assert sanitize_cache_control(payload) == payload


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_removes_cache_control_from_tool_declarations():
    payload = {
        "tools": [
            {
                "name": "bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [],
    }

    clean = sanitize_cache_control(payload)

    assert "cache_control" not in clean["tools"][0]
    assert clean["tools"][0]["input_schema"]["properties"]["command"] == {
        "type": "string"
    }


def test_removes_cache_control_from_nested_input_schema():
    payload = {
        "tools": [
            {
                "name": "edit",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "cache_control": {"type": "ephemeral"},
                            },
                        }
                    },
                },
            }
        ]
    }

    clean = sanitize_cache_control(payload)

    schema = clean["tools"][0]["input_schema"]
    assert "cache_control" not in schema["properties"]["edits"]["items"]


# ---------------------------------------------------------------------------
# Recursion / nesting
# ---------------------------------------------------------------------------


def test_strips_every_marker_from_full_claude_code_payload():
    """A realistic Claude Code request: markers appear at many depths."""
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 8192,
        "system": [
            {
                "type": "text",
                "text": "system prompt",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [
            {
                "name": "bash",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
        "metadata": {"user_id": "u1", "cache_control": {"type": "ephemeral"}},
    }

    clean = sanitize_cache_control(payload)

    text = repr(clean)
    assert "cache_control" not in text
    assert clean["model"] == "claude-sonnet-4-5"
    assert clean["max_tokens"] == 8192
    assert clean["metadata"] == {"user_id": "u1"}


def test_deeply_nested_lists_and_dicts():
    payload = {"a": [{"b": [{"c": {"cache_control": {"type": "ephemeral"}, "d": 1}}]}]}

    clean = sanitize_cache_control(payload)

    assert clean == {"a": [{"b": [{"c": {"d": 1}}]}]}


def test_payload_without_cache_control_round_trips_unchanged():
    payload = {
        "model": "glm-5.3",
        "stream": True,
        "temperature": 0.7,
        "system": [{"type": "text", "text": "sys"}],
        "tools": [{"name": "grep", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
    }
    snapshot = copy.deepcopy(payload)

    clean = sanitize_cache_control(payload)

    assert clean == snapshot
    assert clean is not payload  # a copy, not the same object


def test_cache_control_value_variants_are_all_dropped():
    """Any cache_control value (ephemeral, ttl, garbage) is removed."""
    payload = {
        "messages": [
            {"role": "user", "content": "x", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "y", "cache_control": {"ttl": "5m"}},
            {"role": "user", "content": "z", "cache_control": "bogus"},
        ]
    }

    clean = sanitize_cache_control(payload)

    for msg in clean["messages"]:
        assert "cache_control" not in msg


def test_empty_and_falsy_values_are_preserved():
    payload = {
        "messages": [],
        "system": None,
        "stop_sequences": [],
        "temperature": 0,
        "metadata": {},
    }

    clean = sanitize_cache_control(payload)

    assert clean == payload
    assert clean["messages"] is not payload["messages"]  # fresh containers


def test_circular_reference_does_not_recurse_forever():
    inner: dict = {"role": "user", "content": "x"}
    outer: dict = {"messages": [inner], "cache_control": {"type": "ephemeral"}}
    inner["self_ref"] = outer  # type: ignore[assignment]

    clean = sanitize_cache_control(outer)

    assert "cache_control" not in clean
    assert "cache_control" not in clean["messages"][0]["self_ref"]
    assert clean["messages"][0]["content"] == "x"


def test_shared_substructures_are_copied_consistently():
    shared = {"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}}
    payload = {"messages": [{"content": [shared]}, {"content": [shared]}]}

    clean = sanitize_cache_control(payload)

    first, second = clean["messages"][0]["content"][0], clean["messages"][1]["content"][0]
    assert "cache_control" not in first and "cache_control" not in second
    assert first is second  # shared object stays shared, not duplicated


def test_input_payload_is_not_mutated():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ]
    }
    snapshot = copy.deepcopy(payload)

    sanitize_cache_control(payload)

    assert payload == snapshot


def test_top_level_cache_control_is_removed():
    payload = {"cache_control": {"type": "ephemeral"}, "model": "claude-sonnet-4-5"}

    clean = sanitize_cache_control(payload)

    assert clean == {"model": "claude-sonnet-4-5"}


def test_non_dict_values_inside_lists_are_kept():
    payload = {"messages": ["plain", 42, None, True, {"cache_control": 1}]}

    clean = sanitize_cache_control(payload)

    assert clean == {"messages": ["plain", 42, None, True, {}]}


def test_tuple_values_are_preserved_as_tuples():
    payload = {"flags": ("a", "b"), "messages": ()}

    clean = sanitize_cache_control(payload)

    assert clean["flags"] == ("a", "b")
    assert isinstance(clean["flags"], tuple)


def test_invalid_payload_type_raises_type_error():
    with pytest.raises(TypeError):
        sanitize_cache_control(["not", "a", "dict"])  # type: ignore[arg-type]
