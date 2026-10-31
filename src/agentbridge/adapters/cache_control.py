"""Anthropic Prompt Caching cache_control sanitizer.

Claude Code aggressively annotates messages, system prompts and tool
declarations with ``cache_control: {"type": "ephemeral"}`` breakpoints.
Upstream vendors without Prompt Cache support (DeepSeek, GLM, most
OpenAI-compatible endpoints) reject such fields with::

    400 Bad Request: Extra inputs are not permitted

This module provides :func:`sanitize_cache_control`, a recursive, non-mutating
cleaner that strips every ``cache_control`` marker from an Anthropic
/v1/messages payload before it is forwarded upstream.
"""

from typing import Any, Dict, List

#: The key injected by Claude Code / Anthropic SDK for prompt cache breakpoints.
CACHE_CONTROL_KEY = "cache_control"


def _sanitize(value: Any, _memo: Dict[int, Any]) -> Any:
    """Recursively copy ``value`` while dropping all ``cache_control`` keys."""
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _memo:
            return _memo[obj_id]

        sanitized: Dict[str, Any] = {}
        _memo[obj_id] = sanitized
        for key, item in value.items():
            if key == CACHE_CONTROL_KEY:
                continue
            sanitized[key] = _sanitize(item, _memo)
        return sanitized

    if isinstance(value, list):
        seq_id = id(value)
        if seq_id in _memo:
            return _memo[seq_id]

        sanitized_list: List[Any] = []
        _memo[seq_id] = sanitized_list
        for item in value:
            sanitized_list.append(_sanitize(item, _memo))
        return sanitized_list

    if isinstance(value, tuple):
        return tuple(_sanitize(item, _memo) for item in value)

    # Scalars (str, int, float, bool, None, ...) and any unknown objects are
    # immutable with respect to this transformation; return them as-is.
    return value


def sanitize_cache_control(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively strip all ``cache_control`` markers from ``payload``.

    The function walks messages, system prompt blocks, tool declarations and
    any arbitrarily nested structure, removing every ``cache_control`` key
    while preserving all other fields, their order and their values.

    Args:
        payload: An Anthropic /v1/messages request payload.

    Returns:
        A new sanitized payload. The input dict is never mutated, and objects
        without any ``cache_control`` marker round-trip unchanged.

    Example:
        >>> sanitize_cache_control(
        ...     {"messages": [{"role": "user",
        ...                    "content": [{"type": "text", "text": "hi",
        ...                                 "cache_control": {"type": "ephemeral"}}]}]}
        ... ) == {"messages": [{"role": "user",
        ...                     "content": [{"type": "text", "text": "hi"}]}]}
        True
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"payload must be a dict, got {type(payload).__name__}"
        )
    return _sanitize(payload, {})
