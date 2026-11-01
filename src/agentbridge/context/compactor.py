"""Context budget estimation and multi-stage smart compaction engine."""

import copy
import json
import math
from collections import Counter
from typing import Any

# Head/tail characters kept when folding an oversized tool output.
_HEAD_CHARS = 800
_TAIL_CHARS = 800

_TRUNCATION_MARKER = (
    "\n... [AgentBridge: Truncated {diff} chars of older tool output "
    "(saved ~{saved} tokens)] ...\n"
)

_FOLD_USER_TEMPLATE = (
    "[AgentBridge: Compressed {n} intermediate conversation turns "
    "to preserve context budget]"
)

_FOLD_ASSISTANT_TEMPLATE = (
    "[AgentBridge: Acknowledged. {n} older turns folded; the original goal "
    "and recent context remain available.]"
)

# Weighted char/token heuristic (~4 ASCII chars per token).
_ASCII_WEIGHT = 0.25
_SYMBOL_WEIGHT = 0.3  # code punctuation is denser than prose
_NON_ASCII_WEIGHT = 1.0  # CJK ideographs cost roughly one token per char


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` without any tokenizer dependency.

    ASCII letters/digits/whitespace cost ~0.25 tokens per char, ASCII
    symbols (code punctuation) slightly more, and non-ASCII characters
    (e.g. CJK) roughly one token per char.
    """
    if not text:
        return 0
    counts = Counter(text)
    weighted = 0.0
    for ch, n in counts.items():
        if not ch.isascii():
            weighted += _NON_ASCII_WEIGHT * n
        elif ch.isalnum() or ch.isspace():
            weighted += _ASCII_WEIGHT * n
        else:
            weighted += _SYMBOL_WEIGHT * n
    return max(1, math.ceil(weighted))


def _estimate_block_tokens(block: Any) -> int:
    """Estimate tokens contributed by a single Anthropic content block."""
    if not isinstance(block, dict):
        if isinstance(block, str):
            return estimate_tokens(block)
        return 0
    btype = block.get("type")
    if btype == "text":
        return estimate_tokens(block.get("text", ""))
    if btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return estimate_tokens(content)
        if isinstance(content, list):
            return sum(_estimate_block_tokens(item) for item in content)
        return 0
    if btype == "tool_use":
        try:
            payload = json.dumps(block.get("input", {}), ensure_ascii=False)
        except (TypeError, ValueError):
            payload = str(block.get("input", ""))
        return estimate_tokens(str(block.get("name", "")) + payload)
    if btype == "thinking":
        return estimate_tokens(block.get("thinking", ""))
    return 0


def _estimate_message_tokens(message: Any) -> int:
    """Estimate tokens contributed by a single message dict."""
    if not isinstance(message, dict):
        return 0
    content = message.get("content", "")
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        return sum(_estimate_block_tokens(block) for block in content)
    return 0


def estimate_payload_tokens(messages: list[dict], system: Any = None) -> int:
    """Estimate the total token count of a full Anthropic-style payload."""
    total = 0
    for message in messages or []:
        total += _estimate_message_tokens(message)
    if isinstance(system, str):
        total += estimate_tokens(system)
    elif isinstance(system, list):
        total += sum(_estimate_block_tokens(block) for block in system)
    return total


def _tool_result_text(block: dict) -> str:
    """Extract the flat text carried by a tool_result (or text) block."""
    if block.get("type") == "text":
        return block.get("text", "")
    if block.get("type") == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
    return ""


def _write_back_truncated(block: dict, text: str) -> None:
    """Store folded text back into a tool_result or text block."""
    if block.get("type") == "text":
        block["text"] = text
    else:
        block["content"] = text


class ContextCompactor:
    """Two-stage context budget valve for long agent sessions.

    Stage 1 folds oversized tool outputs in older turns (head/tail keep).
    Stage 2 folds whole intermediate turns into a compact placeholder pair
    when truncation alone is not enough to fit the budget.
    """

    def __init__(
        self,
        max_context_tokens: int | None = None,
        preserve_recent_turns: int = 4,
        max_tool_result_chars: int = 2000,
        preserve_first_turn: bool = True,
    ):
        self.max_context_tokens = max_context_tokens
        self.preserve_recent_turns = preserve_recent_turns
        self.max_tool_result_chars = max_tool_result_chars
        self.preserve_first_turn = preserve_first_turn

    def _protected_indices(self, total: int) -> set[int]:
        """Indices that must never be truncated or folded away."""
        recent_start = max(total - self.preserve_recent_turns, 0)
        protected = set(range(recent_start, total))
        if self.preserve_first_turn and total > 0:
            protected.add(0)
        return protected

    def _fold_oversized_text(self, text: str) -> tuple[str, int]:
        """Fold one oversized text into head + marker + tail.

        Returns ``(new_text, saved_tokens)``.
        """
        head = _HEAD_CHARS
        tail = _TAIL_CHARS
        if head + tail > self.max_tool_result_chars:
            head = max(self.max_tool_result_chars // 2, 1)
            tail = max(self.max_tool_result_chars - head, 0)
        kept = text[:head] + (text[-tail:] if tail else "")
        saved = estimate_tokens(text) - estimate_tokens(kept)
        marker = _TRUNCATION_MARKER.format(diff=len(text) - head - tail, saved=saved)
        return text[:head] + marker + (text[-tail:] if tail else ""), saved

    def _truncate_old_tool_results(
        self, messages: list[dict], protected: set[int]
    ) -> int:
        """Stage 1: head/tail fold oversized blocks in unprotected turns."""
        truncated = 0
        for index, message in enumerate(messages):
            if index in protected or not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                if len(content) > self.max_tool_result_chars:
                    folded, _saved = self._fold_oversized_text(content)
                    message["content"] = folded
                    truncated += 1
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = _tool_result_text(block)
                if len(text) <= self.max_tool_result_chars:
                    continue
                folded, _saved = self._fold_oversized_text(text)
                _write_back_truncated(block, folded)
                truncated += 1
        return truncated

    def _fold_middle_turns(
        self, messages: list[dict], protected: set[int]
    ) -> tuple[list[dict], int]:
        """Stage 2: collapse unprotected intermediate turns into a placeholder."""
        middle = [index for index in range(len(messages)) if index not in protected]
        if not middle:
            return messages, 0
        n = len(middle)
        folded: list[dict] = []
        for index, message in enumerate(messages):
            if index in protected:
                folded.append(message)
            elif index == middle[0]:
                folded.append(
                    {
                        "role": "user",
                        "content": _FOLD_USER_TEMPLATE.format(n=n),
                    }
                )
                folded.append(
                    {
                        "role": "assistant",
                        "content": _FOLD_ASSISTANT_TEMPLATE.format(n=n),
                    }
                )
        return folded, n

    def compact(
        self, messages: list[dict], system: Any = None
    ) -> tuple[list[dict], dict]:
        """Compact ``messages`` to fit the context budget.

        Returns ``(compacted_messages, stats)``. When no budget is set or
        the payload already fits, ``messages`` is returned untouched and
        ``stats`` reports ``compressed=False``. The input structures are
        never mutated; compaction operates on a deep copy.
        """
        original_tokens = estimate_payload_tokens(messages, system)
        stats = {
            "compressed": False,
            "original_tokens": original_tokens,
            "compacted_tokens": original_tokens,
            "saved_tokens": 0,
            "truncated_tools": 0,
            "folded_turns": 0,
        }
        if (
            self.max_context_tokens is None
            or original_tokens <= self.max_context_tokens
        ):
            return messages, stats

        working = copy.deepcopy(messages)
        protected = self._protected_indices(len(working))

        truncated = self._truncate_old_tool_results(working, protected)
        compacted_tokens = estimate_payload_tokens(working, system)

        folded = 0
        if compacted_tokens > self.max_context_tokens:
            working, folded = self._fold_middle_turns(working, protected)
            compacted_tokens = estimate_payload_tokens(working, system)

        stats.update(
            compressed=True,
            compacted_tokens=compacted_tokens,
            saved_tokens=max(original_tokens - compacted_tokens, 0),
            truncated_tools=truncated,
            folded_turns=folded,
        )
        return working, stats
