"""Thinking and reasoning token transformations.

Bridges reasoning tokens between DeepSeek-R1 / GLM / Qwen reasoning_content
and Anthropic Claude 3.7 thinking blocks.
"""

from typing import Any, Dict, Optional, Tuple


class ThinkingTransformer:
    """Manages reasoning tokens and translates them into Anthropic content blocks."""

    @staticmethod
    def extract_reasoning_and_content(
        chunk: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract reasoning_content and regular content from an OpenAI delta chunk."""
        choices = chunk.get("choices", [])
        if not choices:
            return None, None

        delta = choices[0].get("delta", {})

        # 1. Standard OpenAI / DeepSeek delta.reasoning_content
        reasoning = delta.get("reasoning_content")

        # 2. GLM / other providers sometimes use reasoning or reasoning_text
        if reasoning is None:
            reasoning = delta.get("reasoning")

        # Regular text content
        content = delta.get("content")

        return reasoning, content

    @staticmethod
    def build_thinking_start(index: int = 0) -> Dict[str, Any]:
        """Build an Anthropic content_block_start for thinking."""
        return {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""},
        }

    @staticmethod
    def build_thinking_delta(index: int, thinking_chunk: str) -> Dict[str, Any]:
        """Build an Anthropic content_block_delta for thinking_delta."""
        return {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking_chunk},
        }

    @staticmethod
    def build_thinking_stop(index: int) -> Dict[str, Any]:
        """Build an Anthropic content_block_stop."""
        return {"type": "content_block_stop", "index": index}

    @staticmethod
    def build_text_start(index: int) -> Dict[str, Any]:
        """Build an Anthropic content_block_start for text."""
        return {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }

    @staticmethod
    def build_text_delta(index: int, text_chunk: str) -> Dict[str, Any]:
        """Build an Anthropic content_block_delta for text_delta."""
        return {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text_chunk},
        }

    @staticmethod
    def build_text_stop(index: int) -> Dict[str, Any]:
        """Build an Anthropic content_block_stop for text."""
        return {"type": "content_block_stop", "index": index}
