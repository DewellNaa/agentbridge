"""Tool calling schema and streaming conversions.

Translates between Anthropic Tools (input_schema / tool_use / tool_result)
and OpenAI Function Calling (tools / tool_calls / tool role).
"""

from typing import Any, Dict, List


class ToolAdapter:
    """Converts tool definitions and tool calls between Anthropic and OpenAI formats."""

    @staticmethod
    def anthropic_tools_to_openai(
        anthropic_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic tool definitions to OpenAI functions."""
        openai_tools = []
        for tool in anthropic_tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("input_schema", {"type": "object", "properties": {}})

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return openai_tools

    @staticmethod
    def openai_tools_to_anthropic(
        openai_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI function tools to Anthropic tool definitions."""
        anthropic_tools = []
        for tool in openai_tools:
            fn = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return anthropic_tools

    @staticmethod
    def build_tool_use_start(
        index: int, tool_id: str, tool_name: str
    ) -> Dict[str, Any]:
        """Build an Anthropic content_block_start event for tool_use."""
        return {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": {},
            },
        }

    @staticmethod
    def build_tool_use_delta(index: int, partial_json: str) -> Dict[str, Any]:
        """Build an Anthropic content_block_delta event for input_json_delta."""
        return {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "input_json_delta",
                "partial_json": partial_json,
            },
        }

    @staticmethod
    def build_tool_use_stop(index: int) -> Dict[str, Any]:
        """Build an Anthropic content_block_stop event."""
        return {"type": "content_block_stop", "index": index}
