"""Anthropic Messages API to OpenAI Chat Completions API adapter."""

import json
import uuid
from typing import Any, Dict, List, Optional

from agentbridge.adapters.base import BaseAdapter
from agentbridge.adapters.thinking import ThinkingTransformer
from agentbridge.adapters.tools import ToolAdapter


class AnthropicToOpenAIAdapter(BaseAdapter):
    """Translates Anthropic /v1/messages requests and streams to OpenAI completions."""

    def __init__(self, model_map: Optional[Dict[str, str]] = None):
        self.model_map = model_map or {}

    def convert_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic /v1/messages payload to OpenAI /v1/chat/completions payload."""
        anthropic_model = payload.get("model", "")
        target_model = self.model_map.get(anthropic_model, anthropic_model)

        openai_messages: List[Dict[str, Any]] = []

        # 1. System prompt conversion
        system = payload.get("system")
        if system:
            if isinstance(system, str):
                openai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                system_text = "\n".join(
                    block.get("text", "")
                    for block in system
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if system_text:
                    openai_messages.append({"role": "system", "content": system_text})

        # 2. Messages conversion
        for msg in payload.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Handle blocks
                text_parts = []
                tool_calls = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")

                    if btype == "text":
                        text_parts.append(block.get("text", ""))

                    elif btype == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(
                                        block.get("input", {}), ensure_ascii=False
                                    ),
                                },
                            }
                        )

                    elif btype == "tool_result":
                        # Anthropic tool_result maps to OpenAI role 'tool'
                        raw_result = block.get("content", "")
                        result_str = (
                            raw_result
                            if isinstance(raw_result, str)
                            else json.dumps(raw_result, ensure_ascii=False)
                        )
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": result_str,
                            }
                        )

                # Assemble assistant or user message
                combined_text = "\n".join(text_parts) if text_parts else None
                if combined_text or tool_calls:
                    msg_dict: Dict[str, Any] = {"role": role}
                    if combined_text:
                        msg_dict["content"] = combined_text
                    if tool_calls and role == "assistant":
                        msg_dict["tool_calls"] = tool_calls
                    openai_messages.append(msg_dict)

        # 3. Request parameters
        req: Dict[str, Any] = {
            "model": target_model,
            "messages": openai_messages,
            "stream": payload.get("stream", False),
        }

        if "max_tokens" in payload:
            req["max_tokens"] = payload["max_tokens"]

        if "temperature" in payload:
            req["temperature"] = payload["temperature"]

        if "top_p" in payload:
            req["top_p"] = payload["top_p"]

        # 4. Tools conversion
        if "tools" in payload and payload["tools"]:
            req["tools"] = ToolAdapter.anthropic_tools_to_openai(payload["tools"])

        return req

    def convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI non-streaming response into Anthropic format."""
        msg_id = response.get("id", f"msg_{uuid.uuid4().hex[:12]}")
        model = response.get("model", "unknown")
        choices = response.get("choices", [])

        content_blocks: List[Dict[str, Any]] = []
        stop_reason = "end_turn"

        if choices:
            first = choices[0]
            finish_reason = first.get("finish_reason")
            message = first.get("message", {})

            # 1. Reasoning content
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if reasoning:
                content_blocks.append({"type": "thinking", "thinking": reasoning})

            # 2. Text content
            text = message.get("content")
            if text:
                content_blocks.append({"type": "text", "text": text})

            # 3. Tool calls
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = {}
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {"raw": fn.get("arguments", "")}

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )

            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "length":
                stop_reason = "max_tokens"

        usage = response.get("usage", {})
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    def convert_stream_chunk(
        self, chunk: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI SSE chunk into a sequence of Anthropic SSE events."""
        events: List[Dict[str, Any]] = []

        # Initialize stream context
        if not context.get("started"):
            msg_id = chunk.get("id", f"msg_{uuid.uuid4().hex[:12]}")
            model = chunk.get("model", "unknown")
            context["started"] = True
            context["msg_id"] = msg_id
            context["block_index"] = 0
            context["in_thinking"] = False
            context["in_text"] = False
            context["in_tool"] = False
            context["tool_index"] = -1

            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            )

        choices = chunk.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # 1. Reasoning / Thinking
        reasoning, content = ThinkingTransformer.extract_reasoning_and_content(chunk)

        if reasoning:
            if not context["in_thinking"]:
                context["in_thinking"] = True
                events.append(
                    ThinkingTransformer.build_thinking_start(context["block_index"])
                )
            events.append(
                ThinkingTransformer.build_thinking_delta(
                    context["block_index"], reasoning
                )
            )

        # 2. Text Content
        if content:
            # If we were in thinking, close thinking block first
            if context["in_thinking"]:
                events.append(
                    ThinkingTransformer.build_thinking_stop(context["block_index"])
                )
                context["in_thinking"] = False
                context["block_index"] += 1

            if not context["in_text"]:
                context["in_text"] = True
                events.append(
                    ThinkingTransformer.build_text_start(context["block_index"])
                )

            events.append(
                ThinkingTransformer.build_text_delta(context["block_index"], content)
            )

        # 3. Tool Calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_id = tc.get("id")
            fn = tc.get("function", {})
            name = fn.get("name")
            arguments = fn.get("arguments", "")

            # If switching blocks, close text or thinking
            if context["in_thinking"]:
                events.append(
                    ThinkingTransformer.build_thinking_stop(context["block_index"])
                )
                context["in_thinking"] = False
                context["block_index"] += 1

            if context["in_text"]:
                events.append(
                    ThinkingTransformer.build_text_stop(context["block_index"])
                )
                context["in_text"] = False
                context["block_index"] += 1

            if tc_id or name:
                # New tool call block
                if context["in_tool"]:
                    events.append(
                        ToolAdapter.build_tool_use_stop(context["block_index"])
                    )
                    context["block_index"] += 1

                context["in_tool"] = True
                context["tool_id"] = tc_id or f"toolu_{uuid.uuid4().hex[:8]}"
                context["tool_name"] = name or ""
                events.append(
                    ToolAdapter.build_tool_use_start(
                        context["block_index"], context["tool_id"], context["tool_name"]
                    )
                )

            if arguments and context["in_tool"]:
                events.append(
                    ToolAdapter.build_tool_use_delta(context["block_index"], arguments)
                )

        # 4. Finish reason handling
        if finish_reason:
            if context["in_thinking"]:
                events.append(
                    ThinkingTransformer.build_thinking_stop(context["block_index"])
                )
                context["in_thinking"] = False
                context["block_index"] += 1

            if context["in_text"]:
                events.append(
                    ThinkingTransformer.build_text_stop(context["block_index"])
                )
                context["in_text"] = False
                context["block_index"] += 1

            if context["in_tool"]:
                events.append(ToolAdapter.build_tool_use_stop(context["block_index"]))
                context["in_tool"] = False
                context["block_index"] += 1

            stop_reason = "end_turn"
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "length":
                stop_reason = "max_tokens"

            events.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                }
            )
            events.append({"type": "message_stop"})

        return events
