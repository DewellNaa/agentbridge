"""Server-Sent Events (SSE) encoding and decoding."""

import json
from typing import Any, Dict, Optional, Tuple, Union


def format_sse(event: str, data: Union[str, Dict[str, Any]]) -> str:
    """Format an event and payload into standard SSE wire format.

    Example output:
    event: content_block_delta
    data: {"type": "content_block_delta", ...}

    """
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)

    return f"event: {event}\ndata: {payload}\n\n"


def parse_sse_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a single SSE line into (field, value)."""
    line = line.strip("\r\n")
    if not line or line.startswith(":"):
        return None, None

    if ":" in line:
        field, value = line.split(":", 1)
        if value.startswith(" "):
            value = value[1:]
        return field, value

    return line, ""
