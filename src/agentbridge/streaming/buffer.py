"""Stream buffering and chunk boundary handlers."""

from typing import List


class StreamBuffer:
    """Buffers incoming byte chunks from upstream and yields complete lines."""

    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> List[str]:
        """Feed a string chunk and return all complete lines."""
        self._buffer += chunk
        lines = []

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip("\r")
            lines.append(line)

        return lines

    def flush(self) -> List[str]:
        """Flush any remaining buffered content."""
        if self._buffer:
            remaining = self._buffer.strip("\r")
            self._buffer = ""
            if remaining:
                return [remaining]
        return []
