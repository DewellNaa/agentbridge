"""Base adapter interfaces for protocol translation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAdapter(ABC):
    """Abstract base class for protocol converters."""

    @abstractmethod
    def convert_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a client request payload into upstream format."""
        raise NotImplementedError

    @abstractmethod
    def convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an upstream non-streaming response into client format."""
        raise NotImplementedError

    @abstractmethod
    def convert_stream_chunk(
        self, chunk: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert an upstream stream chunk into one or more client event dicts."""
        raise NotImplementedError
