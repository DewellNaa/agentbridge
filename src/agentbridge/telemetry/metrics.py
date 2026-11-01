"""In-memory request telemetry: metric dataclasses, collection and aggregation."""

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class RequestMetric:
    """Telemetry for a single proxied /v1/messages request."""

    request_id: str
    timestamp: float
    client_agent: str = "Unknown"
    model_requested: str = ""
    model_upstream: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    ttft_ms: float = 0.0
    tools_called: List[str] = field(default_factory=list)
    status_code: int = 200
    is_stream: bool = True
    error_message: Optional[str] = None
    # High-resolution clock anchor for TTFT / duration measurement.
    _perf_start: float = field(
        default_factory=time.perf_counter, repr=False, compare=False
    )


class TelemetryManager:
    """Collects per-request metrics into a ring buffer and aggregate counters."""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._history: Deque[RequestMetric] = deque(maxlen=max_history)
        self._in_flight: Dict[str, RequestMetric] = {}
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def start_request(
        self,
        req_id: str,
        model_req: str,
        model_upstream: str,
        client_agent: str,
        is_stream: bool,
    ) -> RequestMetric:
        """Register a new in-flight request and return its metric."""
        metric = RequestMetric(
            request_id=req_id,
            timestamp=time.time(),
            client_agent=client_agent or "Unknown",
            model_requested=model_req,
            model_upstream=model_upstream,
            is_stream=is_stream,
        )
        with self._lock:
            self._in_flight[req_id] = metric
            self.total_requests += 1
        return metric

    def record_first_token(self, metric: RequestMetric) -> None:
        """Record time-to-first-token on the first streamed chunk arrival."""
        with self._lock:
            if metric.ttft_ms == 0.0:
                elapsed = (time.perf_counter() - metric._perf_start) * 1000.0
                metric.ttft_ms = max(0.0, elapsed)

    def record_tokens(
        self,
        metric: RequestMetric,
        prompt: int = 0,
        completion: int = 0,
        reasoning: int = 0,
    ) -> None:
        """Accumulate token usage deltas onto the metric and global totals."""
        with self._lock:
            metric.prompt_tokens += prompt
            metric.completion_tokens += completion
            metric.reasoning_tokens += reasoning
            metric.total_tokens = (
                metric.prompt_tokens
                + metric.completion_tokens
                + metric.reasoning_tokens
            )
            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.total_reasoning_tokens += reasoning

    def record_tool_call(self, metric: RequestMetric, tool_name: str) -> None:
        """Record a tool invocation name (deduplicated per request)."""
        if not tool_name:
            return
        with self._lock:
            if tool_name not in metric.tools_called:
                metric.tools_called.append(tool_name)

    def finish_request(
        self,
        metric: RequestMetric,
        status_code: int = 200,
        error_message: Optional[str] = None,
    ) -> None:
        """Finalize a metric: compute duration and move it into history."""
        with self._lock:
            metric.status_code = status_code
            metric.error_message = error_message
            elapsed = (time.perf_counter() - metric._perf_start) * 1000.0
            metric.duration_ms = max(0.0, elapsed)
            metric.total_tokens = (
                metric.prompt_tokens
                + metric.completion_tokens
                + metric.reasoning_tokens
            )
            self._in_flight.pop(metric.request_id, None)
            self._history.append(metric)
            if status_code < 400 and error_message is None:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    def get_summary(self) -> Dict[str, object]:
        """Return a JSON-serializable overview of the accumulated telemetry."""
        with self._lock:
            finished = list(self._history)
            summary: Dict[str, object] = {
                "uptime_seconds": round(time.time() - self.start_time, 3),
                "start_time": self.start_time,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "active_requests": len(self._in_flight),
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "total_tokens": (
                    self.total_prompt_tokens
                    + self.total_completion_tokens
                    + self.total_reasoning_tokens
                ),
                "avg_duration_ms": (
                    round(sum(m.duration_ms for m in finished) / len(finished), 2)
                    if finished
                    else 0.0
                ),
                "avg_ttft_ms": (
                    round(sum(m.ttft_ms for m in finished) / len(finished), 2)
                    if finished
                    else 0.0
                ),
                "tool_usage": dict(
                    Counter(t for m in finished for t in m.tools_called)
                ),
                "recent_requests": [
                    {
                        "request_id": m.request_id,
                        "time": time.strftime("%H:%M:%S", time.localtime(m.timestamp)),
                        "status_code": m.status_code,
                        "client_agent": m.client_agent,
                        "model_requested": m.model_requested,
                        "model_upstream": m.model_upstream,
                        "prompt_tokens": m.prompt_tokens,
                        "completion_tokens": m.completion_tokens,
                        "reasoning_tokens": m.reasoning_tokens,
                        "duration_ms": round(m.duration_ms, 2),
                        "ttft_ms": round(m.ttft_ms, 2),
                        "is_stream": m.is_stream,
                        "tools_called": list(m.tools_called),
                        "error_message": m.error_message,
                    }
                    for m in list(finished)[-10:]
                ],
            }
            return summary
