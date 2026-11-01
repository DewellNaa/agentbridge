"""High-availability upstream pool with multi-key rotation and failover.

The :class:`FailoverRouter` spreads traffic across several
:class:`UpstreamEndpoint` entries (one per API key, plus an optional
fallback upstream) and transparently retries requests on healthy
endpoints when the active one returns rate-limit / server errors or
fails at the network level.
"""

import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Set

import httpx

from agentbridge.providers.client import UpstreamClient

logger = logging.getLogger(__name__)

#: HTTP status codes that trigger a failover to the next endpoint.
FAILOVER_STATUS_CODES = frozenset({402, 429, 500, 502, 503, 504})

#: Network-level exceptions that trigger a failover.
FAILOVER_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)

#: Routing strategies supported by :class:`FailoverRouter`.
ROUTING_STRATEGIES = ("priority", "round_robin")


class AllUpstreamsFailedError(RuntimeError):
    """Raised when every endpoint in the pool is down or cooling down."""


@dataclass
class UpstreamEndpoint:
    """A single upstream candidate inside the failover pool."""

    name: str
    client: UpstreamClient
    weight: int = 1
    is_healthy: bool = True
    failure_count: int = 0
    cooldown_until: float = 0.0  # 冷却截止时间戳（perf_counter 或 time）
    cooldown_seconds: float = 30.0  # 故障冷却时长


class FailoverRouter:
    """Routes requests across endpoints with automatic failover."""

    def __init__(self, endpoints: List[UpstreamEndpoint], strategy: str = "priority"):
        if not endpoints:
            raise ValueError("FailoverRouter requires at least one endpoint")
        if strategy not in ROUTING_STRATEGIES:
            raise ValueError(
                f"Unknown failover strategy {strategy!r}; "
                f"expected one of {ROUTING_STRATEGIES}"
            )
        self.endpoints = endpoints
        self.strategy = strategy
        self._rr_index = 0

    # ------------------------------------------------------------------
    # Endpoint selection
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.perf_counter()

    def _recover(self) -> None:
        """Bring endpoints whose cooldown has expired back into rotation."""
        now = self._now()
        for endpoint in self.endpoints:
            if (
                not endpoint.is_healthy
                and endpoint.cooldown_until > 0
                and now >= endpoint.cooldown_until
            ):
                endpoint.is_healthy = True
                endpoint.cooldown_until = 0.0

    def _pick(self, exclude: Set[str]) -> UpstreamEndpoint:
        """Select the next endpoint, skipping ``exclude``d and unhealthy ones."""
        self._recover()
        healthy = [e for e in self.endpoints if e.is_healthy and e.name not in exclude]
        if not healthy:
            raise AllUpstreamsFailedError(
                "All upstream endpoints are down or cooling down"
            )
        if self.strategy == "round_robin":
            endpoint = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return endpoint
        return healthy[0]

    def get_next_endpoint(self) -> UpstreamEndpoint:
        """Return the active endpoint according to the routing strategy."""
        return self._pick(set())

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    def _is_failover_error(self, exc: BaseException) -> bool:
        """Return ``True`` when ``exc`` should trigger a failover retry."""
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in FAILOVER_STATUS_CODES
        return isinstance(exc, FAILOVER_EXCEPTIONS)

    def _mark_failure(self, endpoint: UpstreamEndpoint, error: BaseException) -> None:
        """Put an endpoint into cooldown and log the failover."""
        endpoint.failure_count += 1
        endpoint.is_healthy = False
        endpoint.cooldown_until = self._now() + endpoint.cooldown_seconds
        logger.warning(
            "[yellow]Failover:[/yellow] %s failed (%s), "
            "failing over to next upstream...",
            endpoint.name,
            error,
        )

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------
    async def complete(self, request: dict) -> tuple:
        """Complete a request, failing over until one endpoint succeeds.

        Returns ``(upstream_response, endpoint_name)``. Raises
        :class:`AllUpstreamsFailedError` when every endpoint fails.
        """
        exclude: Set[str] = set()
        last_error: Exception | None = None
        while True:
            try:
                endpoint = self._pick(exclude)
            except AllUpstreamsFailedError:
                if last_error is not None:
                    raise AllUpstreamsFailedError(
                        f"All upstream endpoints failed; last error: {last_error}"
                    ) from last_error
                raise
            exclude.add(endpoint.name)
            try:
                response = await endpoint.client.complete(request)
            except httpx.HTTPError as exc:
                if not self._is_failover_error(exc):
                    raise
                last_error = exc
                self._mark_failure(endpoint, exc)
            else:
                return response, endpoint.name

    async def stream_complete(self, request: dict) -> AsyncIterator[Dict]:
        """Stream a completion with pre-first-chunk failover.

        Failures before the first chunk reaches the client (connection
        errors, 429/5xx status responses) transparently retry on the next
        healthy endpoint. Once streaming has started, a mid-flight error is
        logged and the stream ends cleanly to avoid polluting the output.
        """
        exclude: Set[str] = set()
        last_error: Exception | None = None
        while True:
            try:
                endpoint = self._pick(exclude)
            except AllUpstreamsFailedError:
                if last_error is not None:
                    raise AllUpstreamsFailedError(
                        f"All upstream endpoints failed; last error: {last_error}"
                    ) from last_error
                raise
            exclude.add(endpoint.name)
            first_chunk_sent = False
            try:
                async for chunk in endpoint.client.stream_complete(dict(request)):
                    first_chunk_sent = True
                    yield chunk
                return
            except httpx.HTTPError as exc:
                if not first_chunk_sent and self._is_failover_error(exc):
                    last_error = exc
                    self._mark_failure(endpoint, exc)
                    continue
                if first_chunk_sent:
                    logger.error(
                        "[red]Failover:[/red] stream from %s aborted "
                        "mid-flight (%s); stopping to avoid stream pollution.",
                        endpoint.name,
                        exc,
                    )
                    return
                raise

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------
    def get_pool_status(self) -> List[Dict]:
        """Report name, health, failures and remaining cooldown per endpoint."""
        self._recover()
        now = self._now()
        return [
            {
                "name": endpoint.name,
                "is_healthy": endpoint.is_healthy,
                "failure_count": endpoint.failure_count,
                "cooldown_remaining": max(0.0, endpoint.cooldown_until - now)
                if endpoint.cooldown_until > 0
                else 0.0,
            }
            for endpoint in self.endpoints
        ]

    async def close(self) -> None:
        """Close every underlying upstream client."""
        for endpoint in self.endpoints:
            await endpoint.client.close()
