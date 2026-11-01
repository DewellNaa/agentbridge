"""High-availability routing: multi-key pools and failover."""

from agentbridge.routing.pool import (
    FAILOVER_EXCEPTIONS,
    FAILOVER_STATUS_CODES,
    ROUTING_STRATEGIES,
    AllUpstreamsFailedError,
    FailoverRouter,
    UpstreamEndpoint,
)

__all__ = [
    "AllUpstreamsFailedError",
    "FAILOVER_EXCEPTIONS",
    "FAILOVER_STATUS_CODES",
    "FailoverRouter",
    "ROUTING_STRATEGIES",
    "UpstreamEndpoint",
]
