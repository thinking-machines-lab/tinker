from __future__ import annotations

from .._models import BaseModel

__all__ = ["ClientDynamicConfigResponse"]


class ClientDynamicConfigResponse(BaseModel):
    """Server-side flags re-fetched periodically by a live client.

    Unlike ClientConfigResponse (fetched once at client creation), these
    flags are refreshed in the background at refresh_interval_sec cadence,
    so server-side changes take effect on the next request without
    recreating the client. Uses BaseModel (extra="ignore") so new flags
    from the server are silently dropped until the SDK adds fields for
    them.
    """

    refresh_interval_sec: int = 300
    """How often the SDK re-fetches this config from the server."""

    sample_cancel_enabled: bool = False
    """When true, abandoning a sampling future (timeout or local cancel) sends
    an explicit cancel_future request to the server so the in-flight sampling
    work is stopped promptly. When false, cancellation falls back to the
    server's SDK-heartbeat-expiry path."""

    sample_cancel_max_batch_size: int = 64
    """Maximum number of queued sampling cancellations the client dispatches in
    parallel per drain iteration."""
