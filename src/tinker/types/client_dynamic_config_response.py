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
