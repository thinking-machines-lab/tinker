from __future__ import annotations

from .._models import BaseModel

__all__ = ["ClientConfigResponse"]


class ClientConfigResponse(BaseModel):
    """Server-side feature flags resolved for this caller.

    Uses BaseModel (extra="ignore") so new flags from the server are
    silently dropped until the SDK adds fields for them.
    """

    pjwt_auth_enabled: bool = False
    credential_default_source: str = "api_key"
    sample_dispatch_bytes_semaphore_size: int = 10 * 1024 * 1024
    inflight_response_bytes_semaphore_size: int = 50 * 1024 * 1024
