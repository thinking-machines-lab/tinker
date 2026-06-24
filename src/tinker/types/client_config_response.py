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
    parallel_fwdbwd_chunks: bool = True
    proto_write_fwdbwd: bool = False
    """When true, the SDK serializes ForwardBackwardRequest as proto bytes and
    POSTs with Content-Type: application/x-protobuf. Falls back to JSON when
    false (default) or when the request can't be encoded in proto."""
    proto_compress_fwdbwd: bool = False
    """When true (and proto_write_fwdbwd is also true), the SDK zstd-compresses
    the proto fwd/bwd request body and sets Content-Encoding: zstd. Real fwd/bwd
    payloads compress >10× — the API server decompresses transparently via an
    ASGI middleware. Ignored on the JSON path."""
    fwd_via_fwdbwd: bool = False
    """When true (and proto_write_fwdbwd is also true), TrainingClient.forward()
    routes through /api/v1/forward_backward with forward_only=True on the proto
    instead of /api/v1/forward. Ignored when proto_write_fwdbwd
    is false (the JSON /forward path remains)."""
    billing_exception_max_pause_duration_sec: int = 60 * 60
    sample_no_retries: bool = False
    sample_enable_stuck_detection: bool = True
    """When true, the SDK runs the retry handler's progress timeout check that
    raises ``APIConnectionError("...Requests appear to be stuck.")`` when no
    progress is made within ``RetryConfig.progress_timeout``."""
    sample_max_concurrent_requests: int = 2000
    """Maximum number of in-flight sampling requests a SamplingClient will allow
    (the size of its retry handler's semaphore). Always applied as the sampling
    ``RetryConfig.max_connections``, overriding any value in a caller-provided
    ``retry_config``."""
    use_pyqwest_transport: bool = True
    """When true, the SDK builds its default httpx async client on top of the
    pyqwest (reqwest/hyper-based) transport adapter. Set to false server-side
    to force every client to fall back to httpx's default transport."""
