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
    billing_exception_max_pause_duration_sec: int = 60 * 60
    # gRPC endpoint advertised by the server. Scheme-prefixed URL:
    # "grpc://host:port" (plaintext) or "grpcs://host:port" (TLS).
    # None means the SDK stays on REST.
    grpc_target: str | None = None
    # Gate for routing retrieve_future through gRPC. Only honored when
    # grpc_target is also set.
    enable_grpc_retrieve_future: bool = False
    sample_no_retries: bool = False
    use_pyqwest_transport: bool = True
    """When true, the SDK builds its default httpx async client on top of the
    pyqwest (reqwest/hyper-based) transport adapter. Set to false server-side
    to force every client to fall back to httpx's default transport."""
