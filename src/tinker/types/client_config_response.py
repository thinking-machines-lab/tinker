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
    proto_compress_fwdbwd: bool = False
    """When true, the SDK zstd-compresses the proto fwd/bwd request body and
    sets Content-Encoding: zstd. Real fwd/bwd payloads compress >10× — the API
    server decompresses transparently via an ASGI middleware."""
    fwdbwd_max_chunk_len: int = 1024
    """Maximum number of datums per fwd/bwd chunk request."""
    fwdbwd_max_chunk_bytes_count: int = 5_000_000
    """Maximum estimated request payload bytes (serialized proto, before zstd
    compression) per fwd/bwd chunk request."""
    fwdbwd_dispatch_bytes_semaphore_size: int = 50 * 1000 * 1000
    """Maximum estimated fwd/bwd request payload bytes being dispatched at the
    same time (same pre-compression estimate as fwdbwd_max_chunk_bytes_count).
    Bounds client memory for serialized bodies and smooths submission bursts."""
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
    create_model_via_load_weights: bool = False
    """When true, create_training_client_from_state* skips the weights_info
    and create_model round-trips and sends a single LoadWeightsRequest with
    session addressing (session_id + model_seq_id) that creates the model
    server-side, configured from the checkpoint's owning model. Requires a
    server with load-weights-first support."""
    sample_use_retrieve_futures: bool = False
    """When true, each SamplingClient runs a single background task that polls
    ``/api/v1/retrieve_futures`` for its whole sampling session; a sample's
    result future waits for that poller to signal its request complete (and
    carries the completion metadata) before doing the normal payload fetch,
    instead of polling ``/api/v1/retrieve_future`` per request. Requires a
    server exposing the retrieve_futures endpoint."""
