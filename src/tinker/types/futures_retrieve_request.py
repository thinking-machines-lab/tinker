from __future__ import annotations

from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["FuturesRetrieveRequest", "SamplingSessionFuturesTarget"]


class SamplingSessionFuturesTarget(StrictBase):
    """Poll target: one (possibly cloned) sampling session.

    ``cloned_sampler_id`` is ``seq_id // 1_000_000_000`` for cloned
    SamplingClients (0 for the original).
    """

    type: Literal["sampling_session"] = "sampling_session"
    sampling_session_id: str
    cloned_sampler_id: int = 0


class FuturesRetrieveRequest(StrictBase):
    target: SamplingSessionFuturesTarget
    prev_cursor: int = 0
    """Opaque cursor from the previous response; events at or before it are
    considered seen and get trimmed server-side. 0 starts from the beginning."""

    timeout: float | None = None
