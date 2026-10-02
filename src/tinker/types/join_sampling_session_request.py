from __future__ import annotations

from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["JoinSamplingSessionRequest"]


class JoinSamplingSessionRequest(StrictBase):
    sampling_session_id: str
    """The existing sampling session to join"""

    type: Literal["join_sampling_session"] = "join_sampling_session"
