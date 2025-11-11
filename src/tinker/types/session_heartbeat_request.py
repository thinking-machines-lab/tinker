from __future__ import annotations

from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["SessionHeartbeatRequest"]


class SessionHeartbeatRequest(StrictBase):
    session_id: str

    type: Literal["session_heartbeat"] = "session_heartbeat"
