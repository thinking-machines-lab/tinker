from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["SessionHeartbeatResponse"]


class SessionHeartbeatResponse(BaseModel):
    type: Literal["session_heartbeat"] = "session_heartbeat"
