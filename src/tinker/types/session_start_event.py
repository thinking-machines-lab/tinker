from datetime import datetime

from .._models import BaseModel
from .severity import Severity
from .event_type import EventType

__all__ = ["SessionStartEvent"]


class SessionStartEvent(BaseModel):
    event: EventType
    """Telemetry event type"""

    event_id: str

    event_session_index: int

    severity: Severity
    """Log severity level"""

    timestamp: datetime
