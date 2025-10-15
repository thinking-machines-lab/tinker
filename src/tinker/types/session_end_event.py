from datetime import datetime

from .._models import BaseModel
from .severity import Severity
from .event_type import EventType

__all__ = ["SessionEndEvent"]


class SessionEndEvent(BaseModel):
    duration: str
    """ISO 8601 duration string"""

    event: EventType
    """Telemetry event type"""

    event_id: str

    event_session_index: int

    severity: Severity
    """Log severity level"""

    timestamp: datetime
