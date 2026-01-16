from datetime import datetime
from typing import Optional

from .._models import BaseModel
from .event_type import EventType
from .severity import Severity

__all__ = ["UnhandledExceptionEvent"]


class UnhandledExceptionEvent(BaseModel):
    error_message: str

    error_type: str

    event: EventType
    """Telemetry event type"""

    event_id: str

    event_session_index: int

    severity: Severity
    """Log severity level"""

    timestamp: datetime

    traceback: Optional[str] = None
    """Optional Python traceback string"""
