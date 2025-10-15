from datetime import datetime
from typing import Dict

from .._models import BaseModel
from .event_type import EventType
from .severity import Severity

__all__ = ["GenericEvent"]


class GenericEvent(BaseModel):
    event: EventType
    """Telemetry event type"""

    event_id: str

    event_name: str
    """Low-cardinality event name"""

    event_session_index: int

    severity: Severity
    """Log severity level"""

    timestamp: datetime

    event_data: Dict[str, object] = {}
    """Arbitrary structured JSON payload"""
