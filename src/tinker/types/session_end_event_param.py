from __future__ import annotations

from typing import Union
from datetime import datetime
from typing_extensions import Required, Annotated, TypedDict

from .._utils import PropertyInfo
from .severity import Severity
from .event_type import EventType

__all__ = ["SessionEndEventParam"]


class SessionEndEventParam(TypedDict, total=False):
    duration: Required[str]
    """ISO 8601 duration string"""

    event: Required[EventType]
    """Telemetry event type"""

    event_id: Required[str]

    event_session_index: Required[int]

    severity: Required[Severity]
    """Log severity level"""

    timestamp: Required[Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]]
