from typing import List

from .._models import StrictBase
from .telemetry_event import TelemetryEvent

__all__ = ["TelemetrySendRequest"]


class TelemetrySendRequest(StrictBase):
    events: List[TelemetryEvent]

    platform: str
    """Host platform name"""

    sdk_version: str
    """SDK version string"""

    session_id: str
