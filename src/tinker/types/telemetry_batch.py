from typing import List

from .._models import BaseModel
from .telemetry_event import TelemetryEvent

__all__ = ["TelemetryBatch"]


class TelemetryBatch(BaseModel):
    events: List[TelemetryEvent]

    platform: str
    """Host platform name"""

    sdk_version: str
    """SDK version string"""

    session_id: str
