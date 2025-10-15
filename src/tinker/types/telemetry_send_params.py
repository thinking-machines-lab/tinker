from __future__ import annotations

from typing import Iterable
from typing_extensions import Required, TypedDict

from .telemetry_event_param import TelemetryEventParam

__all__ = ["TelemetrySendParams"]


class TelemetrySendParams(TypedDict, total=False):
    events: Required[Iterable[TelemetryEventParam]]

    platform: Required[str]
    """Host platform name"""

    sdk_version: Required[str]
    """SDK version string"""

    session_id: Required[str]
