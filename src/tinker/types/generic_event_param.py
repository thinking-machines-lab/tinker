# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from datetime import datetime
from typing import Dict, Union

from typing_extensions import Annotated, Required, TypedDict

from .._utils import PropertyInfo
from .event_type import EventType
from .severity import Severity

__all__ = ["GenericEventParam"]


class GenericEventParam(TypedDict, total=False):
    event: Required[EventType]
    """Telemetry event type"""

    event_id: Required[str]

    event_name: Required[str]
    """Low-cardinality event name"""

    event_session_index: Required[int]

    severity: Required[Severity]
    """Log severity level"""

    timestamp: Required[Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]]

    event_data: Dict[str, object]
    """Arbitrary structured JSON payload"""
