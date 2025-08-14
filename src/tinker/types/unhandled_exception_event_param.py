# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union, Optional
from datetime import datetime
from typing_extensions import Required, Annotated, TypedDict

from .._utils import PropertyInfo
from .severity import Severity
from .event_type import EventType

__all__ = ["UnhandledExceptionEventParam"]


class UnhandledExceptionEventParam(TypedDict, total=False):
    error_message: Required[str]

    error_type: Required[str]

    event: Required[EventType]
    """Telemetry event type"""

    event_id: Required[str]

    event_session_index: Required[int]

    severity: Required[Severity]
    """Log severity level"""

    timestamp: Required[Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]]

    traceback: Optional[str]
    """Optional Python traceback string"""
