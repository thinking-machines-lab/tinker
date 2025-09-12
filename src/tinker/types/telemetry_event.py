# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union

from typing_extensions import TypeAlias

from .generic_event import GenericEvent
from .session_end_event import SessionEndEvent
from .session_start_event import SessionStartEvent
from .unhandled_exception_event import UnhandledExceptionEvent

__all__ = ["TelemetryEvent"]

TelemetryEvent: TypeAlias = Union[
    SessionStartEvent, SessionEndEvent, UnhandledExceptionEvent, GenericEvent
]
