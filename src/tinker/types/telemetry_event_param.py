# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union

from typing_extensions import TypeAlias

from .generic_event_param import GenericEventParam
from .session_end_event_param import SessionEndEventParam
from .session_start_event_param import SessionStartEventParam
from .unhandled_exception_event_param import UnhandledExceptionEventParam

__all__ = ["TelemetryEventParam"]

TelemetryEventParam: TypeAlias = Union[
    SessionStartEventParam, SessionEndEventParam, UnhandledExceptionEventParam, GenericEventParam
]
