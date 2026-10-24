from typing import Union

from pydantic import Field
from typing_extensions import Literal, TypeAlias

from .._models import StrictBase

__all__ = [
    "ClientSessionFinishReasonType",
    "ActiveSessionStatusQuery",
    "CancelledSessionStatusQuery",
    "FailedSessionStatusQuery",
    "ClientFinishedSessionStatusQuery",
    "SessionStatusQuery",
]

ClientSessionFinishReasonType: TypeAlias = Literal["success", "errored", "interrupted"]


class ActiveSessionStatusQuery(StrictBase):
    """Sessions still running."""

    type: Literal["active"] = "active"


class CancelledSessionStatusQuery(StrictBase):
    """Sessions cancelled from the console."""

    type: Literal["cancelled"] = "cancelled"


class FailedSessionStatusQuery(StrictBase):
    """Sessions failed by a request, training run or sampling session."""

    type: Literal["failed"] = "failed"


class ClientFinishedSessionStatusQuery(StrictBase):
    """Sessions whose client reported it was done."""

    type: Literal["client_finished"] = "client_finished"

    reasons: list[ClientSessionFinishReasonType] | None = Field(default=None, min_length=1)
    """Match any listed client-reported finish reason; None matches every reason"""


SessionStatusQuery: TypeAlias = Union[
    ActiveSessionStatusQuery,
    CancelledSessionStatusQuery,
    FailedSessionStatusQuery,
    ClientFinishedSessionStatusQuery,
]
