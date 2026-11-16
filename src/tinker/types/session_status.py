from typing import Union

from typing_extensions import Annotated, Literal, TypeAlias

from .._models import BaseModel, unrecognized_fallback
from .._utils import PropertyInfo
from .unrecognized_type import UnrecognizedType

__all__ = [
    "SessionFailedRequestReason",
    "SessionFailedTrainingRunReason",
    "SessionFailedSamplingSessionReason",
    "SessionFailedSessionReason",
    "SessionFailedReason",
    "SessionFailed",
    "SessionCancelled",
    "ClientSessionFinishSuccess",
    "ClientSessionFinishErrored",
    "ClientSessionFinishInterrupted",
    "ClientSessionFinishReason",
    "ClientSessionFinished",
    "SessionActive",
    "SessionStatus",
]


class SessionFailedRequestReason(BaseModel):
    type: Literal["request"] = "request"

    request_id: str

    request_type: str


class SessionFailedTrainingRunReason(BaseModel):
    type: Literal["training_run"] = "training_run"

    training_run_id: str


class SessionFailedSamplingSessionReason(BaseModel):
    type: Literal["sampling_session"] = "sampling_session"

    sampling_session_id: str


class SessionFailedSessionReason(BaseModel):
    """The session failed as a whole, not through one request, training run or sampling session."""

    type: Literal["session"] = "session"


SessionFailedReason: TypeAlias = Annotated[
    Union[
        SessionFailedRequestReason,
        SessionFailedTrainingRunReason,
        SessionFailedSamplingSessionReason,
        SessionFailedSessionReason,
        UnrecognizedType,
    ],
    PropertyInfo(discriminator="type"),
    unrecognized_fallback(UnrecognizedType),
]


class SessionFailed(BaseModel):
    """The session failed."""

    type: Literal["failed"] = "failed"

    category: Literal["user", "internal"]

    reason: SessionFailedReason

    message: str


class SessionCancelled(BaseModel):
    type: Literal["cancelled"] = "cancelled"

    detail: str | None = None


class ClientSessionFinishSuccess(BaseModel):
    type: Literal["success"] = "success"


class ClientSessionFinishErrored(BaseModel):
    type: Literal["errored"] = "errored"


class ClientSessionFinishInterrupted(BaseModel):
    type: Literal["interrupted"] = "interrupted"


ClientSessionFinishReason: TypeAlias = Annotated[
    Union[
        ClientSessionFinishSuccess,
        ClientSessionFinishErrored,
        ClientSessionFinishInterrupted,
        UnrecognizedType,
    ],
    PropertyInfo(discriminator="type"),
    unrecognized_fallback(UnrecognizedType),
]


class ClientSessionFinished(BaseModel):
    """The session's client reported it was done."""

    type: Literal["client_finished"] = "client_finished"

    reason: ClientSessionFinishReason

    detail: str | None = None


class SessionActive(BaseModel):
    type: Literal["active"] = "active"


SessionStatus: TypeAlias = Annotated[
    Union[SessionFailed, SessionCancelled, ClientSessionFinished, SessionActive, UnrecognizedType],
    PropertyInfo(discriminator="type"),
    unrecognized_fallback(UnrecognizedType),
]
