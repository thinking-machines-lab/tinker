from typing import Union

from typing_extensions import Annotated, Literal, TypeAlias

from .._models import BaseModel
from .._utils import PropertyInfo

__all__ = [
    "SessionFailedRequestReason",
    "SessionFailedTrainingRunReason",
    "SessionFailedSamplingSessionReason",
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


SessionFailedReason: TypeAlias = Annotated[
    Union[
        SessionFailedRequestReason,
        SessionFailedTrainingRunReason,
        SessionFailedSamplingSessionReason,
    ],
    PropertyInfo(discriminator="type"),
]


class SessionFailed(BaseModel):
    """The session was failed by one of its requests, training runs or sampling sessions."""

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
    Union[ClientSessionFinishSuccess, ClientSessionFinishErrored, ClientSessionFinishInterrupted],
    PropertyInfo(discriminator="type"),
]


class ClientSessionFinished(BaseModel):
    """The session's client reported it was done."""

    type: Literal["client_finished"] = "client_finished"

    reason: ClientSessionFinishReason

    detail: str | None = None


class SessionActive(BaseModel):
    type: Literal["active"] = "active"


SessionStatus: TypeAlias = Annotated[
    Union[SessionFailed, SessionCancelled, ClientSessionFinished, SessionActive],
    PropertyInfo(discriminator="type"),
]
