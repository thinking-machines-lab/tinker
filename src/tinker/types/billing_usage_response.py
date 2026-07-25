from datetime import datetime
from typing import Union

from typing_extensions import Annotated, Literal, TypeAlias

from .._models import BaseModel, StrictBase
from .._utils import PropertyInfo

__all__ = [
    "TrainingBillingEvent",
    "SamplingPrefillBillingEvent",
    "SamplingSampleBillingEvent",
    "CheckpointBillingEvent",
    "StorageBillingEvent",
    "BillingEventInfo",
    "BillingUsageEvent",
    "BillingUsageSession",
    "BillingUsageResponse",
]


class TrainingBillingEvent(StrictBase):
    """Training tokens processed by forward/backward passes."""

    type: Literal["training"] = "training"

    token_count: int
    """Training token count for the bucket"""


class SamplingPrefillBillingEvent(StrictBase):
    """Prompt (prefill) tokens processed while sampling."""

    type: Literal["sampling_prefill"] = "sampling_prefill"

    cached: bool
    """True when the tokens were served from the prefill cache (billed at
    the discounted cached rate); False for full-rate prefill"""

    token_count: int
    """Prefill token count for the bucket"""


class SamplingSampleBillingEvent(StrictBase):
    """Tokens generated while sampling."""

    type: Literal["sampling_sample"] = "sampling_sample"

    token_count: int
    """Sampled token count for the bucket"""


class CheckpointBillingEvent(StrictBase):
    """Checkpoint operations (billed per checkpoint)."""

    type: Literal["checkpoint"] = "checkpoint"

    count: int
    """Number of checkpoints in the bucket"""


class StorageBillingEvent(StrictBase):
    """Checkpoint storage, billed in gigabyte-hours."""

    type: Literal["storage"] = "storage"

    gigabyte_hours: float
    """Storage quantity for the bucket"""


BillingEventInfo: TypeAlias = Annotated[
    Union[
        TrainingBillingEvent,
        SamplingPrefillBillingEvent,
        SamplingSampleBillingEvent,
        CheckpointBillingEvent,
        StorageBillingEvent,
    ],
    PropertyInfo(discriminator="type"),
]


class BillingUsageEvent(BaseModel):
    """One hourly bucket of billing usage: shared attribution dimensions on
    the envelope, with the usage-kind-specific payload in `event_info` (a
    tagged union discriminated on `.type`)."""

    bucket_start: datetime
    """Inclusive start of the UTC hour bucket"""

    bucket_end: datetime
    """Exclusive end of the UTC hour bucket"""

    base_model: str | None = None
    """Base model the usage was billed against (None when unknown, e.g.
    storage from before base-model stamping)"""

    user_id: str | None = None
    """Organization-user urn of the user that created the session the usage
    is attributed to"""

    user_name: str | None = None
    """Display name of `user_id`, as self-reported during onboarding"""

    session_id: str | None = None
    """The session the usage is attributed to: for sampling the session that
    issued the requests, for training the session that created the training
    run. None for events without a session (storage, usage from before
    session tagging). Session user_metadata is available in
    `BillingUsageResponse.sessions` — keyed by this id."""

    project_id: str | None = None
    """Project this usage belongs to, resolved from the current project
    associated with the session identified by `session_id`"""

    event_info: BillingEventInfo
    """What kind of usage this is and its quantity; dispatch on
    `event_info.type`"""


class BillingUsageSession(BaseModel):
    """Session-level attributes tied to a session."""

    user_metadata: dict[str, str] | None = None
    """The session's customer-supplied user_metadata (as passed to
    CreateSessionRequest.user_metadata); None when the session has none."""


class BillingUsageResponse(BaseModel):
    data: list[BillingUsageEvent]
    """Hourly usage events, ordered by bucket then descending token count"""

    sessions: dict[str, BillingUsageSession] = {}
    """session_id -> that session's attributes, for every distinct session
    appearing in `data`"""
