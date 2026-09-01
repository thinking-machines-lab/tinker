from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated, Literal

from .._models import BaseModel
from .request_id import RequestID

__all__ = ["FutureCompletion", "FutureFinished", "FutureFailed"]


class FutureFinished(BaseModel):
    """A finished request reported by ``/api/v1/retrieve_futures``.

    Carries the uncompressed response-payload size (as retrieve_future's
    metadata-only response does) so the SDK can reserve the inflight-bytes budget
    before fetching the payload.
    """

    state: Literal["finished"] = "finished"
    request_id: RequestID
    response_payload_uncompressed_size: int | None = None


class FutureFailed(BaseModel):
    """A failed request reported by ``/api/v1/retrieve_futures`` (no payload)."""

    state: Literal["failed"] = "failed"
    request_id: RequestID


# Payload metadata exists only on the finished variant.
FutureCompletion = Annotated[Union[FutureFinished, FutureFailed], Field(discriminator="state")]
