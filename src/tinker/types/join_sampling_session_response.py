from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["JoinSamplingSessionResponse"]


class JoinSamplingSessionResponse(BaseModel):
    type: Literal["join_sampling_session"] = "join_sampling_session"

    client_counter: int
    """Server-allocated id, unique among the sampling session's clients (the original client is 0)"""
