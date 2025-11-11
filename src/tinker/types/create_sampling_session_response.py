from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["CreateSamplingSessionResponse"]


class CreateSamplingSessionResponse(BaseModel):
    type: Literal["create_sampling_session"] = "create_sampling_session"

    sampling_session_id: str
    """The generated sampling session ID"""
