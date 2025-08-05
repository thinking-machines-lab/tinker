# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal, Sequence

from .._models import BaseModel

__all__ = ["ComputeLogprobsResponse"]


class ComputeLogprobsResponse(BaseModel):
    logprobs: Sequence[Optional[float]]

    type: Literal["compute_logprobs"] = "compute_logprobs"
