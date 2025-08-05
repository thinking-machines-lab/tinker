# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional, Sequence
from typing_extensions import Literal

from .._models import BaseModel
from .sampled_sequence import SampledSequence

__all__ = ["SampleResponse"]


class SampleResponse(BaseModel):
    sequences: Sequence[SampledSequence]

    type: Literal["sample"] = "sample"

    prompt_logprobs: Optional[List[Optional[float]]] = None
    """
    If prompt_logprobs was set to true in the request, logprobs are computed for
    every token in the prompt. The `prompt_logprobs` response contains a float32
    value for every token in the prompt.
    """
