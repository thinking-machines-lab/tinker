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

    topk_prompt_logprobs: Optional[list[Optional[list[tuple[int, float]]]]] = None
    """
    If topk_prompt_logprobs was set to a positive integer k in the request,
    the top-k logprobs are computed for every token in the prompt. The
    `topk_prompt_logprobs` response contains, for every token in the prompt,
    a list of up to k (token_id, logprob) tuples.
    """
