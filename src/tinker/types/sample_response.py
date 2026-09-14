from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Sequence

import numpy as np

from .sampled_sequence import SampledSequence
from .topk_logprobs import TopkLogprobs, topk_to_lists

__all__ = ["SampleResponse"]


@dataclass(frozen=True)
class SampleResponse:
    """Response from a sampling request.

    Contains generated sequences and optional prompt-level log probabilities.
    Numpy fields provide direct array access without format conversion.
    The corresponding Python-list properties convert lazily on first access.
    """

    sequences: Sequence[SampledSequence]
    """Generated sequences. Each contains token IDs, optional logprobs, and stop reason."""

    prompt_logprobs_np: Optional[np.ndarray] = field(default=None, repr=False)
    """Per-token log probabilities for the prompt as a 1-D float32 numpy array,
    shape ``(prompt_length,)``. ``NaN`` at positions where logprobs were not
    computed (e.g. the first prompt token).
    None if prompt logprobs were not requested."""

    topk_prompt_logprobs_np: Optional[TopkLogprobs] = field(default=None, repr=False)
    """Top-k prompt logprobs as a pair of dense matrices
    (see ``TopkLogprobs``).
    None if top-k was not requested."""

    prompt_cache_hit_tokens: int = 0
    """Number of prompt tokens billed as prefix-cache hits.

    Counted on the prompt itself: for ``num_samples > 1`` the prompt is
    shared, so this is not multiplied across samples. Prefill on the shared prompts
    samples (the remaining ``num_samples - 1``) is billed as cache hits."""

    # Private storage for list-based construction path.
    _prompt_logprobs_list: Optional[List[Optional[float]]] = field(default=None, repr=False)
    _topk_prompt_logprobs_list: Optional[list[Optional[list[tuple[int, float]]]]] = field(
        default=None, repr=False
    )

    @cached_property
    def prompt_logprobs(self) -> Optional[List[Optional[float]]]:
        """Per-token log probabilities for the prompt as a Python list.

        If prompt_logprobs was set to true in the request, logprobs are
        computed for every token in the prompt. Each entry is a float, or
        ``None`` for positions where logprobs were not computed (e.g. the
        first prompt token). Returns ``None`` if prompt logprobs were not
        requested.

        Converted from ``prompt_logprobs_np`` on first access (cached afterwards).
        """
        if self._prompt_logprobs_list is not None:
            return self._prompt_logprobs_list
        if self.prompt_logprobs_np is not None:
            result: list[float | None] = self.prompt_logprobs_np.tolist()
            for i in np.flatnonzero(np.isnan(self.prompt_logprobs_np)):
                result[i] = None
            return result
        return None

    @cached_property
    def topk_prompt_logprobs(self) -> Optional[list[Optional[list[tuple[int, float]]]]]:
        """Top-k prompt logprobs as nested Python lists.

        If topk_prompt_logprobs was set to a positive integer k in the request,
        the top-k logprobs are computed for every token in the prompt.
        For each prompt position: a list of up to k ``(token_id, logprob)``
        tuples, or ``None`` for positions where logprobs were not computed.
        Returns ``None`` if top-k was not requested.

        Converted from ``topk_prompt_logprobs_np`` on first access (cached afterwards).
        """
        if self._topk_prompt_logprobs_list is not None:
            return self._topk_prompt_logprobs_list
        if self.topk_prompt_logprobs_np is not None:
            return topk_to_lists(self.topk_prompt_logprobs_np)
        return None
