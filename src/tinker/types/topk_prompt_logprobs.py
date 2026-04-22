from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TopkPromptLogprobs"]


@dataclass(frozen=True, slots=True)
class TopkPromptLogprobs:
    """Top-k most likely tokens at each prompt position, as dense numpy matrices.

    Both matrices have shape ``(prompt_length, k)`` where ``k`` is the number
    of top tokens requested. Empty positions are filled with sentinel values
    (``token_id=0``, ``logprob=-99999.0``).
    """

    token_ids: np.ndarray
    """int32 matrix of token IDs, shape ``(prompt_length, k)``."""

    logprobs: np.ndarray
    """float32 matrix of log probabilities, shape ``(prompt_length, k)``."""
