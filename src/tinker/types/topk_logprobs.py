from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TopkLogprobs"]

MASK_LOGPROB = -99999.0


@dataclass(frozen=True, slots=True)
class TopkLogprobs:
    """Top-k most likely tokens at each position, as dense numpy matrices.

    Both matrices have shape ``(length, k)`` where ``k`` is the number of top
    tokens requested. Empty positions are filled with sentinel values
    (``token_id=0``, ``logprob=-99999.0``).
    """

    token_ids: np.ndarray
    """int32 matrix of token IDs, shape ``(length, k)``."""

    logprobs: np.ndarray
    """float32 matrix of log probabilities, shape ``(length, k)``."""


def topk_to_lists(topk: TopkLogprobs) -> list[list[tuple[int, float]] | None]:
    """Convert TopkLogprobs matrices to Python list format.

    Each position becomes a list of up to k ``(token_id, logprob)`` tuples,
    or ``None`` for positions filled entirely with the sentinel.
    """
    n, k = topk.token_ids.shape
    if n == 0 or k == 0:
        return []

    tid_flat = topk.token_ids.ravel().tolist()
    lp_flat = topk.logprobs.ravel().tolist()
    all_tuples = list(zip(tid_flat, lp_flat, strict=True))

    mask_lp = MASK_LOGPROB
    result: list[list[tuple[int, float]] | None] = []
    for i in range(n):
        start = i * k
        if tid_flat[start] == 0 and lp_flat[start] == mask_lp:
            result.append(None)
        else:
            end = start + k
            while end > start and tid_flat[end - 1] == 0 and lp_flat[end - 1] == mask_lp:
                end -= 1
            result.append(all_tuples[start:end])
    return result
