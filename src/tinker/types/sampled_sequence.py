from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional

import numpy as np

from .stop_reason import StopReason

__all__ = ["SampledSequence"]


@dataclass(frozen=True)
class SampledSequence:
    """A single sampled sequence from the model.

    Provides two ways to access token data:

    - **Numpy arrays** (``tokens_np``, ``logprobs_np``): As numpy arrays
      without format conversion.
    - **Python lists** (``tokens``, ``logprobs``): Standard Python lists,
      converted lazily on first access.
    """

    stop_reason: StopReason
    """Reason why sampling stopped."""

    tokens_np: Optional[np.ndarray] = field(default=None, repr=False)
    """Generated token IDs as a 1-D int32 numpy array, shape ``(num_tokens,)``."""

    logprobs_np: Optional[np.ndarray] = field(default=None, repr=False)
    """Log probabilities for each generated token as a 1-D float32 numpy array,
    shape ``(num_tokens,)``. None if logprobs were not requested."""

    # Private storage for list-based construction path.
    _tokens_list: Optional[List[int]] = field(default=None, repr=False)
    _logprobs_list: Optional[List[float]] = field(default=None, repr=False)

    @cached_property
    def tokens(self) -> List[int]:
        """Generated token IDs as a Python list.

        Converted from ``tokens_np`` on first access (cached afterwards).
        """
        if self._tokens_list is not None:
            return self._tokens_list
        if self.tokens_np is not None:
            return self.tokens_np.tolist()
        return []

    @cached_property
    def logprobs(self) -> Optional[List[float]]:
        """Log probabilities for each generated token (optional).

        None if logprobs were not requested. Converted from ``logprobs_np``
        on first access (cached afterwards).
        """
        if self._logprobs_list is not None:
            return self._logprobs_list
        if self.logprobs_np is not None:
            return self.logprobs_np.tolist()
        return None
