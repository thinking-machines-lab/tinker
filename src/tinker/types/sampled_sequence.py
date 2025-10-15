from typing import List, Optional

from .._models import BaseModel
from .stop_reason import StopReason

__all__ = ["SampledSequence"]


class SampledSequence(BaseModel):
    stop_reason: StopReason
    """Reason why sampling stopped"""

    tokens: List[int]
    """List of generated token IDs"""

    logprobs: Optional[List[float]] = None
    """Log probabilities for each token (optional)"""
