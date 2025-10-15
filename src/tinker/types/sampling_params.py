from typing import Optional, Sequence, Union

from .._models import BaseModel

__all__ = ["SamplingParams"]


class SamplingParams(BaseModel):
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate"""

    seed: Optional[int] = None
    """Random seed for reproducible generation"""

    stop: Union[str, Sequence[str], Sequence[int], None] = None
    """Stop sequences for generation"""

    temperature: float = 1
    """Sampling temperature"""

    top_k: int = -1
    """Top-k sampling parameter (-1 for no limit)"""

    top_p: float = 1
    """Nucleus sampling probability"""
