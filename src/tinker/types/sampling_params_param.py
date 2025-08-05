# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Union, Iterable, Optional
from typing_extensions import TypedDict

__all__ = ["SamplingParamsParam"]


class SamplingParamsParam(TypedDict, total=False):
    max_tokens: Optional[int]
    """Maximum number of tokens to generate"""

    seed: Optional[int]
    """Random seed for reproducible generation"""

    stop: Union[str, List[str], Iterable[int], None]
    """Stop sequences for generation"""

    temperature: float
    """Sampling temperature"""

    top_k: int
    """Top-k sampling parameter (-1 for no limit)"""

    top_p: float
    """Nucleus sampling probability"""
