# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from .._models import StrictBase

__all__ = ["LoraConfig"]


class LoraConfig(StrictBase):
    rank: int
    """LoRA rank (dimension of low-rank matrices)"""

    seed: Optional[int] = None
    """Seed used for initialization of LoRA weights.

    Useful if you need deterministic or reproducible initialization of weights.
    """
