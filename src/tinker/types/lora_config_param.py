# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Optional

from typing_extensions import Required, TypedDict

__all__ = ["LoraConfigParam"]


class LoraConfigParam(TypedDict, total=False):
    rank: Required[int]
    """LoRA rank (dimension of low-rank matrices)"""

    seed: Optional[int]
    """Seed used for initialization of LoRA weights.

    Useful if you need deterministic or reproducible initialization of weights.
    """

    train_unembed: Optional[bool]
    """Whether to add lora to the unembedding layer"""

    train_mlp: Optional[bool]
    """Whether to add loras to the MLP layers (including MoE layers)"""

    train_attn: Optional[bool]
    """Whether to add loras to the attention layers"""
