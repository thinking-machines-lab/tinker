from typing_extensions import Literal, TypeAlias

from .._models import StrictBase

__all__ = ["CustomLossInput", "CustomLossInputRank"]

CustomLossInputRank: TypeAlias = Literal[0, 1, 2]
"""A custom loss input's rank: 0 is a scalar, 1 is ``[T]`` and 2 is ``[T, K]``."""


class CustomLossInput(StrictBase):
    """One input of a custom loss, with the rank every datum's value has."""

    name: str
    """``logprobs``, or the ``loss_fn_inputs`` key the loss reads."""

    rank: CustomLossInputRank
