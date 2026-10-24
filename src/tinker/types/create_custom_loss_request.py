from __future__ import annotations

from typing import List

from typing_extensions import Literal

from .._models import StrictBase
from .custom_loss_graph import CustomLossGraph
from .custom_loss_input import CustomLossInput

__all__ = ["CreateCustomLossRequest"]


class CreateCustomLossRequest(StrictBase):
    session_id: str

    custom_loss_seq_id: int

    name: str

    inputs: List[CustomLossInput]
    """The graph's inputs in order, ``logprobs`` first, with their declared ranks."""

    graph: CustomLossGraph

    type: Literal["create_custom_loss"] = "create_custom_loss"
