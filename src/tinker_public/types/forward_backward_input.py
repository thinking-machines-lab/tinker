# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List

from .datum import Datum
from .._models import StrictBase
from .loss_fn_type import LossFnType

__all__ = ["ForwardBackwardInput"]


class ForwardBackwardInput(StrictBase):
    data: List[Datum]
    """Array of input data for the forward/backward pass"""

    loss_fn: LossFnType
    """Fully qualified function path for the loss function"""
