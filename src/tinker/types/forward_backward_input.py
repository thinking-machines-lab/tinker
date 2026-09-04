from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Union

from .datum import Datum
from .loss_fn_type import LossFnType

__all__ = ["ForwardBackwardInput"]


@dataclass(frozen=True)
class ForwardBackwardInput:
    data: List[Datum]
    """Array of input data for the forward/backward pass"""

    loss_fn: LossFnType
    """Fully qualified function path for the loss function"""

    loss_fn_config: Optional[Mapping[str, Union[float, str]]] = field(default=None)
    """Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)"""
