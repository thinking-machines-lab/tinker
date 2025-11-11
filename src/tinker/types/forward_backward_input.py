from typing import Dict, List, Optional

from .datum import Datum
from .._models import StrictBase
from .loss_fn_type import LossFnType

__all__ = ["ForwardBackwardInput"]


class ForwardBackwardInput(StrictBase):
    data: List[Datum]
    """Array of input data for the forward/backward pass"""

    loss_fn: LossFnType
    """Fully qualified function path for the loss function"""

    loss_fn_config: Optional[Dict[str, float]] = None
    """Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)"""
