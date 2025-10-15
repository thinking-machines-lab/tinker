from typing import Dict, List

from .._models import BaseModel
from .loss_fn_output import LossFnOutput

__all__ = ["ForwardBackwardOutput"]


class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    """The type of the ForwardBackward output. Can be one of [...] TODO"""

    loss_fn_outputs: List[LossFnOutput]
    """Dictionary mapping field names to tensor data"""

    metrics: Dict[str, float]
    """Training metrics as key-value pairs"""
