from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_id import ModelID

__all__ = ["OptimStepRequest", "AdamParams"]


class AdamParams(StrictBase):
    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""


class OptimStepRequest(StrictBase):
    adam_params: AdamParams

    model_id: ModelID

    seq_id: Optional[int] = None

    type: Literal["optim_step"] = "optim_step"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
