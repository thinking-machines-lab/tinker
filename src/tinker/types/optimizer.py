"""Optimizer selection and per-step hyperparameters for training models."""

from pydantic import SerializeAsAny
from typing_extensions import Literal

from .._models import StrictBase


class OptimizerConfigBase(StrictBase, frozen=True, extra="allow"):
    """Optimizer family selected at model creation, identified by `type`; keeps family fields."""

    type: str


class AdamOptimizerConfig(OptimizerConfigBase, frozen=True, extra="forbid"):
    type: Literal["adamw"] = "adamw"


OptimizerConfig = AdamOptimizerConfig | SerializeAsAny[OptimizerConfigBase]


class AdamParams(StrictBase):
    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""

    weight_decay: float = 0.0
    """Weight decay for the optimizer. Uses decoupled weight decay."""

    grad_clip_norm: float = 0.0
    """Maximum global gradient norm. If the global gradient norm is greater than this value, it will be clipped to this value. 0.0 means no clipping."""


class OptimParamsBase(StrictBase, frozen=True, extra="allow"):
    """Per-step hyperparameters of a non-Adam optimizer family, identified by `type`."""

    type: str


OptimParams = AdamParams | SerializeAsAny[OptimParamsBase]
