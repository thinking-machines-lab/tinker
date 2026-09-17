"""Optimizer selection and per-step hyperparameters for training models."""

from pydantic import ConfigDict, Field
from typing_extensions import Annotated, Literal

from .._models import StrictBase


class AdamOptimizerConfig(StrictBase):
    type: Literal["adamw"] = "adamw"


class DimuonOptimizerConfig(StrictBase):
    """Dimuon identity retained by full-state resume."""

    type: Literal["dimuon"] = "dimuon"
    version: Literal[1] = 1


OptimizerConfig = Annotated[
    AdamOptimizerConfig | DimuonOptimizerConfig, Field(discriminator="type")
]


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


class DimuonParams(StrictBase):
    """Hyperparameters for one Dimuon optimizer step.

    The caller supplies schedules. Omitted settings use the defaults on each step;
    previous overrides do not carry forward. Matrix-sign iteration count and
    normalization epsilon are internal constants.
    """

    model_config = ConfigDict(allow_inf_nan=False)
    type: Literal["dimuon"] = "dimuon"
    """Optimizer family for this step."""

    learning_rate: float = Field(ge=0)
    """Learning rate applied to a direction with Frobenius norm sqrt(rank), with rank correction applied uniformly to all LoRA matrices."""

    grad_clip_norm: float = 0.0
    """Maximum global gradient norm. Nonpositive values disable gradient clipping."""

    beta1: float = Field(default=0.9, ge=0, lt=1)
    """Coefficient used for computing running averages of gradients."""

    damping: float = Field(default=1e-4, gt=0)
    """Positive damping that smooths the row and column preconditioner."""

    rank_lr_correction_exponent: float = 0.175
    """Exponent for learning-rate correction relative to rank 32. Zero disables rank correction."""


OptimParams = AdamParams | DimuonParams
