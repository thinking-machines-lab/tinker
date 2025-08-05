# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .model_id import ModelID

__all__ = ["TrainingOptimStepParams", "AdamParams"]


class TrainingOptimStepParams(TypedDict, total=False):
    adam_params: Required[AdamParams]

    model_id: Required[ModelID]

    type: Literal["optim_step"] = "optim_step"


class AdamParams(TypedDict, total=False):
    learning_rate: Required[float]
    """Learning rate for the optimizer"""

    beta1: float
    """Coefficient used for computing running averages of gradient"""

    beta2: float
    """Coefficient used for computing running averages of gradient square"""

    eps: float
    """Term added to the denominator to improve numerical stability"""
