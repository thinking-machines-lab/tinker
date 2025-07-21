# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .model_input_param import ModelInputParam
from .sampling_params_param import SamplingParamsParam

__all__ = ["SamplingAsampleParams"]


class SamplingAsampleParams(TypedDict, total=False):
    num_samples: int
    """Number of samples to generate"""

    prompt: Required[ModelInputParam]

    sampling_params: Required[SamplingParamsParam]

    base_model: str
    """Optional base model name to sample from.

    Is inferred from model_path, if provided. If sampling against a base model, this
    is required.
    """

    model_path: str
    """Optional tinker:// path to your model weights or LoRA weights.

    If not provided, samples against the base model.
    """

    prompt_logprobs: bool
    """If set to `true`, computes and returns logprobs on the prompt tokens.

    Defaults to false.
    """

    type: Literal["sample"]
