from typing import Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_input import ModelInput
from .sampling_params import SamplingParams

__all__ = ["SampleRequest"]


class SampleRequest(StrictBase):
    num_samples: int = 1
    """Number of samples to generate"""

    prompt: ModelInput

    sampling_params: SamplingParams

    base_model: Optional[str] = None
    """Optional base model name to sample from.

    Is inferred from model_path, if provided. If sampling against a base model, this
    is required.
    """

    model_path: Optional[str] = None
    """Optional tinker:// path to your model weights or LoRA weights.

    If not provided, samples against the base model.
    """

    sampling_session_id: Optional[str] = None
    """Optional sampling session ID to use instead of model_path/base_model.

    If provided along with seq_id, the model configuration will be loaded from the
    sampling session. This is useful for multi-turn conversations.
    """

    seq_id: Optional[int] = None
    """Sequence ID within the sampling session.

    Required when sampling_session_id is provided. Used to generate deterministic
    request IDs for the sampling request.
    """

    prompt_logprobs: Optional[bool] = None
    """If set to `true`, computes and returns logprobs on the prompt tokens.

    Defaults to false.
    """

    topk_prompt_logprobs: int = 0
    """If set to a positive integer, returns the top-k logprobs for each prompt token."""

    type: Literal["sample"] = "sample"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
