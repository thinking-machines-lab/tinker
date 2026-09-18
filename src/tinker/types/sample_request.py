from typing import Optional

from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from ._pydantic_types.tensor_data import TensorData as _TensorDataModel
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

    topk_sample_logprobs: int = 0
    """If set to a positive integer, returns the top-k logprobs for each sampled token
    (see ``SampledSequence.topk_logprobs``)."""

    target_prompt_logprobs: Optional[_TensorDataModel] = None
    """Token ids whose prompt logprobs to return, as an int64 tensor of shape
    ``[len(prompt) - 1, K]``: cell ``[i, j]`` is scored at prompt position
    ``i + 1`` (position 0 has no preceding context). A cell of ``-1`` requests
    nothing and gets no logprob. May be sparse CSR, in which case only the
    listed cells request anything. ``SampleResponse.target_prompt_logprobs``
    has the same shape and layout."""

    record_stability_info: Optional[bool] = None

    type: Literal["sample"] = "sample"

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
