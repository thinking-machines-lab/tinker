from typing import Union

from typing_extensions import TypeAlias

from .create_model_response import CreateModelResponse
from .forward_backward_output import ForwardBackwardOutput
from .load_weights_response import LoadWeightsResponse
from .optim_step_response import OptimStepResponse
from .request_failed_response import RequestFailedResponse
from .save_weights_for_sampler_response import SaveWeightsForSamplerResponse
from .save_weights_response import SaveWeightsResponse
from .try_again_response import TryAgainResponse
from .unload_model_response import UnloadModelResponse

__all__ = ["FutureRetrieveResponse"]

FutureRetrieveResponse: TypeAlias = Union[
    TryAgainResponse,
    ForwardBackwardOutput,
    OptimStepResponse,
    SaveWeightsResponse,
    LoadWeightsResponse,
    SaveWeightsForSamplerResponse,
    CreateModelResponse,
    UnloadModelResponse,
    RequestFailedResponse,
]
