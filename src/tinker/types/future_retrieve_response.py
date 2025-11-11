from typing import Union
from typing_extensions import TypeAlias

from .try_again_response import TryAgainResponse
from .optim_step_response import OptimStepResponse
from .create_model_response import CreateModelResponse
from .load_weights_response import LoadWeightsResponse
from .save_weights_response import SaveWeightsResponse
from .unload_model_response import UnloadModelResponse
from .forward_backward_output import ForwardBackwardOutput
from .save_weights_for_sampler_response import SaveWeightsForSamplerResponse
from .request_failed_response import RequestFailedResponse

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
