import typing as _t

from . import types
from ._types import NOT_GIVEN, Omit, NoneType, NotGiven, Transport, ProxiesTypes
from ._utils import file_from_path
from ._client import Timeout, Transport, RequestOptions
from ._models import BaseModel
from ._version import __title__, __version__
from ._response import APIResponse as APIResponse, AsyncAPIResponse as AsyncAPIResponse
from ._constants import DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, DEFAULT_CONNECTION_LIMITS
from ._exceptions import (
    APIError,
    TinkerError,
    ConflictError,
    NotFoundError,
    APIStatusError,
    RateLimitError,
    APITimeoutError,
    BadRequestError,
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    UnprocessableEntityError,
    APIResponseValidationError,
    RequestFailedError,
)
from ._utils._logs import setup_logging as _setup_logging
from .lib.public_interfaces import TrainingClient, ServiceClient, SamplingClient, APIFuture

# Import commonly used types for easier access
from .types import (
    AdamParams,
    Checkpoint,
    CheckpointType,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LoraConfig,
    ModelID,
    ModelInput,
    ModelInputChunk,
    OptimStepRequest,
    OptimStepResponse,
    ParsedCheckpointTinkerPath,
    SampledSequence,
    SampleRequest,
    SampleResponse,
    SamplingParams,
    StopReason,
    TensorData,
    TensorDtype,
    TrainingRun,
)

__all__ = [
    # Core clients
    "TrainingClient",
    "ServiceClient",
    "SamplingClient",
    "APIFuture",

    # Commonly used types
    "AdamParams",
    "Checkpoint",
    "CheckpointType",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LoraConfig",
    "ModelID",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepRequest",
    "OptimStepResponse",
    "ParsedCheckpointTinkerPath",
    "SampledSequence",
    "SampleRequest",
    "SampleResponse",
    "SamplingParams",
    "StopReason",
    "TensorData",
    "TensorDtype",
    "TrainingRun",

    # Client configuration
    "Timeout",
    "RequestOptions",

    # Exception types
    "TinkerError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "APIConnectionError",
    "APIResponseValidationError",
    "RequestFailedError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",

    # Keep types module for advanced use
    "types",

    # Version info
    "__version__",
    "__title__",
]

if not _t.TYPE_CHECKING:
    from ._utils._resources_proxy import resources as resources

_setup_logging()

# Update the __module__ attribute for exported symbols so that
# error messages point to this module instead of the module
# it was originally defined in, e.g.
# tinker._exceptions.NotFoundError -> tinker.NotFoundError
__locals = locals()
for __name in __all__:
    if not __name.startswith("__"):
        try:
            __locals[__name].__module__ = "tinker"
        except (TypeError, AttributeError):
            # Some of our exported symbols are builtins which we can't set attributes for.
            pass
