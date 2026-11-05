"""Backward compatibility: `ServiceClient.create_*training_client*` returned the
`TrainingClient` itself before they returned a future of it.

Everything in this module is temporary and goes away with that compatibility window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import deprecated

from tinker import types
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.public_interfaces.backcompat_client_future import BackcompatClientFuture
from tinker.lib.public_interfaces.backcompat_sampling_client_future import (
    BackcompatSamplingClientFuture,
)
from tinker.lib.public_interfaces.sampling_client import SamplingClient
from tinker.lib.public_interfaces.training_client import TrainingClient
from tinker.lib.sdk_method import with_sdk_method_telemetry
from tinker.lib.telemetry import Telemetry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@with_sdk_method_telemetry
class BackcompatTrainingClientFuture(BackcompatClientFuture[TrainingClient], TrainingClient):
    """A temporary wrapper kept for backwards compatibility with the synchronous API.

    The `TrainingClient`-creation methods return a future of the client. Prefer to `await` it in
    an async context, or call `.result()` in a sync context, to retrieve the `TrainingClient`
    and use that instead:

    ```python
    # Sync
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B").result()

    # Async
    training_client = await service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")

    fwd_bwd_future = training_client.forward_backward(data, loss_fn="cross_entropy")
    ```

    For now the wrapper also accepts `TrainingClient` calls directly, so code that treats the
    return value as the client keeps working; that usage is deprecated:

    ```python
    # Deprecated: using the future as the client
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
    fwd_bwd_future = training_client.forward_backward(data, loss_fn="cross_entropy")
    ```

    This wrapper will be removed in a subsequent version, after which you must `await` the
    future or call `.result()` on it.
    """

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def get_console_url(self, *args: Any, **kwargs: Any) -> str:
        return self._awaited_client().get_console_url(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def forward(self, *args: Any, **kwargs: Any) -> APIFuture[types.ForwardBackwardOutput]:
        return self._awaited_client().forward(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def forward_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await (await self._awaited_client_async()).forward_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def forward_backward(self, *args: Any, **kwargs: Any) -> APIFuture[types.ForwardBackwardOutput]:
        return self._awaited_client().forward_backward(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def forward_backward_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await (await self._awaited_client_async()).forward_backward_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def forward_backward_custom(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self._awaited_client().forward_backward_custom(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def forward_backward_custom_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await (await self._awaited_client_async()).forward_backward_custom_async(
            *args, **kwargs
        )

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def optim_step(self, *args: Any, **kwargs: Any) -> APIFuture[types.OptimStepResponse]:
        return self._awaited_client().optim_step(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def optim_step_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.OptimStepResponse]:
        return await (await self._awaited_client_async()).optim_step_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def save_state(self, *args: Any, **kwargs: Any) -> APIFuture[types.SaveWeightsResponse]:
        return self._awaited_client().save_state(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def save_state_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.SaveWeightsResponse]:
        return await (await self._awaited_client_async()).save_state_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def load_state(self, *args: Any, **kwargs: Any) -> APIFuture[types.LoadWeightsResponse]:
        return self._awaited_client().load_state(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def load_state_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.LoadWeightsResponse]:
        return await (await self._awaited_client_async()).load_state_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def load_state_with_optimizer(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.LoadWeightsResponse]:
        return self._awaited_client().load_state_with_optimizer(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def load_state_with_optimizer_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.LoadWeightsResponse]:
        return await (await self._awaited_client_async()).load_state_with_optimizer_async(
            *args, **kwargs
        )

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def save_weights_for_sampler(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return self._awaited_client().save_weights_for_sampler(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def save_weights_for_sampler_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return await (await self._awaited_client_async()).save_weights_for_sampler_async(
            *args, **kwargs
        )

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def save_weights_external(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.SaveWeightsExternalResponse]:
        return self._awaited_client().save_weights_external(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def save_weights_external_async(
        self, *args: Any, **kwargs: Any
    ) -> APIFuture[types.SaveWeightsExternalResponse]:
        return await (await self._awaited_client_async()).save_weights_external_async(
            *args, **kwargs
        )

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def get_info(self, *args: Any, **kwargs: Any) -> types.GetInfoResponse:
        return self._awaited_client().get_info(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def get_info_async(self, *args: Any, **kwargs: Any) -> types.GetInfoResponse:
        return await (await self._awaited_client_async()).get_info_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def get_tokenizer(self, *args: Any, **kwargs: Any) -> PreTrainedTokenizer:
        return self._awaited_client().get_tokenizer(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def create_sampling_client(self, *args: Any, **kwargs: Any) -> BackcompatSamplingClientFuture:
        return self._awaited_client().create_sampling_client(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def create_sampling_client_async(self, *args: Any, **kwargs: Any) -> SamplingClient:
        return await (await self._awaited_client_async()).create_sampling_client_async(
            *args, **kwargs
        )

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def save_weights_and_get_sampling_client(
        self, *args: Any, **kwargs: Any
    ) -> BackcompatSamplingClientFuture:
        return self._awaited_client().save_weights_and_get_sampling_client(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    async def save_weights_and_get_sampling_client_async(
        self, *args: Any, **kwargs: Any
    ) -> SamplingClient:
        return await (
            await self._awaited_client_async()
        ).save_weights_and_get_sampling_client_async(*args, **kwargs)

    @deprecated(
        "Using the future as the TrainingClient is deprecated; call .result() on it, or await "
        "it, to get the TrainingClient first",
        category=FutureWarning,
    )
    def get_telemetry(self, *args: Any, **kwargs: Any) -> Telemetry | None:
        return self._awaited_client().get_telemetry(*args, **kwargs)
