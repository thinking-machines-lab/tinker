"""Backward compatibility: `ServiceClient.create_sampling_client` and
`TrainingClient.save_weights_and_get_sampling_client` returned the `SamplingClient` itself
before they returned a future of it.

Everything in this module is temporary and goes away with that compatibility window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import deprecated

from tinker import types
from tinker.lib.public_interfaces.api_future import APIFuture
from tinker.lib.public_interfaces.backcompat_client_future import BackcompatClientFuture
from tinker.lib.public_interfaces.sampling_client import SamplingClient, _SamplingClientPickleState
from tinker.lib.sdk_method import with_sdk_method_telemetry
from tinker.lib.telemetry import Telemetry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@with_sdk_method_telemetry
class BackcompatSamplingClientFuture(BackcompatClientFuture[SamplingClient], SamplingClient):
    """A temporary wrapper kept for backwards compatibility with the synchronous API.

    The `SamplingClient`-creation methods return a future of the client. Prefer to `await` it in
    an async context, or call `.result()` in a sync context, to retrieve the `SamplingClient`
    and use that instead:

    ```python
    # Sync
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B").result()

    # Async
    sampling_client = await service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")

    sample_future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    ```

    For now the wrapper also accepts `SamplingClient` calls directly, so code that treats the
    return value as the client keeps working; that usage is deprecated:

    ```python
    # Deprecated: using the future as the client
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
    sample_future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    ```

    This wrapper will be removed in a subsequent version, after which you must `await` the
    future or call `.result()` on it.
    """

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def sample(self, *args: Any, **kwargs: Any) -> APIFuture[types.SampleResponse]:
        return self._awaited_client().sample(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    async def sample_async(self, *args: Any, **kwargs: Any) -> types.SampleResponse:
        return await (await self._awaited_client_async()).sample_async(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def compute_logprobs(self, *args: Any, **kwargs: Any) -> APIFuture[list[float | None]]:
        return self._awaited_client().compute_logprobs(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    async def compute_logprobs_async(self, *args: Any, **kwargs: Any) -> list[float | None]:
        return await (await self._awaited_client_async()).compute_logprobs_async(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def get_tokenizer(self, *args: Any, **kwargs: Any) -> PreTrainedTokenizer:
        return self._awaited_client().get_tokenizer(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def get_base_model(self, *args: Any, **kwargs: Any) -> str:
        return self._awaited_client().get_base_model(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    async def get_base_model_async(self, *args: Any, **kwargs: Any) -> str:
        return await (await self._awaited_client_async()).get_base_model_async(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def get_telemetry(self, *args: Any, **kwargs: Any) -> Telemetry | None:
        return self._awaited_client().get_telemetry(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def get_sampler_handle(self, *args: Any, **kwargs: Any) -> str:
        return self._awaited_client().get_sampler_handle(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def on_queue_state_change(self, *args: Any, **kwargs: Any) -> None:
        return self._awaited_client().on_queue_state_change(*args, **kwargs)

    @deprecated(
        "Using the future as the SamplingClient is deprecated; call .result() on it, or await "
        "it, to get the SamplingClient first",
        category=FutureWarning,
    )
    def __reduce__(self) -> tuple[Any, tuple[_SamplingClientPickleState]]:
        return self._awaited_client().__reduce__()
