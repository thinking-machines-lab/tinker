"""TrainingClient for Tinker API."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Tuple

from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import (
    _APIFuture,
    _CombinedAPIFuture,
)
from ..chunked_fwdbwd_helpers import combine_fwd_bwd_output_results
from ..queue_state_logger import QueueStateLogger
from ..retry_handler import RetryConfig
from ..sync_only import sync_only
from .sampling_client import SamplingClient, _load_tokenizer_from_model_info

try:
    import torch
except ImportError:
    torch = None


if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)


# FwdBwdChunkSize
MAX_CHUNK_LEN = 1024
MAX_CHUNK_BYTES_COUNT = 5000000
MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."

# Type alias for custom loss functions.
# Args: (data: List[Datum], model_outputs: List[Any]) -> (loss: Any, metrics: Dict[str, float])
CustomLossFnV1 = Callable[[List[types.Datum], List[Any]], Tuple[Any, Dict[str, float]]]

_SUPPORTED_CUSTOM_BACKEND_LOSS_FNS = frozenset({"cross_entropy"})
_CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE: dict[Literal["logprobs"], types.LossFnType] = {
    "logprobs": "cross_entropy",
}


class TrainingClient(TelemetryProvider):
    """Client for training ML models with forward/backward passes and optimization.

    The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
    You typically get one by calling `service_client.create_lora_training_client()`.
    Key methods:
    - forward_backward() - compute gradients for training
    - optim_step() - update model parameters with Adam optimizer
    - save_weights_and_get_sampling_client() - export trained model for inference

    Args:
    - `holder`: Internal client managing HTTP connections and async operations
    - `model_id`: Unique identifier for the model to train. Required for training operations.

    Example:
    ```python
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
    fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
    fwdbwd_result = fwdbwd_future.result()  # Wait for gradients
    optim_result = optim_future.result()    # Wait for parameter update
    sampling_client = training_client.save_weights_and_get_sampling_client("my-model")
    ```
    """

    def __init__(self, holder: InternalClientHolder, model_seq_id: int, model_id: types.ModelID):
        self.holder = holder
        self.model_id = model_id

        self._training_client_id: int = model_seq_id

        self._request_id_lock: threading.Lock = threading.Lock()
        self._request_id_counter: int = 0

        self._turn_counter: int = 0
        # Maps turn id → list of events to wake when _turn_counter reaches that id.
        self._turn_waiters: dict[int, list[asyncio.Event]] = {}

        self._queue_state_logger = QueueStateLogger(str(model_id), "Training")

    # Reserves a request id for a request. Requests are to be executed in the order of request ids.
    def _get_request_id(self) -> int:
        with self._request_id_lock:
            request_id = self._request_id_counter
            self._request_id_counter += 1
            return request_id

    # Waits for the turn to be executed.
    # Counter is advanced on entry (before yield), not on exit.
    #
    # With a single turn (max_turn not provided, min == max), this is the
    # classic sequential turn: wait until _turn_counter == min_turn.
    #
    # With a range (min_turn < max_turn), multiple callers sharing the
    # same [min, max] can proceed as soon as _turn_counter >= min_turn.
    # Each caller increments the counter on entry, so the counter
    # advances past max_turn before any subsequent caller can proceed.
    # This is used for fwdbwd chunks: all chunks in one forward_backward()
    # call share the same range and fire concurrently once the range is
    # reached.  To gate this off, just don't pass max_turn and chunks
    # will take turns sequentially.
    @asynccontextmanager
    async def _take_turn(self, min_turn: int, max_turn: int | None = None):
        if max_turn is None:
            max_turn = min_turn

        assert min_turn <= max_turn

        # Wait until _turn_counter reaches at least min_turn
        if self._turn_counter < min_turn:
            event = asyncio.Event()
            self._turn_waiters.setdefault(min_turn, []).append(event)
            await event.wait()

        assert self._turn_counter >= min_turn
        assert self._turn_counter <= max_turn, (
            f"Turn counter {self._turn_counter} overshot max {max_turn}"
        )

        try:
            yield
        finally:
            self._turn_counter += 1
            for event in self._turn_waiters.pop(self._turn_counter, []):
                event.set()

    def _guaranteed_model_id(self) -> types.ModelID:
        assert self.model_id is not None, MODEL_ID_NOT_SET_ERROR
        return self.model_id

    def _estimate_bytes_count(self, datum: types.Datum) -> int:
        return self.holder.estimate_bytes_count_in_model_input(datum.model_input) + sum(
            len(value.data) * 10 for _, value in datum.loss_fn_inputs.items()
        )

    def _chunked_requests_generator(
        self, data: List[types.Datum]
    ) -> Generator[List[types.Datum], None, None]:
        current_chunk: List[types.Datum] = []
        current_chunk_bytes_count = 0

        for datum in data:
            estimated_bytes_count = self._estimate_bytes_count(datum)
            if (
                len(current_chunk) > 0
                and current_chunk_bytes_count + estimated_bytes_count > MAX_CHUNK_BYTES_COUNT
            ) or (len(current_chunk) == MAX_CHUNK_LEN):
                yield current_chunk
                current_chunk = []
                current_chunk_bytes_count = 0

            current_chunk.append(datum)
            current_chunk_bytes_count += estimated_bytes_count

        if len(current_chunk) > 0:
            yield current_chunk

    def _chunked_requests(self, data: List[types.Datum]) -> List[tuple[int, List[types.Datum]]]:
        return [(self._get_request_id(), chunk) for chunk in self._chunked_requests_generator(data)]

    def forward(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward pass without gradients.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Loss function type (e.g., "cross_entropy")
        - `loss_fn_config`: Optional configuration for the loss function

        Returns:
        - `APIFuture` containing the forward pass outputs and loss

        Example:
        ```python
        data = [types.Datum(
            model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
            loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
        )]
        future = training_client.forward(data, "cross_entropy")
        result = await future
        print(f"Loss: {result.metrics['loss:sum']}")
        ```
        """
        cfg = self.holder._client_config
        if cfg.fwd_via_fwdbwd and cfg.proto_write_fwdbwd:
            # Route through /forward_backward. Falls through to the legacy /forward
            # JSON path when either flag is off.
            return self._run_fwd_bwd(data, loss_fn, loss_fn_config, forward_only=True)
        requests = self._chunked_requests(data)

        @capture_exceptions(fatal=True)
        async def _forward_async():
            start_time = time.time()

            async def _send_request(request_id: int, data: List[types.Datum]):
                request = types.ForwardRequest(
                    forward_input=types.ForwardBackwardInput(
                        data=data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
                    ),
                    model_id=self._guaranteed_model_id(),
                    seq_id=request_id + 1,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.training.forward(
                        request=request,
                    )

            futures = []
            for request_id, data in requests:
                async with self._take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(
                        _send_request, request_id, data
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="Forward",
                    queue_state_observer=self._queue_state_logger,
                )
                futures.append(api_future)
            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_async())

    async def forward_async(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward."""
        return self.forward(data, loss_fn, loss_fn_config)

    def forward_backward(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward pass and backward pass to calculate gradients.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Loss function type (e.g., "cross_entropy")
        - `loss_fn_config`: Optional configuration for the loss function

        Returns:
        - `APIFuture` containing the forward/backward outputs, loss, and gradients

        Example:
        ```python
        data = [types.Datum(
            model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
            loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
        )]

        # Compute gradients
        fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

        # Update parameters
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=1e-4)
        )

        fwdbwd_result = await fwdbwd_future
        print(f"Loss: {fwdbwd_result.metrics['loss:sum']}")
        ```
        """
        return self._run_fwd_bwd(data, loss_fn, loss_fn_config, forward_only=False)

    def _run_fwd_bwd(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None,
        *,
        forward_only: bool,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Shared implementation for /forward_backward submissions.

        Drives chunking, optional parallel submit. ``forward_only=True``
        is only allowed when proto_write_fwdbwd flag in client config is true.
        """
        assert not forward_only or self.holder._client_config.proto_write_fwdbwd, (
            "forward_only is only allowed when proto_write_fwdbwd is true"
        )

        requests = self._chunked_requests(data)
        if not requests:
            raise ValueError("No data provided")

        parallel = self.holder._client_config.parallel_fwdbwd_chunks
        # When parallel, all chunks share [min, max] range and fire concurrently.
        # When serial, each chunk takes its own turn sequentially.
        min_rid = requests[0][0]
        max_rid = requests[-1][0] if parallel else None
        request_type = "Forward" if forward_only else "ForwardBackward"

        @capture_exceptions(fatal=True)
        async def _run_async():
            start_time = time.time()

            async def _send_request(request_id: int, data: List[types.Datum]):
                request = types.ForwardBackwardRequest(
                    forward_backward_input=types.ForwardBackwardInput(
                        data=data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
                    ),
                    model_id=self._guaranteed_model_id(),
                    seq_id=request_id + 1,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.training.forward_backward(
                        request=request, forward_only=forward_only
                    )

            async def _submit_chunk(
                request_id: int, data: List[types.Datum]
            ) -> APIFuture[types.ForwardBackwardOutput]:
                turn_min = min_rid if parallel else request_id
                turn_max = max_rid if parallel else None
                async with self._take_turn(turn_min, turn_max):
                    untyped_future = await self.holder.execute_with_retries(
                        _send_request, request_id, data
                    )
                    return _APIFuture(
                        types.ForwardBackwardOutput,
                        self.holder,
                        untyped_future,
                        request_start_time=start_time,
                        request_type=request_type,
                        queue_state_observer=self._queue_state_logger,
                    )

            if parallel and len(requests) > 1:
                # Send all chunks in parallel, but submit the first chunk
                # last.  The server won't process later chunks until the
                # first one arrives (seq_id ordering), so by the time chunk
                # 1 lands the rest are already queued and the server can
                # pick the whole batch together.
                rest_futures = list(
                    await asyncio.gather(*[_submit_chunk(rid, d) for rid, d in requests[1:]])
                )
                first_rid, first_data = requests[0]
                first_future = await _submit_chunk(first_rid, first_data)
                futures = [first_future] + rest_futures
            else:
                # gather is safe even when serial — _take_turn orders execution.
                futures = list(
                    await asyncio.gather(
                        *[_submit_chunk(request_id, data) for request_id, data in requests]
                    )
                )

            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_run_async())

    async def forward_backward_async(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward_backward."""
        return self.forward_backward(data, loss_fn, loss_fn_config)

    @sync_only
    def forward_backward_custom(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV1,
        *,
        loss_type_input: Literal["logprobs"] = "logprobs",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward/backward with a custom loss function.

        Allows you to define custom loss functions that operate on log probabilities.
        The custom function receives logprobs and computes loss and gradients.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Custom loss function that takes (data, logprobs) and returns (loss, metrics)
        - `loss_type_input`: Input space for `loss_fn`. Currently the only supported value is `"logprobs"`.

        Returns:
        - `APIFuture` containing the forward/backward outputs with custom loss

        Example:
        ```python
        def custom_loss(data, logprobs_list):
            # Custom loss computation
            loss = torch.mean(torch.stack([torch.mean(lp) for lp in logprobs_list]))
            metrics = {"custom_metric": loss.item()}
            return loss, metrics

        future = training_client.forward_backward_custom(data, custom_loss)
        result = future.result()
        print(f"Metrics: {result.metrics}")
        ```
        """
        return self.holder.run_coroutine_threadsafe(
            self.forward_backward_custom_async(
                data,
                loss_fn,
                loss_type_input=loss_type_input,
            )
        ).result()

    async def forward_backward_custom_async(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV1,
        *,
        loss_type_input: Literal["logprobs"] = "logprobs",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward_backward_custom."""
        if torch is None:
            raise ImportError("PyTorch is not installed. Cannot run custom forward_backward.")

        if loss_type_input not in _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE:
            supported = ", ".join(sorted(_CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE))
            raise ValueError(
                f"Unsupported loss_type_input={loss_type_input!r}. "
                f"Supported values are: {supported}"
            )

        surrogate_loss_fn = _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE[loss_type_input]

        forward_data = self._get_custom_loss_forward_data(data, surrogate_loss_fn)

        # First do a forward pass and get logprobs
        forward_future = await self.forward_async(
            forward_data,
            surrogate_loss_fn,
            None,
        )
        forward_result = await forward_future.result_async()
        logprobs_list = []
        for out in forward_result.loss_fn_outputs:
            logprob = torch.tensor(out["logprobs"].data)
            if out["logprobs"].shape is not None:
                logprob = logprob.reshape(out["logprobs"].shape)
            logprob = logprob.clone().detach().requires_grad_(True)
            logprobs_list.append(logprob)

        # Now apply user-provided function
        loss, metrics = loss_fn(data, logprobs_list)
        loss.backward()
        grads = []
        for logprob in logprobs_list:
            if logprob.grad is None:
                raise ValueError("No gradient computed for logprob tensor")
            grads.append(logprob.grad)

        linear_loss_data = []
        for datum, grad in zip(data, grads, strict=True):
            loss_fn_inputs: Any = {
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                # Backend CE is L = sum(-logprobs * weights), so to backpropagate a
                # client-side custom loss C(logprobs) we must send weights = -dC/dlogprobs.
                "weights": -grad,
            }
            linear_loss_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

        # Do the backward pass with the gradients
        backward_future = await self.forward_backward_async(
            linear_loss_data,
            surrogate_loss_fn,
            None,
        )

        # We need to slightly modify the future to add the custom metrics, so we use _CombinedAPIFuture
        # to transform the future.
        def add_custom_metrics(
            results: List[types.ForwardBackwardOutput],
        ) -> types.ForwardBackwardOutput:
            result = results[0]  # Single result
            result.metrics.update(metrics)
            return result

        return _CombinedAPIFuture([backward_future], add_custom_metrics, self.holder)

    def _get_custom_loss_forward_data(
        self,
        data: List[types.Datum],
        surrogate_loss_fn: types.LossFnType,
    ) -> List[types.Datum]:
        assert surrogate_loss_fn in _SUPPORTED_CUSTOM_BACKEND_LOSS_FNS, (
            "forward_backward_custom_async should validate surrogate_loss_fn before "
            "_get_custom_loss_forward_data is called"
        )

        forward_data = []
        for datum in data:
            target_tokens = datum.loss_fn_inputs.get("target_tokens")
            if target_tokens is None:
                raise ValueError("target_tokens must be provided when using cross_entropy")

            unexpected_keys = sorted(set(datum.loss_fn_inputs) - {"target_tokens", "weights"})
            if unexpected_keys:
                raise ValueError(
                    "forward_backward_custom only supports loss_fn_inputs keys "
                    "{'target_tokens', 'weights'}; "
                    f"found unexpected keys: {unexpected_keys}"
                )

            if "weights" in datum.loss_fn_inputs:
                forward_data.append(datum)
                continue

            forward_loss_fn_inputs = dict(datum.loss_fn_inputs)
            forward_loss_fn_inputs["weights"] = types.TensorData(
                data=[0.0] * len(target_tokens.data),
                dtype="float32",
                shape=target_tokens.shape,
            )
            forward_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs=forward_loss_fn_inputs,
                )
            )

        return forward_data

    def optim_step(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        """Update model parameters using Adam optimizer.

        The Adam optimizer used by tinker is identical
        to [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        Note that unlike PyTorch, Tinker's default weight decay value is 0.0 (no weight decay).


        Args:
        - `adam_params`: Adam optimizer parameters (learning_rate, betas, eps, weight_decay)

        Returns:
        - `APIFuture` containing optimizer step response

        Example:
        ```python
        # First compute gradients
        fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

        # Then update parameters
        optim_future = training_client.optim_step(
            types.AdamParams(
                learning_rate=1e-4,
                weight_decay=0.01
            )
        )

        # Wait for both to complete
        fwdbwd_result = await fwdbwd_future
        optim_result = await optim_future
        ```
        """
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _optim_step_async():
            start_time = time.time()

            async def _send_request():
                request = types.OptimStepRequest(
                    adam_params=adam_params,
                    model_id=self._guaranteed_model_id(),
                    seq_id=request_id + 1,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.training.optim_step(
                        request=request,
                    )

            async with self._take_turn(request_id):
                untyped_future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.OptimStepResponse,
                self.holder,
                untyped_future,
                request_start_time=start_time,
                request_type="OptimStep",
                queue_state_observer=self._queue_state_logger,
            )

        return self.holder.run_coroutine_threadsafe(_optim_step_async())

    async def optim_step_async(
        self, adam_params: types.AdamParams
    ) -> APIFuture[types.OptimStepResponse]:
        """Async version of optim_step."""
        return self.optim_step(adam_params)

    def save_state(
        self, name: str, ttl_seconds: int | None = None, overwrite: bool = False
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Save model weights to persistent storage.

        Args:
        - `name`: Name for the saved checkpoint
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint (None = never expires)
        - `overwrite`: If True, overwrite any existing checkpoint with the same name

        Returns:
        - `APIFuture` containing the save response with checkpoint path

        Example:
        ```python
        # Save after training
        save_future = training_client.save_state("checkpoint-001")
        result = await save_future
        print(f"Saved to: {result.path}")
        ```
        """
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _save_state_async():
            start_time = time.time()

            async def _send_request():
                request = types.SaveWeightsRequest(
                    model_id=self._guaranteed_model_id(),
                    path=name,
                    seq_id=request_id + 1,
                    ttl_seconds=ttl_seconds,
                    overwrite=overwrite,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.save(
                        request=request,
                        max_retries=0,
                    )

            async with self._take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)

            return await _APIFuture(
                types.SaveWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeights",
                queue_state_observer=self._queue_state_logger,
            )

        return self.holder.run_coroutine_threadsafe(_save_state_async())

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None, overwrite: bool = False
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Async version of save_state."""
        return self.save_state(name, ttl_seconds=ttl_seconds, overwrite=overwrite)

    def _load_state_impl(
        self, path: str, optimizer: bool, weights_access_token: str | None
    ) -> APIFuture[types.LoadWeightsResponse]:
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _load_state_async() -> types.LoadWeightsResponse:
            start_time = time.time()

            async def _send_request():
                request = types.LoadWeightsRequest(
                    model_id=self._guaranteed_model_id(),
                    path=path,
                    seq_id=request_id + 1,
                    optimizer=optimizer,
                    weights_access_token=weights_access_token,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.load(
                        request=request,
                    )

            async with self._take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.LoadWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="LoadWeights",
                queue_state_observer=self._queue_state_logger,
            )

        return self.holder.run_coroutine_threadsafe(_load_state_async())

    def load_state(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights from a saved checkpoint.

        This loads only the model weights, not optimizer state (e.g., Adam momentum).
        To also restore optimizer state, use load_state_with_optimizer.

        Args:
        - `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `weights_access_token`: Optional access token for loading checkpoints under a different account.

        Returns:
        - `APIFuture` containing the load response

        Example:
        ```python
        # Load checkpoint to continue training (weights only, optimizer resets)
        load_future = training_client.load_state("tinker://run-id/weights/checkpoint-001")
        await load_future
        # Continue training from loaded state
        ```
        """
        return self._load_state_impl(
            path, optimizer=False, weights_access_token=weights_access_token
        )

    async def load_state_async(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state."""
        return self.load_state(path, weights_access_token=weights_access_token)

    def load_state_with_optimizer(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights and optimizer state from a checkpoint.

        Args:
        - `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `weights_access_token`: Optional access token for loading checkpoints under a different account.

        Returns:
        - `APIFuture` containing the load response

        Example:
        ```python
        # Resume training with optimizer state
        load_future = training_client.load_state_with_optimizer(
            "tinker://run-id/weights/checkpoint-001"
        )
        await load_future
        # Continue training with restored optimizer momentum
        ```
        """
        return self._load_state_impl(
            path, optimizer=True, weights_access_token=weights_access_token
        )

    async def load_state_with_optimizer_async(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state_with_optimizer."""
        return self.load_state_with_optimizer(path, weights_access_token=weights_access_token)

    def _save_weights_for_sampler_impl(
        self, name: str | None, ttl_seconds: int | None
    ) -> APIFuture[types.SaveWeightsForSamplerResponse | str]:
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _save_weights_for_sampler_async() -> types.SaveWeightsForSamplerResponse | str:
            start_time = time.time()

            async def _send_request():
                if name is not None:
                    request = types.SaveWeightsForSamplerRequest(
                        model_id=self._guaranteed_model_id(),
                        path=name,
                        seq_id=request_id + 1,
                        ttl_seconds=ttl_seconds,
                    )
                else:
                    # Training client can never be created from a shadow holder, so we can safely assert
                    assert self.holder._sampling_client_counter is not None
                    sampling_session_seq_id = self.holder._sampling_client_counter
                    self.holder._sampling_client_counter += 1
                    request = types.SaveWeightsForSamplerRequest(
                        model_id=self._guaranteed_model_id(),
                        seq_id=request_id + 1,
                        sampling_session_seq_id=sampling_session_seq_id,
                        ttl_seconds=ttl_seconds,
                    )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.save_for_sampler(
                        request=request,
                        max_retries=0,
                    )

            async with self._take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)

            result = await _APIFuture(
                types.SaveWeightsForSamplerResponseInternal,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeightsForSampler",
                queue_state_observer=self._queue_state_logger,
            )
            if name is not None:
                assert result.path is not None
                return types.SaveWeightsForSamplerResponse(path=result.path)
            else:
                assert result.sampling_session_id is not None
                return result.sampling_session_id

        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    def save_weights_for_sampler(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Save model weights for use with a SamplingClient.

        Args:
        - `name`: Name for the saved sampler weights
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint (None = never expires)

        Returns:
        - `APIFuture` containing the save response with sampler path

        Example:
        ```python
        # Save weights for inference
        save_future = training_client.save_weights_for_sampler("sampler-001")
        result = await save_future
        print(f"Sampler weights saved to: {result.path}")

        # Use the path to create a sampling client
        sampling_client = service_client.create_sampling_client(
            model_path=result.path
        )
        ```
        """

        async def _save_weights_for_sampler_async() -> types.SaveWeightsForSamplerResponse:
            result = await self._save_weights_for_sampler_impl(name, ttl_seconds)
            assert isinstance(result, types.SaveWeightsForSamplerResponse)
            return result

        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Async version of save_weights_for_sampler."""
        return self.save_weights_for_sampler(name, ttl_seconds=ttl_seconds)

    def _get_info_submit(self) -> AwaitableConcurrentFuture[types.GetInfoResponse]:
        @capture_exceptions(fatal=True)
        async def _get_info_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    request = types.GetInfoRequest(model_id=self._guaranteed_model_id())
                    return await client.models.get_info(
                        request=request,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_info_async())

    @sync_only
    def get_info(self) -> types.GetInfoResponse:
        """Get information about the current model.

        Returns:
        - `GetInfoResponse` with model configuration and metadata

        Example:
        ```python
        info = training_client.get_info()
        print(f"Model ID: {info.model_data.model_id}")
        print(f"Base model: {info.model_data.model_name}")
        print(f"LoRA rank: {info.model_data.lora_rank}")
        ```
        """
        return self._get_info_submit().result()

    async def get_info_async(self) -> types.GetInfoResponse:
        """Async version of get_info."""
        return await self._get_info_submit()

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the current model.

        Returns:
        - `PreTrainedTokenizer` compatible with the model

        Example:
        ```python
        tokenizer = training_client.get_tokenizer()
        tokens = tokenizer.encode("Hello world")
        text = tokenizer.decode(tokens)
        ```
        """
        return _get_tokenizer(self._guaranteed_model_id(), self.holder)

    def create_sampling_client(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Create a SamplingClient from saved weights.

        Args:
        - `model_path`: Tinker path to saved weights
        - `retry_config`: Optional configuration for retrying failed requests

        Returns:
        - `SamplingClient` configured with the specified weights

        Example:
        ```python
        sampling_client = training_client.create_sampling_client(
            "tinker://run-id/weights/checkpoint-001"
        )
        # Use sampling_client for inference
        ```
        """
        return SamplingClient.create(
            self.holder, model_path=model_path, retry_config=retry_config
        ).result()

    async def create_sampling_client_async(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Async version of create_sampling_client."""
        return await SamplingClient.create(
            self.holder, model_path=model_path, retry_config=retry_config
        )

    def save_weights_and_get_sampling_client(
        self, name: str | None = None, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Save current weights and create a SamplingClient for inference.

        Args:
        - `name`: Deprecated, has no effect. Will be removed in a future release.
        - `retry_config`: Optional configuration for retrying failed requests

        Returns:
        - `SamplingClient` configured with the current model weights

        Example:
        ```python
        # After training, create a sampling client directly
        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Now use it for inference
        prompt = types.ModelInput.from_ints(tokenizer.encode("Hello"))
        params = types.SamplingParams(max_tokens=20)
        result = sampling_client.sample(prompt, 1, params).result()
        ```
        """
        if name is not None:
            warnings.warn(
                "The 'name' parameter of save_weights_and_get_sampling_client() is deprecated "
                "and has no effect — checkpoints are always ephemeral. "
                "This parameter will be removed in a future release. "
                "Remove the 'name' argument from your call. "
                "If you need a persistent checkpoint, use "
                "save_weights_for_sampler(name=...) + create_sampling_client(model_path=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        sampling_session_id = self._save_weights_for_sampler_impl(None, None).result()
        assert isinstance(sampling_session_id, str)
        return SamplingClient.create(
            self.holder,
            sampling_session_id=sampling_session_id,
            retry_config=retry_config,
        ).result()

    async def save_weights_and_get_sampling_client_async(
        self, name: str | None = None, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Async version of save_weights_and_get_sampling_client."""
        if name is not None:
            warnings.warn(
                "The 'name' parameter of save_weights_and_get_sampling_client_async() is deprecated "
                "and has no effect — checkpoints are always ephemeral. "
                "This parameter will be removed in a future release. "
                "Remove the 'name' argument from your call. "
                "If you need a persistent checkpoint, use "
                "save_weights_for_sampler(name=...) + create_sampling_client(model_path=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        sampling_session_id = self._save_weights_for_sampler_impl(None, None).result()
        assert isinstance(sampling_session_id, str)
        return await SamplingClient.create(
            self.holder,
            sampling_session_id=sampling_session_id,
            retry_config=retry_config,
        )

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


def _get_tokenizer(model_id: types.ModelID, holder: InternalClientHolder) -> PreTrainedTokenizer:
    """Get tokenizer for a training model by fetching model info first."""

    async def _get_info_async():
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            request = types.GetInfoRequest(model_id=model_id)
            return await client.models.get_info(request=request)

    info = holder.run_coroutine_threadsafe(_get_info_async()).result()
    model_name = info.model_data.model_name
    assert model_name is not None, "This shouldn't happen: model_name is None"

    return _load_tokenizer_from_model_info(model_name, info.model_data.tokenizer_id)
