"""TrainingClient for Tinker API."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import warnings
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Tuple,
)

from typing_extensions import deprecated

from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.console_urls import training_run_console_url
from tinker.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from tinker.lib.sdk_method import with_sdk_method_telemetry
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider
from tinker.types.optimizer import OptimParamsBase

from ..api_future_impl import (
    _APIFuture,
    _CombinedAPIFuture,
)
from ..chunked_fwdbwd_helpers import combine_fwd_bwd_output_results
from ..queue_state_logger import QueueStateLogger
from ..retry_handler import RetryConfig
from ..sync_only import sync_only
from .custom_loss import LossHandle, check_loss_inputs, run_loss_handle
from .sampling_client import SamplingClient, _load_tokenizer_from_model_info

try:
    import torch
except ImportError:
    torch = None


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)


# fwd/bwd requests are serialized as proto (proto/request_conv.py): tokens
# pack as int32, sparse CSR crow/col indices as int64.
_PROTO_BYTES_PER_TOKEN = 4
_SPARSE_INDEX_BYTES = 8


def _estimate_chunk_bytes_count_proto(chunk: types.ModelInputChunk) -> int:
    if isinstance(chunk, types.DmelChunk):
        return len(chunk.dmel)
    if isinstance(chunk, types.ImageChunk):
        return len(chunk.data)
    if isinstance(chunk, types.ImageAssetPointerChunk):
        return len(chunk.location)
    return chunk.length * _PROTO_BYTES_PER_TOKEN


def _estimate_tensor_bytes_count_proto(td: types.TensorData) -> int:
    # _numpy is stored in the declared dtype, so nbytes matches the dense
    # proto payload exactly.
    bytes_count = td._numpy.nbytes
    if td.sparse_crow_indices is not None:
        bytes_count += _SPARSE_INDEX_BYTES * len(td.sparse_crow_indices)
    if td.sparse_col_indices is not None:
        bytes_count += _SPARSE_INDEX_BYTES * len(td.sparse_col_indices)
    return bytes_count


def _estimate_datum_bytes_count_proto(datum: types.Datum) -> int:
    return sum(
        _estimate_chunk_bytes_count_proto(chunk) for chunk in datum.model_input.chunks
    ) + sum(_estimate_tensor_bytes_count_proto(td) for td in datum.loss_fn_inputs.values())


MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."

# Type alias for custom loss functions.
# Args: (data: List[Datum], model_outputs: List[Any]) -> (loss: Any, metrics: Dict[str, float])
CustomLossFnV1 = Callable[[List[types.Datum], List[Any]], Tuple[Any, Dict[str, float]]]

_LOAD_STATE_DEPRECATION_MESSAGE = (
    "DEPRECATED: TrainingClient.load_state*() is deprecated and will NOT be supported in "
    "future tinker SDK versions. Migrate now: create a new training client from the "
    "checkpoint with ServiceClient.create_training_client_from_state() or "
    "ServiceClient.create_training_client_from_state_with_optimizer()."
)

_custom_loss_loop: asyncio.AbstractEventLoop | None = None
_custom_loss_loop_lock = threading.Lock()

_SUPPORTED_CUSTOM_BACKEND_LOSS_FNS = frozenset({"cross_entropy"})

_CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE: dict[Literal["logprobs"], types.LossFnType] = {
    "logprobs": "cross_entropy",
}


def _check_loss_type_input(loss_type_input: str | None, method: str) -> None:
    if loss_type_input is None:
        return
    warnings.warn(
        f"The 'loss_type_input' parameter of {method}() is deprecated: the loss always "
        "receives logprobs. This parameter will be removed in a future release; remove it "
        "from your call.",
        DeprecationWarning,
        stacklevel=3,
    )
    if loss_type_input not in _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE:
        supported = ", ".join(sorted(_CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE))
        raise ValueError(
            f"Unsupported loss_type_input={loss_type_input!r}. Supported values are: {supported}"
        )


class _RequestChunk(NamedTuple):
    data: List[types.Datum]
    estimated_bytes_count: int


class _ReservedRequestChunk(NamedTuple):
    request_id: int
    data: List[types.Datum]
    estimated_bytes_count: int


@dataclass(frozen=True)
class _CustomLossResult:
    data: List[types.Datum]
    metrics: Dict[str, float]
    loss_sum: float


@with_sdk_method_telemetry
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

        self._request_error: BaseException | None = None
        self._turn_counter: int = 0
        # Maps turn id → list of events to wake when _turn_counter reaches that id.
        self._turn_waiters: dict[int, list[asyncio.Event]] = {}

        self._queue_state_logger = QueueStateLogger(str(model_id), "Training")

    # Reserves a request id for a request. Requests are to be executed in the order of request ids.
    def _get_request_id(self) -> int:
        return self._reserve_request_ids(1).start

    def _reserve_request_ids(self, count: int) -> range:
        with self._request_id_lock:
            start = self._request_id_counter
            self._request_id_counter += count
            return range(start, self._request_id_counter)

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
        if self._request_error is None and self._turn_counter < min_turn:
            event = asyncio.Event()
            self._turn_waiters.setdefault(min_turn, []).append(event)
            await event.wait()

        if self._request_error is not None:
            raise RuntimeError(
                "A previous request failed; this TrainingClient is poisoned "
                "and cannot accept further requests."
            ) from self._request_error

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

    def get_console_url(self) -> str:
        """Return the Tinker Console URL for this training run."""
        return training_run_console_url(self._guaranteed_model_id())

    def _chunked_requests_generator(
        self, data: List[types.Datum]
    ) -> Generator[_RequestChunk, None, None]:
        cfg = self.holder._client_config
        current_chunk: List[types.Datum] = []
        current_chunk_bytes_count = 0

        for datum in data:
            estimated_bytes_count = _estimate_datum_bytes_count_proto(datum)
            if (
                len(current_chunk) > 0
                and current_chunk_bytes_count + estimated_bytes_count
                > cfg.fwdbwd_max_chunk_bytes_count
            ) or (len(current_chunk) == cfg.fwdbwd_max_chunk_len):
                yield _RequestChunk(current_chunk, current_chunk_bytes_count)
                current_chunk = []
                current_chunk_bytes_count = 0

            current_chunk.append(datum)
            current_chunk_bytes_count += estimated_bytes_count

        if len(current_chunk) > 0:
            yield _RequestChunk(current_chunk, current_chunk_bytes_count)

    def _chunked_requests(self, data: List[types.Datum]) -> List[_ReservedRequestChunk]:
        return [
            _ReservedRequestChunk(self._get_request_id(), chunk.data, chunk.estimated_bytes_count)
            for chunk in self._chunked_requests_generator(data)
        ]

    def forward(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Mapping[str, float | str] | None = None,
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
        print(f"Loss: {result.loss}")
        ```
        """
        return self._run_fwd_bwd(
            self._chunked_requests(data), loss_fn, loss_fn_config, forward_only=True
        )

    async def forward_async(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Mapping[str, float | str] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward."""
        return self.forward(data, loss_fn, loss_fn_config)

    def forward_backward(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Mapping[str, float | str] | None = None,
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
        print(f"Loss: {fwdbwd_result.loss}")
        ```
        """
        return self._run_fwd_bwd(
            self._chunked_requests(data), loss_fn, loss_fn_config, forward_only=False
        )

    def _run_fwd_bwd(
        self,
        requests: List[_ReservedRequestChunk],
        loss_fn: types.LossFnType,
        loss_fn_config: Mapping[str, float | str] | None,
        *,
        forward_only: bool,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Submit preallocated /forward_backward chunks, optionally in parallel."""
        if not requests:
            raise ValueError("No data provided")

        parallel = self.holder._client_config.parallel_fwdbwd_chunks
        # When parallel, all chunks share [min, max] range and fire concurrently.
        # When serial, each chunk takes its own turn sequentially.
        min_rid = requests[0].request_id
        max_rid = requests[-1].request_id if parallel else None
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
                chunk: _ReservedRequestChunk,
            ) -> APIFuture[types.ForwardBackwardOutput]:
                turn_min = min_rid if parallel else chunk.request_id
                turn_max = max_rid if parallel else None
                async with self._take_turn(turn_min, turn_max):
                    async with self.holder.fwdbwd_dispatch_rate_limit(chunk.estimated_bytes_count):
                        untyped_future = await self.holder.execute_with_retries(
                            _send_request, chunk.request_id, chunk.data
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
                    await asyncio.gather(*[_submit_chunk(chunk) for chunk in requests[1:]])
                )
                first_future = await _submit_chunk(requests[0])
                futures = [first_future] + rest_futures
            else:
                # gather is safe even when serial — _take_turn orders execution.
                futures = list(await asyncio.gather(*[_submit_chunk(chunk) for chunk in requests]))

            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_run_async())

    async def forward_backward_async(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Mapping[str, float | str] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward_backward."""
        return self.forward_backward(data, loss_fn, loss_fn_config)

    def forward_backward_custom(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV1 | LossHandle,
        *,
        loss_type_input: Literal["logprobs"] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward/backward with a custom loss function.

        Allows you to define custom loss functions that operate on log probabilities.
        A callable, or a `LossHandle` that runs client-side, receives every
        `loss_fn_inputs` field, but only `target_tokens` and `weights` are sent to the
        server. A `LossHandle` the server runs (one with a `loss_id`) is sent with its
        data unchanged and runs in a single server-side forward/backward.

        Returns a future without waiting for server submission or processing. Requests on
        this client execute in submission order, so you can submit an optimizer step before
        waiting for this future. If a client-side custom loss operation fails or is cancelled,
        this client rejects queued and subsequent training requests.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Custom loss function that takes (data, logprobs) and returns (loss, metrics),
          or a `LossHandle` from `ServiceClient.create_custom_loss`
        - `loss_type_input`: Deprecated; the loss always receives logprobs. Will be removed in
          a future release.

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
        print(f"Custom loss: {result.loss}")
        print(f"Metrics: {result.metrics}")
        ```

        For example, to provide a target distribution for a custom loss:
        ```python
        data = [types.Datum(
            model_input=types.ModelInput.from_ints([1]),
            loss_fn_inputs={
                "target_tokens": [[2, 3]],
                "target_distribution": types.TensorData(
                    data=[0.25, 0.75], dtype="float32", shape=[1, 2],
                ),
            },
        )]

        def custom_loss(data, logprobs_list):
            loss = sum(
                -(datum.loss_fn_inputs["target_distribution"].to_torch() * logprobs).sum()
                for datum, logprobs in zip(data, logprobs_list, strict=True)
            )
            return loss, {}

        future = training_client.forward_backward_custom(data, custom_loss)
        ```
        """
        _check_loss_type_input(loss_type_input, "forward_backward_custom")
        if torch is None:
            raise ImportError("PyTorch is not installed. Cannot run custom forward_backward.")

        if isinstance(loss_fn, LossHandle) and loss_fn.session_id != self.holder.get_session_id():
            raise ValueError(
                f"LossHandle {loss_fn.name!r} was created in session {loss_fn.session_id}; it can "
                "only be used by clients of that session"
            )
        if isinstance(loss_fn, LossHandle):
            check_loss_inputs(loss_fn, data)

        if isinstance(loss_fn, LossHandle) and loss_fn.loss_id is not None:
            # An uploaded loss runs on the server in one forward_backward, with no
            # client-side forward pass or surrogate.
            return self.forward_backward(data, "custom", {"loss_id": loss_fn.loss_id})

        surrogate_loss_fn = _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE["logprobs"]

        forward_data = self._get_custom_loss_forward_data(data, surrogate_loss_fn)
        forward_chunks = list(self._chunked_requests_generator(forward_data))
        if not forward_chunks:
            raise ValueError("No data provided")

        # Both phases must precede later requests, even while forward is pending.
        num_chunks = len(forward_chunks)
        reserved_ids = self._reserve_request_ids(2 * num_chunks)
        forward_ids = reserved_ids[:num_chunks]
        forward_requests = [
            _ReservedRequestChunk(request_id, chunk.data, chunk.estimated_bytes_count)
            for request_id, chunk in zip(forward_ids, forward_chunks, strict=True)
        ]

        async def run_reserved_operation() -> types.ForwardBackwardOutput:
            try:
                forward_result = await self._run_fwd_bwd(
                    forward_requests, surrogate_loss_fn, None, forward_only=True
                ).result_async()
                custom_loss = self._compute_custom_loss(data, loss_fn, forward_result)
                return await self._run_custom_backward(forward_requests, custom_loss)
            except BaseException as exc:
                # Missing reserved IDs cannot be skipped by the server.
                # Turn waiters belong to the holder loop.
                await self.holder.run_coroutine_threadsafe(
                    self._fail_custom_requests(exc)
                ).result_async()
                raise

        future = AwaitableConcurrentFuture(
            asyncio.run_coroutine_threadsafe(run_reserved_operation(), _get_custom_loss_loop())
        )

        def on_done(result: Future[types.ForwardBackwardOutput]) -> None:
            # A task cancelled before it starts never enters the coroutine's try/except.
            if result.cancelled():
                self.holder.run_coroutine_threadsafe(
                    self._fail_custom_requests(asyncio.CancelledError())
                )

        future.future().add_done_callback(on_done)
        return future

    async def forward_backward_custom_async(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV1 | LossHandle,
        *,
        loss_type_input: Literal["logprobs"] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward_backward_custom."""
        return self.forward_backward_custom(data, loss_fn, loss_type_input=loss_type_input)

    @staticmethod
    def _compute_custom_loss(
        data: List[types.Datum],
        loss_fn: CustomLossFnV1 | LossHandle,
        forward_result: types.ForwardBackwardOutput,
    ) -> _CustomLossResult:
        assert torch is not None
        logprobs_list = []
        for out in forward_result.loss_fn_outputs:
            logprob = torch.tensor(out["logprobs"].data)
            if out["logprobs"].shape is not None:
                logprob = logprob.reshape(out["logprobs"].shape)
            logprob = logprob.clone().detach().requires_grad_(True)
            logprobs_list.append(logprob)

        # Differentiate the local loss to produce weights for the server's backward pass.
        if isinstance(loss_fn, LossHandle):
            loss, metrics = run_loss_handle(loss_fn, data, logprobs_list)
        else:
            loss, metrics = loss_fn(data, logprobs_list)
        custom_loss_sum = float(loss.detach().sum().item())
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
                    model_input_spans=datum.model_input_spans,
                    loss_fn_input_spans=datum.loss_fn_input_spans,
                )
            )

        return _CustomLossResult(linear_loss_data, metrics, custom_loss_sum)

    async def _run_custom_backward(
        self,
        forward_requests: List[_ReservedRequestChunk],
        custom_loss: _CustomLossResult,
    ) -> types.ForwardBackwardOutput:
        # Reuse the planned boundaries: gradient weights may change the byte estimates,
        # but rechunking here would leave gaps in the reserved sequence IDs.
        num_chunks = len(forward_requests)
        backward_requests = []
        offset = 0
        for chunk in forward_requests:
            backward_request_id = chunk.request_id + num_chunks
            backward_chunk = custom_loss.data[offset : offset + len(chunk.data)]
            offset += len(chunk.data)
            backward_requests.append(
                _ReservedRequestChunk(
                    backward_request_id,
                    backward_chunk,
                    sum(_estimate_datum_bytes_count_proto(datum) for datum in backward_chunk),
                )
            )

        backward_future = self._run_fwd_bwd(
            backward_requests,
            _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE["logprobs"],
            None,
            forward_only=False,
        )
        backward_result = await backward_future.result_async()
        surrogate_loss_sum = backward_result.metrics.pop("loss:sum")
        backward_result.metrics.update(custom_loss.metrics)
        backward_result.metrics["loss:sum"] = custom_loss.loss_sum
        backward_result.metrics["surrogate_loss:sum"] = surrogate_loss_sum
        return backward_result

    async def _fail_custom_requests(self, error: BaseException) -> None:
        if self._request_error is None:
            self._request_error = error
        for waiters in self._turn_waiters.values():
            for event in waiters:
                event.set()
        self._turn_waiters.clear()

    def _get_custom_loss_forward_data(
        self,
        data: List[types.Datum],
        surrogate_loss_fn: types.LossFnType,
    ) -> List[types.Datum]:
        assert surrogate_loss_fn in _SUPPORTED_CUSTOM_BACKEND_LOSS_FNS, (
            "forward_backward_custom should validate surrogate_loss_fn before "
            "_get_custom_loss_forward_data is called"
        )

        forward_data = []
        for datum in data:
            target_tokens = datum.loss_fn_inputs.get("target_tokens")
            if target_tokens is None:
                raise ValueError("target_tokens must be provided when using cross_entropy")

            weights = datum.loss_fn_inputs.get("weights")
            if weights is None:
                weights = types.TensorData(
                    data=[0.0] * len(target_tokens.data),
                    dtype="float32",
                    shape=target_tokens.shape,
                )
            forward_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
                    model_input_spans=datum.model_input_spans,
                    loss_fn_input_spans=datum.loss_fn_input_spans,
                )
            )

        return forward_data

    def optim_step(
        self,
        optim_params: types.OptimParams | None = None,
        *,
        adam_params: types.AdamParams | None = None,
    ) -> APIFuture[types.OptimStepResponse]:
        """Update model parameters using the optimizer selected at model creation.

        The Adam optimizer used by tinker is identical
        to [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        Note that unlike PyTorch, Tinker's default weight decay value is 0.0 (no weight decay).


        Args:
        - `optim_params`: AdamParams or DimuonParams matching the optimizer selected at model creation.
        - `adam_params`: Legacy keyword alias for AdamParams; cannot be combined with optim_params.

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
        if optim_params is not None and adam_params is not None:
            raise ValueError("Specify optim_params or legacy adam_params, not both")
        params = optim_params if optim_params is not None else adam_params
        if not isinstance(params, (types.AdamParams, OptimParamsBase)):
            raise TypeError("Expected AdamParams or OptimParamsBase")
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _optim_step_async():
            start_time = time.time()

            async def _send_request():
                request = types.OptimStepRequest(
                    optim_params=params,
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
        self,
        optim_params: types.OptimParams | None = None,
        *,
        adam_params: types.AdamParams | None = None,
    ) -> APIFuture[types.OptimStepResponse]:
        """Async version of optim_step."""
        return self.optim_step(optim_params, adam_params=adam_params)

    def save_state(
        self,
        name: str,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Save model weights to persistent storage.

        Args:
        - `name`: Name for the saved checkpoint
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint, between 1 hour (3600) and
          10 years (None = never expires)
        - `overwrite`: If True, overwrite any existing checkpoint with the same name. This
          replaces the entire existing `user_metadata` mapping; if `user_metadata` is not
          provided, the previous user metadata is deleted.
        - `user_metadata`: Optional user-provided metadata to attach to the checkpoint

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
                    user_metadata=user_metadata,
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
        self,
        name: str,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Async version of save_state."""
        return self.save_state(
            name,
            ttl_seconds=ttl_seconds,
            overwrite=overwrite,
            user_metadata=user_metadata,
        )

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

    def _deprecated_load_state(
        self, path: str, optimizer: bool, weights_access_token: str | None
    ) -> APIFuture[types.LoadWeightsResponse]:
        logger.error(_LOAD_STATE_DEPRECATION_MESSAGE)
        return self._load_state_impl(
            path, optimizer=optimizer, weights_access_token=weights_access_token
        )

    @deprecated(_LOAD_STATE_DEPRECATION_MESSAGE, category=FutureWarning)
    def load_state(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights from a saved checkpoint.

        This loads only the model weights, not optimizer state (e.g., Adam momentum).
        To also restore optimizer state, use load_state_with_optimizer.

        Deprecated: will not be supported in future SDK versions. Use
        `ServiceClient.create_training_client_from_state()` instead.

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
        return self._deprecated_load_state(
            path, optimizer=False, weights_access_token=weights_access_token
        )

    @deprecated(_LOAD_STATE_DEPRECATION_MESSAGE, category=FutureWarning)
    async def load_state_async(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state."""
        return self._deprecated_load_state(
            path, optimizer=False, weights_access_token=weights_access_token
        )

    @deprecated(_LOAD_STATE_DEPRECATION_MESSAGE, category=FutureWarning)
    def load_state_with_optimizer(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights and optimizer state from a checkpoint.

        Deprecated: will not be supported in future SDK versions. Use
        `ServiceClient.create_training_client_from_state_with_optimizer()` instead.

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
        return self._deprecated_load_state(
            path, optimizer=True, weights_access_token=weights_access_token
        )

    @deprecated(_LOAD_STATE_DEPRECATION_MESSAGE, category=FutureWarning)
    async def load_state_with_optimizer_async(
        self, path: str, weights_access_token: str | None = None
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state_with_optimizer."""
        return self._deprecated_load_state(
            path, optimizer=True, weights_access_token=weights_access_token
        )

    def _save_weights_for_sampler_impl(
        self,
        name: str | None,
        ttl_seconds: int | None,
        user_metadata: dict[str, str] | None,
        kind: types.SamplingClientKind,
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
                        user_metadata=user_metadata,
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
                        kind=kind,
                        ttl_seconds=ttl_seconds,
                        user_metadata=user_metadata,
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
        self,
        name: str,
        ttl_seconds: int | None = None,
        user_metadata: dict[str, str] | None = None,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Save model weights for use with a SamplingClient.

        Args:
        - `name`: Name for the saved sampler weights
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint, between 1 hour (3600) and
          10 years (None = never expires)
        - `user_metadata`: Optional user-provided metadata to attach to the checkpoint

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
            result = await self._save_weights_for_sampler_impl(
                name, ttl_seconds, user_metadata, "snapshot"
            )
            assert isinstance(result, types.SaveWeightsForSamplerResponse)
            return result

        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    async def save_weights_for_sampler_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
        user_metadata: dict[str, str] | None = None,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Async version of save_weights_for_sampler."""
        return self.save_weights_for_sampler(
            name,
            ttl_seconds=ttl_seconds,
            user_metadata=user_metadata,
        )

    def save_weights_external(
        self,
        name: str,
        ttl_seconds: int | None = None,
        age_encryption_recipients: list[str] | None = None,
    ) -> APIFuture[types.SaveWeightsExternalResponse]:
        """Save model weights in an external (e.g. HuggingFace) format.

        The saved checkpoint is stored in the external-weights bucket and can be
        managed (listed, deleted) like any other checkpoint.

        Args:
        - `name`: Name for the saved external weights
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint, between 1 hour (3600) and
          10 years (None = never expires)
        - `age_encryption_recipients`: Optional age (filippo.io/age) recipients (at most 10),
          native `age1...` keys or SSH public keys. When set, every exported file is
          encrypted to them and stored with a `.age` suffix; decrypt with
          `age -d -i <identity>`.

        Returns:
        - `APIFuture` containing the save response with the external weights path
        """
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _save_weights_external_async() -> types.SaveWeightsExternalResponse:
            start_time = time.time()

            async def _send_request():
                request = types.SaveWeightsExternalRequest(
                    model_id=self._guaranteed_model_id(),
                    path=name,
                    seq_id=request_id + 1,
                    ttl_seconds=ttl_seconds,
                    age_encryption_recipients=age_encryption_recipients,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.save_external(request=request, max_retries=0)

            async with self._take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)

            result = await _APIFuture(
                types.SaveWeightsExternalResponseInternal,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeightsExternal",
                queue_state_observer=self._queue_state_logger,
            )
            assert result.path is not None
            return types.SaveWeightsExternalResponse(path=result.path)

        return self.holder.run_coroutine_threadsafe(_save_weights_external_async())

    async def save_weights_external_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
        age_encryption_recipients: list[str] | None = None,
    ) -> APIFuture[types.SaveWeightsExternalResponse]:
        """Async version of save_weights_external."""
        return self.save_weights_external(
            name, ttl_seconds=ttl_seconds, age_encryption_recipients=age_encryption_recipients
        )

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
        self,
        model_path: str,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
    ) -> SamplingClient:
        """Create a SamplingClient from saved weights.

        Args:
        - `model_path`: Tinker path to saved weights
        - `retry_config`: Optional configuration for retrying failed requests
        - `record_stability_info`: When true, every sample from this client
            records stability information for its sequences (StableReinforce
            training); recorded data is referenced from training requests by
            `sequence_id`, is session-scoped, and expires after inactivity.

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
            self.holder,
            model_path=model_path,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        ).result()

    async def create_sampling_client_async(
        self,
        model_path: str,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
    ) -> SamplingClient:
        """Async version of create_sampling_client."""
        return await SamplingClient.create(
            self.holder,
            model_path=model_path,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        )

    def save_weights_and_get_sampling_client(
        self,
        name: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
        kind: types.SamplingClientKind = "snapshot",
    ) -> SamplingClient:
        """Save current weights and create a SamplingClient for inference.

        Args:
        - `name`: Deprecated, has no effect. Will be removed in a future release.
        - `retry_config`: Optional configuration for retrying failed requests
        - `record_stability_info`: When true, every sample from this client
            records stability information for its sequences (StableReinforce
            training); recorded data is referenced from training requests by
            `sequence_id`, is session-scoped, and expires after inactivity.
        - `kind`: `snapshot` (default) captures fixed weights. `inflight` publishes
          mutable weights shared by this training model's inflight sampling clients.

        See `types.SamplingClientKind` for the snapshot and inflight sampling contracts.

        Returns:
        - `SamplingClient` using the saved snapshot or shared inflight weights

        Example:
        ```python
        # After training, create a sampling client directly
        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Now use it for inference
        prompt = types.ModelInput.from_ints(tokenizer.encode("Hello"))
        params = types.SamplingParams(max_tokens=20)
        result = sampling_client.sample(prompt, 1, params).result()
        ```

        Inflight example:
        ```python
        # Publish W1 and use sampler_a for the rollouts starting now.
        sampler_a = training_client.save_weights_and_get_sampling_client(kind="inflight")

        # ... train further ...

        # Publish W2 and give sampler_b to newly starting rollouts.
        sampler_b = training_client.save_weights_and_get_sampling_client(kind="inflight")

        # sampler_b requires W2 or newer and has a fresh KV namespace.
        # sampler_a keeps its namespace and can adopt W2 as it propagates.
        # Until then, sampler_a can still use a replica serving W1.
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
        sampling_session_id = self._save_weights_for_sampler_impl(None, None, None, kind).result()
        assert isinstance(sampling_session_id, str)
        return SamplingClient.create(
            self.holder,
            sampling_session_id=sampling_session_id,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        ).result()

    async def save_weights_and_get_sampling_client_async(
        self,
        name: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
        kind: types.SamplingClientKind = "snapshot",
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
        sampling_session_id = self._save_weights_for_sampler_impl(None, None, None, kind).result()
        assert isinstance(sampling_session_id, str)
        return await SamplingClient.create(
            self.holder,
            sampling_session_id=sampling_session_id,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        )

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


def _get_custom_loss_loop() -> asyncio.AbstractEventLoop:
    global _custom_loss_loop
    with _custom_loss_loop_lock:
        if _custom_loss_loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="tinker-custom-loss", daemon=True
            ).start()
            _custom_loss_loop = loop
        return _custom_loss_loop


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
