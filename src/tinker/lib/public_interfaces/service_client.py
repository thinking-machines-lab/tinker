"""ServiceClient for Tinker API."""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import Future as ConcurrentFuture
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self

from typing_extensions import deprecated

from tinker import types
from tinker._types import NoneType
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.console_urls import session_console_url, sessions_console_url
from tinker.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from tinker.lib.sdk_method import with_sdk_method_telemetry
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import _APIFuture
from ..internal_client_holder import InternalClientHolder
from ..queue_state_logger import QueueStateLogger
from ..retry_handler import RetryConfig
from ..sync_only import sync_only
from .custom_loss import LossHandle, PerDatumLossFn, encode_graph, trace_loss

if TYPE_CHECKING:
    from .backcompat_sampling_client_future import BackcompatSamplingClientFuture
    from .backcompat_training_client_future import BackcompatTrainingClientFuture
    from .rest_client import RestClient
    from .sampling_client import SamplingClient
    from .training_client import TrainingClient

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)


@with_sdk_method_telemetry
class ServiceClient(TelemetryProvider):
    """The ServiceClient is the main entry point for the Tinker API. It provides methods to:
    - Generate TrainingClient instances for model training workflows
    - Generate SamplingClient instances for text generation and inference
    - Close the client by finishing the current session when the training script is done

    Args:
        user_metadata: Optional metadata attached to the created session.
        project_id: Optional project ID to attach to the created session. If not
            provided, falls back to the `TINKER_PROJECT_ID` environment variable.
        display_name: Optional human-readable display name for the session. Does
            not need to be unique. Only used for display and search purposes.
        base_url: The Tinker API URL. Defaults to the `TINKER_BASE_URL` environment
            variable, then to the production API.
        api_key: The Tinker API key. Defaults to the `TINKER_API_KEY` environment
            variable, then to the stored credentials.
        default_headers_overrides: Headers sent with every request, on top of the ones
            the SDK derives from environment variables.
        **kwargs: Unsupported and ignored, with a warning. Passing them will be an
            error in a future version of the SDK.

    Example:
    ```python
    # Near instant
    client = ServiceClient()

    # Takes a moment as we initialize the model and assign resources
    training_client = client.create_lora_training_client(base_model="Qwen/Qwen3-8B").result()

    # Near-instant
    sampling_client = client.create_sampling_client(base_model="Qwen/Qwen3-8B").result()
    ```

    To check the server's health or fetch the models you can use, use `RestClient()` instead;
    it needs no session.
    """

    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        *,
        display_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers_overrides: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        if kwargs:
            warnings.warn(
                f"ServiceClient ignores unsupported arguments: {', '.join(sorted(kwargs))}. "
                "Passing them will be an error in a future version of the Tinker SDK.",
                FutureWarning,
                stacklevel=2,
            )
        if project_id is None:
            project_id = os.environ.get("TINKER_PROJECT_ID") or None
        self._user_metadata: dict[str, str] | None = user_metadata
        self._project_id: str | None = project_id
        self._display_name: str | None = display_name
        self._base_url: str | None = base_url
        self._api_key: str | None = api_key
        self._default_headers_overrides: dict[str, str] = dict(default_headers_overrides or {})
        self._session_holder: InternalClientHolder | None = None
        self._session_holder_lock: threading.Lock = threading.Lock()
        self._rest_holder: InternalClientHolder | None = None
        self._rest_holder_lock: threading.Lock = threading.Lock()
        self._lifecycle_lock: threading.Lock = threading.Lock()
        self._closed: bool = False

    def __enter__(self) -> Self:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("ServiceClient is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        status: Literal["success", "errored", "interrupted"]
        if exc_type is None:
            status = "success"
        elif issubclass(exc_type, Exception):
            status = "errored"
        else:
            status = "interrupted"
        self.close(status).result()

    # The unlocked fast paths below keep event-loop-thread callers from blocking
    # on the lock while another thread creates a holder.

    def _get_session_holder(self) -> InternalClientHolder:
        """Lazily create and cache the sessionful holder used by training/sampling."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("ServiceClient is closed")
            if self._session_holder is not None:
                return self._session_holder
        with self._session_holder_lock:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("ServiceClient is closed")
                if self._session_holder is None:
                    self._session_holder = InternalClientHolder(
                        user_metadata=self._user_metadata,
                        project_id=self._project_id,
                        display_name=self._display_name,
                        base_url=self._base_url,
                        api_key=self._api_key,
                        default_headers_overrides=self._default_headers_overrides,
                    )
                    logger.info(
                        f"ServiceClient initialized for session {self._session_holder._session_id}"
                    )
                return self._session_holder

    def _get_rest_holder(self) -> InternalClientHolder:
        """Lazily create and cache the session-less holder used by REST clients."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("ServiceClient is closed")
            if self._rest_holder is not None:
                return self._rest_holder
        with self._rest_holder_lock:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("ServiceClient is closed")
                if self._rest_holder is None:
                    self._rest_holder = InternalClientHolder(
                        base_url=self._base_url,
                        api_key=self._api_key,
                        default_headers_overrides=self._default_headers_overrides,
                        _skip_session=True,
                    )
                return self._rest_holder

    @property
    def holder(self) -> InternalClientHolder:
        """The sessionful holder. Deprecated: kept for backwards compatibility
        with callers that reach into ServiceClient internals."""
        return self._get_session_holder()

    def get_console_url(self) -> str:
        """Return the Tinker Console URL for this session."""
        if self._session_holder is None or self._session_holder._session_id is None:
            return sessions_console_url()
        return session_console_url(self._session_holder._session_id)

    @deprecated(
        "DEPRECATED: ServiceClient.get_server_capabilities() is deprecated and will be removed "
        "in a future tinker SDK version. Use tinker.RestClient().get_server_capabilities() "
        "instead; it returns an APIFuture: call .result() on it, or await it. RestClient needs "
        "no ServiceClient.",
        category=FutureWarning,
    )
    @sync_only
    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        """Deprecated: use `tinker.RestClient().get_server_capabilities().result()`, which
        needs no session. This method will be removed in a subsequent release."""
        return self._rest_client().get_server_capabilities().result()

    @deprecated(
        "DEPRECATED: ServiceClient.get_server_capabilities_async() is deprecated and will be "
        "removed in a future tinker SDK version. Use `await "
        "tinker.RestClient().get_server_capabilities()` instead; RestClient needs no "
        "ServiceClient.",
        category=FutureWarning,
    )
    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        """Deprecated: use `await tinker.RestClient().get_server_capabilities()`. This method will
        be removed in a subsequent release."""
        return await self._rest_client().get_server_capabilities()

    def _check_accessible(self) -> None:
        """Make a single request to a billing-gated endpoint, raising on failure.

        This deliberately skips `execute_with_retries`: an account without
        billing set up gets a 402, which that path treats as a pause-and-retry
        condition and sits on for minutes. Callers of this check want the
        error itself, so every failure propagates as an exception instead.
        """
        holder = self._get_rest_holder()

        async def _send_request() -> None:
            with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                await client.service.get_server_capabilities()

        holder.run_coroutine_threadsafe(_send_request()).result()

    def _create_lora_training_client_submit(
        self,
        base_model: str,
        rank: int,
        seed: int | None,
        train_mlp: bool,
        train_attn: bool,
        train_unembed: bool,
        user_metadata: dict[str, str] | None,
        optimizer: types.OptimizerConfig | None,
    ) -> AwaitableConcurrentFuture[TrainingClient]:
        assert any([train_mlp, train_attn, train_unembed]), (
            "At least one of train_mlp, train_attn, or train_unembed must be True"
        )
        session_id = self.holder.get_session_id()
        model_seq_id = self.holder.get_training_client_id()
        lora_config = types.LoraConfig(
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )

        @capture_exceptions(fatal=True)
        async def _create_lora_training_client_async():
            start_time = time.time()
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                request = types.CreateModelRequest(
                    session_id=session_id,
                    model_seq_id=model_seq_id,
                    base_model=base_model,
                    lora_config=lora_config,
                    user_metadata=user_metadata,
                    optimizer_config=optimizer or types.AdamOptimizerConfig(),
                )
                future = await client.models.create(request=request)
            create_model_response = await _APIFuture(
                types.CreateModelResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="CreateModel",
                queue_state_observer=QueueStateLogger(base_model, "Model creation"),
            ).result_async()
            model_id = create_model_response.model_id
            from .training_client import TrainingClient, fetch_training_info

            training_client = TrainingClient(
                self.holder,
                model_seq_id=model_seq_id,
                model_id=model_id,
                info=await fetch_training_info(self.holder, model_id),
            )
            logger.info(f"TrainingClient initialized for model {model_id}")
            return training_client

        return self.holder.run_coroutine_threadsafe(_create_lora_training_client_async())

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
        optimizer: types.OptimizerConfig | None = None,
    ) -> BackcompatTrainingClientFuture:
        """Create a TrainingClient for LoRA fine-tuning.

        Args:
        - `base_model`: Name of the base model to fine-tune (e.g., "Qwen/Qwen3-8B")
        - `rank`: LoRA rank controlling the size of adaptation matrices (default 32)
        - `seed`: Random seed for initialization. None means random seed.
        - `train_mlp`: Whether to train MLP layers (default True)
        - `train_attn`: Whether to train attention layers (default True)
        - `train_unembed`: Whether to train unembedding layers (default True)
        - `user_metadata`: Optional metadata to attach to the training run
        - `optimizer`: Fixed optimizer configuration. Defaults to AdamW.

        Returns:
        - A future of the `TrainingClient` configured for LoRA training. Await it, or
            call `.result()`.

        Example:
        ```python
        training_client = service_client.create_lora_training_client(
            base_model="Qwen/Qwen3-8B",
            rank=16,
            train_mlp=True,
            train_attn=True
        ).result()
        # Now use training_client.forward_backward() to train
        ```
        """
        from .backcompat_training_client_future import BackcompatTrainingClientFuture

        return BackcompatTrainingClientFuture(
            self._create_lora_training_client_submit(
                base_model,
                rank,
                seed,
                train_mlp,
                train_attn,
                train_unembed,
                user_metadata,
                optimizer,
            ),
        )

    @deprecated(
        "DEPRECATED: ServiceClient.create_lora_training_client_async() is deprecated and will "
        "be removed in a future tinker SDK version. Use `await "
        "ServiceClient.create_lora_training_client(...)` instead; it returns a future of the "
        "client that can be awaited.",
        category=FutureWarning,
    )
    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
        optimizer: types.OptimizerConfig | None = None,
    ) -> TrainingClient:
        """Deprecated: use `await service_client.create_lora_training_client(...)`, which returns a
        future of the client that can be awaited. This method will be removed in a
        subsequent release."""
        return await self.create_lora_training_client(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
            optimizer,
        )

    def _create_training_client_via_load_weights_submit(
        self,
        path: str,
        optimizer: bool,
        optimizer_config: types.OptimizerConfig | None,
        base_model: str | None,
        user_metadata: dict[str, str] | None,
        weights_access_token: str | None,
    ) -> AwaitableConcurrentFuture[TrainingClient]:
        """Create a TrainingClient loaded from `path` with one LoadWeightsRequest: with the
        checkpoint's optimizer state (and optimizer) when `optimizer` is set, else with
        `optimizer_config`. The server resolves the access token, so no token exchange here.
        """
        session_id = self.holder.get_session_id()
        model_seq_id = self.holder.get_training_client_id()
        # Same model id the server derives from (session_id, model_seq_id).
        model_id = f"{session_id}:train:{model_seq_id}"

        @capture_exceptions(fatal=True)
        async def _create_via_load_weights_async():
            start_time = time.time()

            async def _send_request():
                request = types.LoadWeightsRequest(
                    session_id=session_id,
                    model_seq_id=model_seq_id,
                    base_model=base_model,
                    user_metadata=user_metadata,
                    path=path,
                    optimizer=optimizer,
                    optimizer_config=optimizer_config,
                    weights_access_token=weights_access_token,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.load(request=request)

            future = await self.holder.execute_with_retries(_send_request)
            await _APIFuture(
                types.LoadWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="LoadWeights",
                queue_state_observer=QueueStateLogger(model_id, "Model creation"),
            ).result_async()
            from .training_client import TrainingClient, fetch_training_info

            training_client = TrainingClient(
                self.holder,
                model_seq_id=model_seq_id,
                model_id=model_id,
                info=await fetch_training_info(self.holder, model_id),
            )
            logger.info(f"TrainingClient initialized for model {model_id} via load_weights")
            return training_client

        return self.holder.run_coroutine_threadsafe(_create_via_load_weights_async())

    def copy_weights(
        self,
        path: str,
        *,
        ttl_seconds: int | None = None,
        weights_access_token: str | None = None,
    ) -> APIFuture[str]:
        """Copy weights into this client's project, using a new session.

        Storage is shared with the source, so no bytes are duplicated. Either kind
        of weights can be copied, and the copy keeps that kind. A new training
        run in a new session is created to hold it, which cannot be trained on.

        Args:
        - `path`: Tinker path of the weights to copy
        - `ttl_seconds`: Seconds until the copy expires, between 1 hour (3600) and 10 years,
          or None for no expiry
        - `weights_access_token`: Optional access token for copying weights readable
          under a different account

        Returns:
        - A future for the tinker path of the copy. Await it, or call `.result()`.

        Example:
        ```python
        # The copy lands in this client's project.
        archive = tinker.ServiceClient(project_id="proj-archive")
        archived_path = archive.copy_weights("tinker://run-id/weights/step-400").result()
        ```
        """
        return self._rest_client().copy_weights(
            path,
            destination_project_id=self._project_id,
            ttl_seconds=ttl_seconds,
            weights_access_token=weights_access_token,
        )

    def create_training_client_from_state(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
        optimizer: types.OptimizerConfig | None = None,
    ) -> BackcompatTrainingClientFuture:
        """Create a TrainingClient from saved model weights.

        This loads only the model weights, not optimizer state. To also restore
        optimizer state (e.g., Adam momentum), use create_training_client_from_state_with_optimizer.

        Args:
        - `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `base_model`: Optional override of the checkpoint's base model; must be
          compatible with it (e.g. a different context length)
        - `user_metadata`: Optional metadata to attach to the new training run
        - `weights_access_token`: Optional access token for loading checkpoints under a different account.
        - `optimizer`: Optimizer for the new training run; defaults to Adam.

        Returns:
        - A future of the `TrainingClient` loaded with the specified weights. Await it,
            or call `.result()`.

        Example:
        ```python
        # Resume training from a checkpoint (weights only, optimizer resets)
        training_client = service_client.create_training_client_from_state(
            "tinker://run-id/weights/checkpoint-001"
        ).result()
        # Continue training from the loaded state
        ```
        """
        from .backcompat_training_client_future import BackcompatTrainingClientFuture

        return BackcompatTrainingClientFuture(
            self._create_training_client_via_load_weights_submit(
                path,
                optimizer=False,
                optimizer_config=optimizer,
                base_model=base_model,
                user_metadata=user_metadata,
                weights_access_token=weights_access_token,
            ),
        )

    @deprecated(
        "DEPRECATED: ServiceClient.create_training_client_from_state_async() is deprecated and "
        "will be removed in a future tinker SDK version. Use `await "
        "ServiceClient.create_training_client_from_state(...)` instead; it returns a future of "
        "the client that can be awaited.",
        category=FutureWarning,
    )
    async def create_training_client_from_state_async(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
        optimizer: types.OptimizerConfig | None = None,
    ) -> TrainingClient:
        """Deprecated: use `await service_client.create_training_client_from_state(...)`, which
        returns a future of the client that can be awaited. This method will be removed
        in a subsequent release."""
        return await self.create_training_client_from_state(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=optimizer,
        )

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> BackcompatTrainingClientFuture:
        """Create a TrainingClient from saved model weights and optimizer state.

        This is similar to create_training_client_from_state but also restores
        optimizer state (e.g., Adam momentum), which is useful for resuming
        training exactly where it left off.

        Args:
        - `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `base_model`: Optional override of the checkpoint's base model; must be
          compatible with it (e.g. a different context length)
        - `user_metadata`: Optional metadata to attach to the new training run
        - `weights_access_token`: Optional access token for loading checkpoints under a different account.

        Returns:
        - A future of the `TrainingClient` loaded with the specified weights and
            optimizer state. Await it, or call `.result()`.

        Example:
        ```python
        # Resume training from a checkpoint with optimizer state
        training_client = service_client.create_training_client_from_state_with_optimizer(
            "tinker://run-id/weights/checkpoint-001"
        ).result()
        # Continue training with restored optimizer momentum
        ```
        """
        from .backcompat_training_client_future import BackcompatTrainingClientFuture

        return BackcompatTrainingClientFuture(
            self._create_training_client_via_load_weights_submit(
                path,
                optimizer=True,
                optimizer_config=None,
                base_model=base_model,
                user_metadata=user_metadata,
                weights_access_token=weights_access_token,
            ),
        )

    @deprecated(
        "DEPRECATED: ServiceClient.create_training_client_from_state_with_optimizer_async() is "
        "deprecated and will be removed in a future tinker SDK version. Use `await "
        "ServiceClient.create_training_client_from_state_with_optimizer(...)` instead; it "
        "returns a future of the client that can be awaited.",
        category=FutureWarning,
    )
    async def create_training_client_from_state_with_optimizer_async(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        """Deprecated: use `await
        service_client.create_training_client_from_state_with_optimizer(...)`, which
        returns a future of the client that can be awaited. This method will be removed
        in a subsequent release."""
        return await self.create_training_client_from_state_with_optimizer(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
        )

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
    ) -> BackcompatSamplingClientFuture:
        """Create a SamplingClient for text generation.

        Args:
        - `model_path`: Path to saved model weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `base_model`: Name of base model to use (e.g., "Qwen/Qwen3-8B")
        - `retry_config`: Optional configuration for retrying failed requests
        - `record_stability_info`: When true, every sample from this client
            records stability information for its sequences (StableReinforce
            training); recorded data is referenced from training requests by
            `sequence_id`, is session-scoped, and expires after inactivity.

        Returns:
        - A future of the `SamplingClient` configured for text generation. Await it, or
            call `.result()`.

        Raises:
            ValueError: If neither model_path nor base_model is provided

        Example:
        ```python
        # Use a base model
        sampling_client = service_client.create_sampling_client(
            base_model="Qwen/Qwen3-8B"
        ).result()

        # Or use saved weights
        sampling_client = service_client.create_sampling_client(
            model_path="tinker://run-id/weights/checkpoint-001"
        ).result()
        ```
        """
        from .backcompat_sampling_client_future import BackcompatSamplingClientFuture
        from .sampling_client import SamplingClient

        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return BackcompatSamplingClientFuture(
            SamplingClient.create(
                self.holder,
                model_path=model_path,
                base_model=base_model,
                retry_config=retry_config,
                record_stability_info=record_stability_info,
            ),
        )

    def _create_custom_loss_submit(
        self, fn: PerDatumLossFn, input_ranks: Mapping[str, types.CustomLossInputRank]
    ) -> AwaitableConcurrentFuture[LossHandle]:
        loss_handle = trace_loss(fn, input_ranks, self.holder.get_session_id())
        upload = self.holder.get_client_config().custom_loss_server_side

        @capture_exceptions
        async def _create_custom_loss_async():
            self.holder._register_custom_loss(loss_handle)
            if not upload:
                return loss_handle
            start_time = time.time()
            future = await self.holder._create_custom_loss(
                loss_handle.name,
                list(loss_handle.inputs),
                encode_graph(loss_handle),
            )
            create_custom_loss_response = await _APIFuture(
                types.CreateCustomLossResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="CreateCustomLoss",
            ).result_async()
            return dataclasses.replace(loss_handle, loss_id=create_custom_loss_response.loss_id)

        return self.holder.run_coroutine_threadsafe(_create_custom_loss_async())

    def create_custom_loss(
        self, fn: PerDatumLossFn, input_ranks: Mapping[str, types.CustomLossInputRank]
    ) -> APIFuture[LossHandle]:
        """Create a custom loss function for ``TrainingClient.forward_backward_custom``.

        Alpha: the loss is traced and validated here. Where the server supports it, it is
        then uploaded and runs on the server; otherwise it runs client-side, under the same
        constraints.

        ``fn`` runs once per datum and the per-datum losses are summed. It takes ``logprobs``
        (the datum's target logprobs) first, then one parameter per ``loss_fn_inputs`` key it
        reads, each as a tensor. It returns ``(loss, metrics)``: a scalar tensor and a dict
        of scalar tensors keyed ``custom/<name>:sum`` or ``custom/<name>:mean`` in lowercase; the
        suffix says how each metric combines across datums.

        ``input_ranks`` gives the rank of each parameter, ``logprobs`` included: 0 for a
        scalar, 1 for ``[T]`` and 2 for ``[T, K]``. ``logprobs`` has the rank of the datum's
        ``target_tokens``. Every datum passed with the loss must match.

        The function is traced with ``torch.fx`` and must be representable with torch and
        math alone: no Python branching on tensor values, no other libraries, no
        ``torch.nn`` modules, and no tensors captured from outside the function. A
        function that breaks these rules is rejected here.

        Each loss is created once per session, and its ``LossHandle`` only works with
        clients of that session. Its name is ``fn``'s; wrap ``fn`` to change it.

        Args:
        - `fn`: The loss function
        - `input_ranks`: The rank of each of ``fn``'s parameters

        Returns:
        - `APIFuture` of the `LossHandle` to pass to ``forward_backward_custom``

        Example:
        ```python
        def neg_ll(logprobs, weights):
            loss = -(logprobs * weights).sum()
            return loss, {"custom/neg_ll:sum": loss}

        loss_handle = service_client.create_custom_loss(
            neg_ll, input_ranks={"logprobs": 1, "weights": 1}
        ).result()
        result = training_client.forward_backward_custom(data, loss_handle).result()
        ```
        """
        return self._create_custom_loss_submit(fn, input_ranks)

    @deprecated(
        "DEPRECATED: ServiceClient.create_sampling_client_async() is deprecated and will be "
        "removed in a future tinker SDK version. Use `await "
        "ServiceClient.create_sampling_client(...)` instead; it returns a future of the client "
        "that can be awaited.",
        category=FutureWarning,
    )
    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
    ) -> SamplingClient:
        """Deprecated: use `await service_client.create_sampling_client(...)`, which returns a
        future of the client that can be awaited. This method will be removed in a
        subsequent release."""
        return await self.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        )

    def _rest_client(self) -> RestClient:
        """A RestClient over this client's session-less holder."""
        from .rest_client import RestClient

        return RestClient(self._get_rest_holder())

    @deprecated(
        "DEPRECATED: ServiceClient.create_rest_client() is deprecated and will be removed in a "
        "future tinker SDK version. Construct tinker.RestClient() directly instead; it takes the "
        "same base_url and api_key and needs no ServiceClient.",
        category=FutureWarning,
    )
    def create_rest_client(self) -> RestClient:
        """Deprecated: construct `RestClient()` directly instead. This method will be
        removed in a subsequent release.

        Returns:
        - `RestClient` for accessing REST API endpoints
        """
        return self._rest_client()

    def _close_submit(
        self,
        status: Literal["success", "errored", "interrupted"],
        detail: str | None,
    ) -> AwaitableConcurrentFuture[None]:
        with self._lifecycle_lock:
            if self._closed:
                return _completed_none_future()
            self._closed = True
            session_holder = self._session_holder
            cleanup_holder = session_holder or self._rest_holder

        if cleanup_holder is None:
            return _completed_none_future()

        async def _close_async() -> None:
            try:
                if session_holder is not None:
                    await _finish_session_async(session_holder)
            finally:
                await self._close_holders()

        @capture_exceptions
        async def _finish_session_async(holder: InternalClientHolder) -> None:
            _ = self  # keep `self` in the closure so capture_exceptions finds telemetry
            session_id = holder.get_session_id()

            async def _send_request() -> None:
                with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.post(
                        f"/api/v1/sessions/{session_id}/finish",
                        body={"reason": {"type": status}, "detail": detail},
                        cast_to=NoneType,
                    )

            await holder.execute_with_retries(_send_request)

        return cleanup_holder.run_coroutine_threadsafe(_close_async())

    async def _close_holders(self) -> None:
        if self._session_holder is not None:
            await self._session_holder.aclose()
        if self._rest_holder is not None:
            await self._rest_holder.aclose()

    def close(
        self,
        status: Literal["success", "errored", "interrupted"],
        detail: str | None = None,
    ) -> AwaitableConcurrentFuture[None]:
        """Finish the session and release local client resources.

        Marks the session terminal. Further operations against it will be rejected.
        A finish reason is first-wins and cannot replace an existing one.

        Also stops session heartbeats and releases HTTP/telemetry resources.

        Args:
        - `status`: `"success"`, `"errored"`, or `"interrupted"`
        - `detail`: Optional human-readable explanation

        Returns:
        - A future that completes when the session is finished. Await it, or call `.result()`.

        Example:
        ```python
        service_client.close("success", detail="training complete").result()
        ```
        """
        return self._close_submit(status, detail)

    def get_telemetry(self) -> Telemetry | None:
        # Report from whichever holder exists; don't create one just for telemetry.
        holder = self._session_holder or self._rest_holder
        return holder.get_telemetry() if holder is not None else None


def _completed_none_future() -> AwaitableConcurrentFuture[None]:
    future: ConcurrentFuture[None] = ConcurrentFuture()
    future.set_result(None)
    return AwaitableConcurrentFuture(future)
