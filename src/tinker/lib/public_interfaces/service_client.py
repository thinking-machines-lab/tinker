"""ServiceClient for Tinker API."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import _APIFuture
from ..internal_client_holder import InternalClientHolder
from ..queue_state_logger import QueueStateLogger
from ..retry_handler import RetryConfig
from ..sync_only import sync_only

if TYPE_CHECKING:
    from .rest_client import RestClient
    from .sampling_client import SamplingClient
    from .training_client import TrainingClient

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)


class ServiceClient(TelemetryProvider):
    """The ServiceClient is the main entry point for the Tinker API. It provides methods to:
    - Query server capabilities and health status
    - Generate TrainingClient instances for model training workflows
    - Generate SamplingClient instances for text generation and inference
    - Generate RestClient instances for REST API operations like listing weights

    Args:
        user_metadata: Optional metadata attached to the created session.
        project_id: Optional project ID to attach to the created session. If not
            provided, falls back to the `TINKER_PROJECT_ID` environment variable.
        **kwargs: advanced options passed to the underlying HTTP client,
                 including API keys, headers, and connection settings.

    Example:
    ```python
    # Near instant
    client = ServiceClient()

    # Takes a moment as we initialize the model and assign resources
    training_client = client.create_lora_training_client(base_model="Qwen/Qwen3-8B")

    # Near-instant
    sampling_client = client.create_sampling_client(base_model="Qwen/Qwen3-8B")

    # Near-instant
    rest_client = client.create_rest_client()
    ```
    """

    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        **kwargs: Any,
    ):
        default_headers = _get_default_headers() | kwargs.pop("default_headers", {})
        kwargs["_strict_response_validation"] = True
        kwargs["default_headers"] = default_headers
        if project_id is None:
            project_id = os.environ.get("TINKER_PROJECT_ID") or None
        self._user_metadata: dict[str, str] | None = user_metadata
        self._project_id: str | None = project_id
        self._holder_kwargs: dict[str, Any] = kwargs
        self._session_holder: InternalClientHolder | None = None
        self._session_holder_lock: threading.Lock = threading.Lock()
        self._rest_holder: InternalClientHolder | None = None
        self._rest_holder_lock: threading.Lock = threading.Lock()

    # The unlocked fast paths below keep event-loop-thread callers from blocking
    # on the lock while another thread creates a holder.

    def _get_session_holder(self) -> InternalClientHolder:
        """Lazily create and cache the sessionful holder used by training/sampling."""
        if self._session_holder is not None:
            return self._session_holder
        with self._session_holder_lock:
            if self._session_holder is None:
                self._session_holder = InternalClientHolder(
                    user_metadata=self._user_metadata,
                    project_id=self._project_id,
                    **self._holder_kwargs,
                )
                logger.info(
                    f"ServiceClient initialized for session {self._session_holder._session_id}"
                )
            return self._session_holder

    def _get_rest_holder(self) -> InternalClientHolder:
        """Lazily create and cache the session-less holder used by REST clients."""
        if self._rest_holder is not None:
            return self._rest_holder
        with self._rest_holder_lock:
            if self._rest_holder is None:
                self._rest_holder = InternalClientHolder(_skip_session=True, **self._holder_kwargs)
            return self._rest_holder

    @property
    def holder(self) -> InternalClientHolder:
        """The sessionful holder. Deprecated: kept for backwards compatibility
        with callers that reach into ServiceClient internals."""
        return self._get_session_holder()

    def _get_server_capabilities_submit(
        self,
    ) -> AwaitableConcurrentFuture[types.GetServerCapabilitiesResponse]:
        # Resolve before scheduling: lazy holder creation must not run on the event loop.
        holder = self._get_rest_holder()

        @capture_exceptions(fatal=True)
        async def _get_server_capabilities_async():
            _ = self  # keep `self` in the closure so capture_exceptions finds telemetry

            async def _send_request():
                with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.service.get_server_capabilities()

            return await holder.execute_with_retries(_send_request)

        return holder.run_coroutine_threadsafe(_get_server_capabilities_async())

    @sync_only
    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        """Query the server's supported features and capabilities.

        Returns:
        - `GetServerCapabilitiesResponse` with available models, features, and limits

        Example:
        ```python
        capabilities = service_client.get_server_capabilities()
        print(f"Supported models: {capabilities.supported_models}")
        print(f"Max batch size: {capabilities.max_batch_size}")
        ```
        """
        return self._get_server_capabilities_submit().result()

    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        """Async version of get_server_capabilities."""
        return await self._get_server_capabilities_submit()

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
            from .training_client import TrainingClient

            training_client = TrainingClient(
                self.holder, model_seq_id=model_seq_id, model_id=model_id
            )
            logger.info(f"TrainingClient initialized for model {model_id}")
            return training_client

        return self.holder.run_coroutine_threadsafe(_create_lora_training_client_async())

    @sync_only
    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        """Create a TrainingClient for LoRA fine-tuning.

        Args:
        - `base_model`: Name of the base model to fine-tune (e.g., "Qwen/Qwen3-8B")
        - `rank`: LoRA rank controlling the size of adaptation matrices (default 32)
        - `seed`: Random seed for initialization. None means random seed.
        - `train_mlp`: Whether to train MLP layers (default True)
        - `train_attn`: Whether to train attention layers (default True)
        - `train_unembed`: Whether to train unembedding layers (default True)
        - `user_metadata`: Optional metadata to attach to the training run

        Returns:
        - `TrainingClient` configured for LoRA training

        Example:
        ```python
        training_client = service_client.create_lora_training_client(
            base_model="Qwen/Qwen3-8B",
            rank=16,
            train_mlp=True,
            train_attn=True
        )
        # Now use training_client.forward_backward() to train
        ```
        """
        return self._create_lora_training_client_submit(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
        ).result()

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        """Async version of create_lora_training_client."""
        return await self._create_lora_training_client_submit(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
        ).result_async()

    def _get_rest_client_for_weights(self, weights_access_token: str | None = None) -> RestClient:
        """Get a rest client for weights info lookups.

        If weights_access_token is provided, creates a separate ServiceClient
        authenticated with that token.
        """
        if weights_access_token is None:
            return self.create_rest_client()

        token_client_kwargs: dict[str, Any] = {
            **self._holder_kwargs,
            "api_key": weights_access_token,
        }
        token_client = ServiceClient(**token_client_kwargs)
        return token_client.create_rest_client()

    def _create_training_client_via_load_weights_submit(
        self,
        path: str,
        optimizer: bool,
        base_model: str | None,
        user_metadata: dict[str, str] | None,
        weights_access_token: str | None,
    ) -> AwaitableConcurrentFuture[TrainingClient]:
        """Create a TrainingClient with a single LoadWeightsRequest."""
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
            from .training_client import TrainingClient

            training_client = TrainingClient(
                self.holder, model_seq_id=model_seq_id, model_id=model_id
            )
            logger.info(f"TrainingClient initialized for model {model_id} via load_weights")
            return training_client

        return self.holder.run_coroutine_threadsafe(_create_via_load_weights_async())

    def _copy_weights_submit(
        self,
        path: str,
        ttl_seconds: int | None,
        weights_access_token: str | None,
    ) -> AwaitableConcurrentFuture[str]:
        """Copy weights with a single CopyWeightsRequest."""
        session_id = self.holder.get_session_id()
        model_seq_id = self.holder.get_training_client_id()

        @capture_exceptions(fatal=True)
        async def _copy_weights_async() -> str:
            request = types.CopyWeightsRequest(
                session_id=session_id,
                model_seq_id=model_seq_id,
                source_path=path,
                ttl_seconds=ttl_seconds,
                weights_access_token=weights_access_token,
            )

            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.models.copy_weights(request=request)

            # No _APIFuture: copy is synchronous, since nothing is queued.
            response = await self.holder.execute_with_retries(_send_request)
            logger.info(f"Copied {path} to {response.tinker_path}")
            return response.tinker_path

        return self.holder.run_coroutine_threadsafe(_copy_weights_async())

    def copy_weights(
        self,
        path: str,
        *,
        ttl_seconds: int | None = None,
        weights_access_token: str | None = None,
    ) -> AwaitableConcurrentFuture[str]:
        """Copy weights into this client's project.

        Storage is shared with the source, so no bytes are duplicated. Either kind
        of weights can be copied, and the copy keeps that kind. A new training
        run is created to hold it, which cannot be trained on.

        Args:
        - `path`: Tinker path of the weights to copy
        - `ttl_seconds`: Seconds until the copy expires, or None for no expiry
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
        return self._copy_weights_submit(path, ttl_seconds, weights_access_token)

    @sync_only
    def create_training_client_from_state(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        """Create a TrainingClient from saved model weights.

        This loads only the model weights, not optimizer state. To also restore
        optimizer state (e.g., Adam momentum), use create_training_client_from_state_with_optimizer.

        Args:
        - `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `base_model`: Optional override of the checkpoint's base model; must be
          compatible with it (e.g. a different context length)
        - `user_metadata`: Optional metadata to attach to the new training run
        - `weights_access_token`: Optional access token for loading checkpoints under a different account.

        Returns:
        - `TrainingClient` loaded with the specified weights

        Example:
        ```python
        # Resume training from a checkpoint (weights only, optimizer resets)
        training_client = service_client.create_training_client_from_state(
            "tinker://run-id/weights/checkpoint-001"
        )
        # Continue training from the loaded state
        ```
        """
        if self.holder._client_config.create_model_via_load_weights:
            return self._create_training_client_via_load_weights_submit(
                path,
                optimizer=False,
                base_model=base_model,
                user_metadata=user_metadata,
                weights_access_token=weights_access_token,
            ).result()

        rest_client = self._get_rest_client_for_weights(weights_access_token)
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = rest_client.get_weights_info_by_tinker_path(path).result()

        training_client = self.create_lora_training_client(
            base_model=base_model or weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        auth_token = (
            rest_client.holder.run_coroutine_threadsafe(
                rest_client.holder._default_auth.get_token()
            ).result()
            if weights_access_token is not None
            else None
        )

        training_client.load_state(path, weights_access_token=auth_token).result()
        return training_client

    async def create_training_client_from_state_async(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        """Async version of create_training_client_from_state."""
        if self.holder._client_config.create_model_via_load_weights:
            return await self._create_training_client_via_load_weights_submit(
                path,
                optimizer=False,
                base_model=base_model,
                user_metadata=user_metadata,
                weights_access_token=weights_access_token,
            ).result_async()

        rest_client = self._get_rest_client_for_weights(weights_access_token)
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = await rest_client.get_weights_info_by_tinker_path(path)

        # Right now all training runs are LoRa runs.
        assert weights_info.is_lora and weights_info.lora_rank is not None

        training_client = await self.create_lora_training_client_async(
            base_model=base_model or weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        load_future = await training_client.load_state_async(
            path, weights_access_token=weights_access_token
        )
        await load_future.result_async()
        return training_client

    @sync_only
    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
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
        - `TrainingClient` loaded with the specified weights and optimizer state

        Example:
        ```python
        # Resume training from a checkpoint with optimizer state
        training_client = service_client.create_training_client_from_state_with_optimizer(
            "tinker://run-id/weights/checkpoint-001"
        )
        # Continue training with restored optimizer momentum
        ```
        """
        if self.holder._client_config.create_model_via_load_weights:
            return self._create_training_client_via_load_weights_submit(
                path,
                optimizer=True,
                base_model=base_model,
                user_metadata=user_metadata,
                weights_access_token=weights_access_token,
            ).result()

        rest_client = self._get_rest_client_for_weights(weights_access_token)
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = rest_client.get_weights_info_by_tinker_path(path).result()

        training_client = self.create_lora_training_client(
            base_model=base_model or weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        training_client.load_state_with_optimizer(
            path, weights_access_token=weights_access_token
        ).result()
        return training_client

    async def create_training_client_from_state_with_optimizer_async(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        """Async version of create_training_client_from_state_with_optimizer."""
        if self.holder._client_config.create_model_via_load_weights:
            return await self._create_training_client_via_load_weights_submit(
                path,
                optimizer=True,
                base_model=base_model,
                user_metadata=user_metadata,
                weights_access_token=weights_access_token,
            ).result_async()

        rest_client = self._get_rest_client_for_weights(weights_access_token)
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = await rest_client.get_weights_info_by_tinker_path(path)

        # Right now all training runs are LoRa runs.
        assert weights_info.is_lora and weights_info.lora_rank is not None

        training_client = await self.create_lora_training_client_async(
            base_model=base_model or weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        load_future = await training_client.load_state_with_optimizer_async(
            path, weights_access_token=weights_access_token
        )
        await load_future.result_async()
        return training_client

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
    ) -> SamplingClient:
        """Create a SamplingClient for text generation.

        Args:
        - `model_path`: Path to saved model weights (e.g., "tinker://run-id/weights/checkpoint-001")
        - `base_model`: Name of base model to use (e.g., "Qwen/Qwen3-8B")
        - `retry_config`: Optional configuration for retrying failed requests

        Returns:
        - `SamplingClient` configured for text generation

        Raises:
            ValueError: If neither model_path nor base_model is provided

        Example:
        ```python
        # Use a base model
        sampling_client = service_client.create_sampling_client(
            base_model="Qwen/Qwen3-8B"
        )

        # Or use saved weights
        sampling_client = service_client.create_sampling_client(
            model_path="tinker://run-id/weights/checkpoint-001"
        )
        ```
        """
        from .sampling_client import SamplingClient

        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return SamplingClient.create(
            self.holder,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        ).result()

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
    ) -> SamplingClient:
        """Async version of create_sampling_client."""
        from .sampling_client import SamplingClient

        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return await SamplingClient.create(
            self.holder,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
            record_stability_info=record_stability_info,
        )

    def create_rest_client(self) -> RestClient:
        """Create a RestClient for REST API operations.

        The RestClient provides access to various REST endpoints for querying
        model information, checkpoints, sessions, and managing checkpoint visibility.

        Returns:
        - `RestClient` for accessing REST API endpoints

        Example:
        ```python
        rest_client = service_client.create_rest_client()

        # List checkpoints for a training run
        checkpoints = rest_client.list_checkpoints("run-id").result()

        # Get training run info
        training_run = rest_client.get_training_run("run-id").result()

        # Publish a checkpoint
        rest_client.publish_checkpoint_from_tinker_path(
            "tinker://run-id/weights/checkpoint-001"
        ).result()
        ```
        """
        from .rest_client import RestClient

        return RestClient(self._get_rest_holder())

    def get_telemetry(self) -> Telemetry | None:
        # Report from whichever holder exists; don't create one just for telemetry.
        holder = self._session_holder or self._rest_holder
        return holder.get_telemetry() if holder is not None else None


def _get_default_headers() -> dict[str, str]:
    headers = {}

    if (api_key := os.environ.get("TINKER_API_KEY", "")) and "X-API-Key" not in headers:
        headers["X-API-Key"] = api_key

    if (
        client_id := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    ) and "CF-Access-Client-Id" not in headers:
        headers["CF-Access-Client-Id"] = client_id
    if (
        client_secret := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    ) and "CF-Access-Client-Secret" not in headers:
        headers["CF-Access-Client-Secret"] = client_secret
    return headers
