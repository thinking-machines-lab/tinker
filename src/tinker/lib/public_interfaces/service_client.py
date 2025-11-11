"""ServiceClient for Tinker API."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from tinker import types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import _APIFuture
from ..internal_client_holder import InternalClientHolder
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
        **kwargs: advanced options passed to the underlying HTTP client,
                 including API keys, headers, and connection settings.

    Example:
        >>> client = ServiceClient()
            # ^^^ near-instant
        >>> training_client = client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
            # ^^^ takes a moment as we initialize the model and assign resources
        >>> sampling_client = client.create_sampling_client(base_model="Qwen/Qwen3-8B")
            # ^^^ near-instant
        >>> rest_client = client.create_rest_client()
            # ^^^ near-instant
    """

    def __init__(self, user_metadata: dict[str, str] | None = None, **kwargs: Any):
        default_headers = _get_default_headers() | kwargs.pop("default_headers", {})
        self.holder = InternalClientHolder(
            user_metadata=user_metadata,
            **kwargs,
            default_headers=default_headers,
            _strict_response_validation=True,
        )
        logger.info(f"ServiceClient initialized for session {self.holder._session_id}")

    def _get_server_capabilities_submit(
        self,
    ) -> AwaitableConcurrentFuture[types.GetServerCapabilitiesResponse]:
        async def _get_server_capabilities_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.service.get_server_capabilities()

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_server_capabilities_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        return self._get_server_capabilities_submit().result()

    @capture_exceptions(fatal=True)
    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        return await self._get_server_capabilities_submit()

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
            ).result_async()
            model_id = create_model_response.model_id
            logger.info(f"Creating TrainingClient for {model_id=}")
            from .training_client import TrainingClient

            return TrainingClient(self.holder, model_seq_id=model_seq_id, model_id=model_id)

        return self.holder.run_coroutine_threadsafe(_create_lora_training_client_async())

    @sync_only
    @capture_exceptions(fatal=True)
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
        return self._create_lora_training_client_submit(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
        ).result()

    @capture_exceptions(fatal=True)
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
        return await self._create_lora_training_client_submit(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
        ).result_async()

    @sync_only
    @capture_exceptions(fatal=True)
    def create_training_client_from_state(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> TrainingClient:
        rest_client = self.create_rest_client()
        training_run = rest_client.get_training_run_by_tinker_path(path).result()

        # Merge user metadata dicts
        user_metdata = {**(training_run.user_metadata or {}), **(user_metadata or {})}

        training_client = self.create_lora_training_client(
            base_model=training_run.base_model,
            rank=training_run.lora_rank,
            user_metadata=user_metdata,
        )

        training_client.load_state(path).result()
        return training_client

    @capture_exceptions(fatal=True)
    async def create_training_client_from_state_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> TrainingClient:
        rest_client = self.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(path)

        # Right now all training runs are LoRa runs.
        assert training_run.is_lora and training_run.lora_rank is not None
        # Merge user metadata dicts
        user_metdata = {**(training_run.user_metadata or {}), **(user_metadata or {})}

        training_client = await self.create_lora_training_client_async(
            base_model=training_run.base_model,
            rank=training_run.lora_rank,
            user_metadata=user_metdata,
        )

        load_future = await training_client.load_state_async(path)
        await load_future.result_async()
        return training_client

    @capture_exceptions(fatal=True)
    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> SamplingClient:
        from .sampling_client import SamplingClient

        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return SamplingClient(
            self.holder,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )

    @capture_exceptions(fatal=True)
    def create_rest_client(self) -> RestClient:
        """Create a RestClient for REST API operations.

        Returns:
            RestClient: A client for listing weights and other REST operations

        Example:
            >>> rest_client = service_client.create_rest_client()
            >>> weights = rest_client.list_model_weights("my-model-id").result()
        """
        from .rest_client import RestClient

        return RestClient(self.holder)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


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
