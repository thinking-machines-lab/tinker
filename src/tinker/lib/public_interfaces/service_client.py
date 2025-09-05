"""ServiceClient for Tinker API."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, cast

from tinker import types
from tinker.lib.telemetry import Telemetry, TelemetryProvider, capture_exceptions

from ..internal_client_holder import InternalClientHolder
from ..retry_handler import RetryConfig
from ..sync_only import sync_only
from .api_future import AwaitableConcurrentFuture, _APIFuture

if TYPE_CHECKING:
    from .sampling_client import SamplingClient
    from .training_client import TrainingClient

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)


class ServiceClient(TelemetryProvider):
    """The ServiceClient is the main entry point for the Tinker API. It provides methods to:
    - Query server capabilities and health status
    - Generate TrainingClient instances for model training workflows
    - Generate SamplingClient instances for text generation and inference

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
    """

    def __init__(self, **kwargs: Any):
        default_headers = _get_default_headers() | kwargs.pop("default_headers", {})
        self.holder = InternalClientHolder(
            **kwargs, default_headers=default_headers, _strict_response_validation=True
        )

    def _get_server_capabilities_submit(
        self,
    ) -> AwaitableConcurrentFuture[types.GetServerCapabilitiesResponse]:
        async def _get_server_capabilities_async():
            async def _send_request():
                with self.holder.aclient() as client:
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

    def _create_model_submit(
        self, base_model: str, lora_config: types.LoraConfig
    ) -> AwaitableConcurrentFuture[types.ModelID]:
        async def _create_model_async():
            start_time = time.time()
            with self.holder.aclient() as client:
                future = await client.models.create(
                    base_model=base_model, lora_config=_to_lora_config_params(lora_config)
                )
            create_model_response = await _APIFuture(
                types.CreateModelResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="CreateModel",
            ).result_async()
            return create_model_response.model_id

        return self.holder.run_coroutine_threadsafe(_create_model_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def create_lora_training_client(
        self, base_model: str, rank: int = 32, seed: int | None = None
    ) -> TrainingClient:
        model_id = self._create_model_submit(
            base_model, types.LoraConfig(rank=rank, seed=seed)
        ).result()
        logger.info(f"Creating TrainingClient for {model_id=}")
        return self.create_training_client(model_id)

    @capture_exceptions(fatal=True)
    async def create_lora_training_client_async(
        self, base_model: str, rank: int = 32, seed: int | None = None
    ) -> TrainingClient:
        model_id = await self._create_model_submit(
            base_model, types.LoraConfig(rank=rank, seed=seed)
        )
        logger.info(f"Creating TrainingClient for {model_id=}")
        return self.create_training_client(model_id)

    @capture_exceptions(fatal=True)
    def create_training_client(self, model_id: types.ModelID | None = None) -> TrainingClient:
        from .training_client import TrainingClient

        return TrainingClient(self.holder, model_id=model_id)

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

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


def _get_default_headers() -> dict[str, str]:
    headers = {}

    if (api_key := os.environ.get("TINKER_API_KEY", "")) and "X-API-Key" not in headers:
        headers["X-API-Key"] = api_key

    headers["X-Username"] = os.environ.get("USER", "")

    if (
        client_id := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    ) and "CF-Access-Client-Id" not in headers:
        headers["CF-Access-Client-Id"] = client_id
    if (
        client_secret := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    ) and "CF-Access-Client-Secret" not in headers:
        headers["CF-Access-Client-Secret"] = client_secret
    return headers


def _to_lora_config_params(x: types.LoraConfig) -> types._LoraConfigParam:
    return cast(types._LoraConfigParam, x.model_dump())
