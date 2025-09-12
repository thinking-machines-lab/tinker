"""RestClient for Tinker API REST operations."""

from __future__ import annotations

import logging
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING

from tinker import types
from tinker.lib.async_tinker_provider import ClientConnectionPoolType
from tinker.lib.telemetry import Telemetry, TelemetryProvider, capture_exceptions

from ..sync_only import sync_only
from .api_future import AwaitableConcurrentFuture

if TYPE_CHECKING:
    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)


class RestClient(TelemetryProvider):
    """Client for REST API operations like listing checkpoints and metadata.

    The RestClient provides access to various REST endpoints for querying
    model information, checkpoints, and other resources. You typically get one
    by calling `service_client.create_rest_client()`.

    Key methods:
    - list_checkpoints() - list available model checkpoints (both training and sampler)
    - get_training_run() - get model information and metadata as ModelEntry
    - download_sampler_weights_archive() - download sampler weights checkpoint as tar.gz archive

    Args:
        holder: Internal client managing HTTP connections and async operations

    Example:
        >>> rest_client = service_client.create_rest_client()
        >>> training_run = rest_client.get_training_run("run-id").result()
        >>> print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
        >>> checkpoints = rest_client.list_checkpoints("run-id").result()
        >>> print(f"Found {len(checkpoints.checkpoints)} checkpoints")
        >>> for checkpoint in checkpoints.checkpoints:
        ...     print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
    """

    def __init__(self, holder: InternalClientHolder):
        self.holder = holder

    def _get_training_run_submit(
        self, training_run_id: types.ModelID
    ) -> AwaitableConcurrentFuture[types.TrainingRun]:
        """Internal method to submit get model request."""
        async def _get_training_run_async() -> types.TrainingRun:
            async def _send_request() -> types.TrainingRun:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/training_runs/{training_run_id}",
                        cast_to=types.TrainingRun,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_training_run_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_training_run(self, training_run_id: types.ModelID) -> ConcurrentFuture[types.TrainingRun]:
        """Get training run info.

        Args:
            training_run_id: The training run ID to get information for

        Returns:
            A Future containing the training run information

        Example:
            >>> future = rest_client.get_training_run("run-id")
            >>> response = future.result()
            >>> print(f"Model: {response.training_run_id}, Base: {response.base_model}")
        """
        return self._get_training_run_submit(training_run_id).future()

    @capture_exceptions(fatal=True)
    async def get_training_run_async(self, training_run_id: types.ModelID) -> types.TrainingRun:
        """Async version of get_training_run.

        Args:
            training_run_id: The training run ID to get information for

        Returns:
            Training run information

        Example:
            >>> response = await rest_client.get_training_run_async("run-id")
            >>> print(f"Model: {response.training_run_id}, Base: {response.base_model}")
        """
        return await self._get_training_run_submit(training_run_id)

    def _list_training_run_ids_submit(
        self, limit: int = 100, after_id: str | None = None
    ) -> AwaitableConcurrentFuture[types.TrainingRunIdsResponse]:
        """Internal method to submit list training run IDs request."""
        async def _list_training_run_ids_async() -> types.TrainingRunIdsResponse:
            async def _send_request() -> types.TrainingRunIdsResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    params: dict[str, object] = {"limit": limit}
                    if after_id is not None:
                        params["after_id"] = after_id

                    return await client.get(
                        "/api/v1/training_runs",
                        options={"params": params},
                        cast_to=types.TrainingRunIdsResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_list_training_run_ids_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def list_training_run_ids(
        self, limit: int = 100, after_id: str | None = None
    ) -> ConcurrentFuture[types.TrainingRunIdsResponse]:
        """List training run IDs with pagination support.

        Args:
            limit: Maximum number of training run IDs to return (default 100)
            after_id: Optional cursor for pagination - returns IDs after this ID

        Returns:
            A Future containing the TrainingRunIdsResponse with training run IDs and pagination info

        Example:
            >>> future = rest_client.list_training_run_ids(limit=50)
            >>> response = future.result()
            >>> print(f"Found {len(response.training_run_ids)} training runs")
            >>> if response.has_more:
            ...     last_id = response.training_run_ids[-1]
            ...     next_page = rest_client.list_training_run_ids(limit=50, after_id=last_id)
        """
        return self._list_training_run_ids_submit(limit, after_id).future()

    @capture_exceptions(fatal=True)
    async def list_training_run_ids_async(
        self, limit: int = 100, after_id: str | None = None
    ) -> types.TrainingRunIdsResponse:
        """Async version of list_training_run_ids.

        Args:
            limit: Maximum number of training run IDs to return (default 100)
            after_id: Optional cursor for pagination - returns IDs after this ID

        Returns:
            TrainingRunIdsResponse with training run IDs and pagination info

        Example:
            >>> response = await rest_client.list_training_run_ids_async(limit=50)
            >>> print(f"Found {len(response.training_run_ids)} training runs")
            >>> if response.has_more:
            ...     last_id = response.training_run_ids[-1]
            ...     next_page = await rest_client.list_training_run_ids_async(limit=50, after_id=last_id)
        """
        return await self._list_training_run_ids_submit(limit, after_id)

    def _list_checkpoints_submit(
        self, training_run_id: types.ModelID
    ) -> AwaitableConcurrentFuture[types.CheckpointsListResponse]:
        """Internal method to submit list model checkpoints request."""
        async def _list_checkpoints_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.list(training_run_id)

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_list_checkpoints_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def list_checkpoints(self, training_run_id: types.ModelID) -> ConcurrentFuture[types.CheckpointsListResponse]:
        """List available checkpoints (both training and sampler).

        Args:
            training_run_id: The training run ID to list checkpoints for

        Returns:
            A Future containing the CheckpointsListResponse with available checkpoints

        Example:
            >>> future = rest_client.list_checkpoints("run-id")
            >>> response = future.result()
            >>> for checkpoint in response.checkpoints:
            ...     if checkpoint.checkpoint_type == "training":
            ...         print(f"Training checkpoint: {checkpoint.checkpoint_id}")
            ...     elif checkpoint.checkpoint_type == "sampler":
            ...         print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
        """
        return self._list_checkpoints_submit(training_run_id).future()

    @capture_exceptions(fatal=True)
    async def list_checkpoints_async(self, training_run_id: types.ModelID) -> types.CheckpointsListResponse:
        """Async version of list_checkpoints.

        Args:
            training_run_id: The training run ID to list checkpoints for

        Returns:
            CheckpointsListResponse with available checkpoints

        Example:
            >>> response = await rest_client.list_checkpoints_async("run-id")
            >>> for checkpoint in response.checkpoints:
            ...     if checkpoint.checkpoint_type == "training":
            ...         print(f"Training checkpoint: {checkpoint.checkpoint_id}")
            ...     elif checkpoint.checkpoint_type == "sampler":
            ...         print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
        """
        return await self._list_checkpoints_submit(training_run_id)

    def _download_checkpoint_archive_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> AwaitableConcurrentFuture[bytes]:
        """Internal method to submit download checkpoint archive request."""
        async def _download_checkpoint_archive_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}/archive",
                        cast_to=bytes,
                        options={"headers": {"accept": "application/gzip"}},
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_download_checkpoint_archive_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def download_checkpoint_archive(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> ConcurrentFuture[bytes]:
        """Download checkpoint as a tar.gz archive.

        Args:
            training_run_id: The training run ID to download weights for
            checkpoint_id: The checkpoint ID to download

        Returns:
            A Future containing the archive data as bytes

        Example:
            >>> future = rest_client.download_checkpoint_archive("run-id", "checkpoint-123")
            >>> archive_data = future.result()
            >>> with open(f"model-checkpoint.tar.gz", "wb") as f:
            ...     f.write(archive_data)
        """
        return self._download_checkpoint_archive_submit(training_run_id, checkpoint_id).future()

    @capture_exceptions(fatal=True)
    async def download_checkpoint_archive_async(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> bytes:
        """Async version of download_checkpoint_archive.

        Args:
            training_run_id: The model ID to download weights for
            checkpoint_id: The checkpoint ID to download

        Returns:
            Archive data as bytes

        Example:
            >>> archive_data = await rest_client.download_checkpoint_archive_async("run-id", "checkpoint-123")
            >>> with open(f"model-checkpoint.tar.gz", "wb") as f:
            ...     f.write(archive_data)
        """
        return await self._download_checkpoint_archive_submit(training_run_id, checkpoint_id)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()

    @sync_only
    @capture_exceptions(fatal=True)
    def download_checkpoint_archive_from_tinker_path(
        self, tinker_path: str
    ) -> ConcurrentFuture[bytes]:
        """Download checkpoint as a tar.gz archive.

        Args:
            tinker_path: The tinker path to the checkpoint

        Returns:
            A Future containing the archive data as bytes
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._download_checkpoint_archive_submit(parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id).future()

    @capture_exceptions(fatal=True)
    async def download_checkpoint_archive_from_tinker_path_async(
        self, tinker_path: str
    ) -> bytes:
        """Async version of download_checkpoint_archive_from_tinker_path.

        Args:
            tinker_path: The tinker path to the checkpoint
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return await self._download_checkpoint_archive_submit(parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id)
