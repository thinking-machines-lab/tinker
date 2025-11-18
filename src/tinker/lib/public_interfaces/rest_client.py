"""RestClient for Tinker API REST operations."""

from __future__ import annotations

import logging
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING

from tinker import NoneType, types
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..sync_only import sync_only

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
    - list_user_checkpoints() - list all checkpoints across all user's training runs
    - get_training_run() - get model information and metadata as ModelEntry
    - delete_checkpoint() - delete an existing checkpoint for a training run
    - get_checkpoint_archive_url() - get signed URL to download checkpoint archive
    - publish_checkpoint_from_tinker_path() - publish a checkpoint to make it public
    - unpublish_checkpoint_from_tinker_path() - unpublish a checkpoint to make it private

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
    def get_training_run(
        self, training_run_id: types.ModelID
    ) -> ConcurrentFuture[types.TrainingRun]:
        """Get training run info.

        Args:
            training_run_id: The training run ID to get information for

        Returns:
            A Future containing the training run information

        Example:
            >>> future = rest_client.get_training_run("run-id")
            >>> response = future.result()
            >>> print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
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
            >>> print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
        """
        return await self._get_training_run_submit(training_run_id)

    @sync_only
    @capture_exceptions(fatal=True)
    def get_training_run_by_tinker_path(
        self, tinker_path: str
    ) -> ConcurrentFuture[types.TrainingRun]:
        """Get training run info.

        Args:
            tinker_path: The tinker path to the checkpoint

        Returns:
            A Future containing the training run information

        Example:
            >>> future = rest_client.get_training_run_by_tinker_path("tinker://run-id/weights/checkpoint-001")
            >>> response = future.result()
            >>> print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
        """
        parsed_checkpoint_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(
            tinker_path
        )
        return self.get_training_run(parsed_checkpoint_tinker_path.training_run_id)

    @capture_exceptions(fatal=True)
    async def get_training_run_by_tinker_path_async(self, tinker_path: str) -> types.TrainingRun:
        """Async version of get_training_run.

        Args:
            tinker_path: The tinker path to the checkpoint

        Returns:
            Training run information

        Example:
            >>> response = await rest_client.get_training_run_by_tinker_path_async("tinker://run-id/weights/checkpoint-001")
            >>> print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
        """
        parsed_checkpoint_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(
            tinker_path
        )
        return await self.get_training_run_async(parsed_checkpoint_tinker_path.training_run_id)

    def _list_training_runs_submit(
        self, limit: int = 20, offset: int = 0
    ) -> AwaitableConcurrentFuture[types.TrainingRunsResponse]:
        """Internal method to submit list training runs request."""

        async def _list_training_runs_async() -> types.TrainingRunsResponse:
            async def _send_request() -> types.TrainingRunsResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    params: dict[str, object] = {"limit": limit, "offset": offset}

                    return await client.get(
                        "/api/v1/training_runs",
                        options={"params": params},
                        cast_to=types.TrainingRunsResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_list_training_runs_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def list_training_runs(
        self, limit: int = 20, offset: int = 0
    ) -> ConcurrentFuture[types.TrainingRunsResponse]:
        """List training runs with pagination support.

        Args:
            limit: Maximum number of training runs to return (default 20)
            offset: Offset for pagination (default 0)

        Returns:
            A Future containing the TrainingRunsResponse with training runs and cursor info

        Example:
            >>> future = rest_client.list_training_runs(limit=50)
            >>> response = future.result()
            >>> print(f"Found {len(response.training_runs)} training runs")
            >>> print(f"Total: {response.cursor.total_count}")
            >>> # Get next page
            >>> next_page = rest_client.list_training_runs(limit=50, offset=50)
        """
        return self._list_training_runs_submit(limit, offset).future()

    @capture_exceptions(fatal=True)
    async def list_training_runs_async(
        self, limit: int = 20, offset: int = 0
    ) -> types.TrainingRunsResponse:
        """Async version of list_training_runs.

        Args:
            limit: Maximum number of training runs to return (default 20)
            offset: Offset for pagination (default 0)

        Returns:
            TrainingRunsResponse with training runs and cursor info

        Example:
            >>> response = await rest_client.list_training_runs_async(limit=50)
            >>> print(f"Found {len(response.training_runs)} training runs")
            >>> print(f"Total: {response.cursor.total_count}")
            >>> # Get next page
            >>> next_page = await rest_client.list_training_runs_async(limit=50, offset=50)
        """
        return await self._list_training_runs_submit(limit, offset)

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
    def list_checkpoints(
        self, training_run_id: types.ModelID
    ) -> ConcurrentFuture[types.CheckpointsListResponse]:
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
    async def list_checkpoints_async(
        self, training_run_id: types.ModelID
    ) -> types.CheckpointsListResponse:
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

    def _get_checkpoint_archive_url_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> AwaitableConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        """Internal method to submit get checkpoint archive URL request."""

        async def _get_checkpoint_archive_url_async() -> types.CheckpointArchiveUrlResponse:
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.weights.get_checkpoint_archive_url(
                    model_id=training_run_id,
                    checkpoint_id=checkpoint_id,
                )

        return self.holder.run_coroutine_threadsafe(_get_checkpoint_archive_url_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_checkpoint_archive_url(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        """Get signed URL to download checkpoint archive.

        Args:
            training_run_id: The training run ID to download weights for
            checkpoint_id: The checkpoint ID to download

        Returns:
            A Future containing the CheckpointArchiveUrlResponse with signed URL and expiration

        Example:
            >>> future = rest_client.get_checkpoint_archive_url("run-id", "checkpoint-123")
            >>> response = future.result()
            >>> print(f"Download URL: {response.url}")
            >>> print(f"Expires at: {response.expires_at}")
            >>> # Use the URL to download the archive with your preferred HTTP client
        """
        return self._get_checkpoint_archive_url_submit(training_run_id, checkpoint_id).future()

    @capture_exceptions(fatal=True)
    async def get_checkpoint_archive_url_async(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> types.CheckpointArchiveUrlResponse:
        """Async version of get_checkpoint_archive_url.

        Args:
            training_run_id: The model ID to download weights for
            checkpoint_id: The checkpoint ID to download

        Returns:
            CheckpointArchiveUrlResponse with signed URL and expiration

        Example:
            >>> response = await rest_client.get_checkpoint_archive_url_async("run-id", "checkpoint-123")
            >>> print(f"Download URL: {response.url}")
            >>> print(f"Expires at: {response.expires_at}")
            >>> # Use the URL to download the archive with your preferred HTTP client
        """
        return await self._get_checkpoint_archive_url_submit(training_run_id, checkpoint_id)

    def _delete_checkpoint_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> AwaitableConcurrentFuture[None]:
        """Internal method to submit delete checkpoint request."""

        async def _delete_checkpoint_async() -> None:
            async def _send_request() -> None:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.delete(
                        f"/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}",
                        cast_to=NoneType,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_delete_checkpoint_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def delete_checkpoint(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> ConcurrentFuture[None]:
        """Delete a checkpoint for a training run."""

        return self._delete_checkpoint_submit(training_run_id, checkpoint_id).future()

    @capture_exceptions(fatal=True)
    async def delete_checkpoint_async(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> None:
        """Async version of delete_checkpoint."""

        await self._delete_checkpoint_submit(training_run_id, checkpoint_id)

    @sync_only
    @capture_exceptions(fatal=True)
    def delete_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        """Delete a checkpoint referenced by a tinker path."""

        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._delete_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        """Async version of delete_checkpoint_from_tinker_path."""

        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        await self._delete_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        )

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()

    @sync_only
    @capture_exceptions(fatal=True)
    def get_checkpoint_archive_url_from_tinker_path(
        self, tinker_path: str
    ) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        """Get signed URL to download checkpoint archive.

        Args:
            tinker_path: The tinker path to the checkpoint

        Returns:
            A Future containing the CheckpointArchiveUrlResponse with signed URL and expiration
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._get_checkpoint_archive_url_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def get_checkpoint_archive_url_from_tinker_path_async(
        self, tinker_path: str
    ) -> types.CheckpointArchiveUrlResponse:
        """Async version of get_checkpoint_archive_url_from_tinker_path.

        Args:
            tinker_path: The tinker path to the checkpoint

        Returns:
            CheckpointArchiveUrlResponse with signed URL and expiration
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return await self._get_checkpoint_archive_url_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        )

    def _publish_checkpoint_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> AwaitableConcurrentFuture[None]:
        """Internal method to submit publish checkpoint request."""

        async def _publish_checkpoint_async() -> None:
            async def _send_request() -> None:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.post(
                        f"/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}/publish",
                        cast_to=NoneType,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_publish_checkpoint_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def publish_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        """Publish a checkpoint referenced by a tinker path to make it publicly accessible.

        Only the exact owner of the training run can publish checkpoints.
        Published checkpoints can be unpublished using the unpublish_checkpoint_from_tinker_path method.

        Args:
            tinker_path: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

        Returns:
            A Future that completes when the checkpoint is published

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 409 if checkpoint is already public
            HTTPException: 500 if there's an error publishing the checkpoint

        Example:
            >>> future = rest_client.publish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
            >>> future.result()  # Wait for completion
            >>> print("Checkpoint published successfully")
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._publish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def publish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        """Async version of publish_checkpoint_from_tinker_path.

        Only the exact owner of the training run can publish checkpoints.
        Published checkpoints can be unpublished using the unpublish_checkpoint_from_tinker_path_async method.

        Args:
            tinker_path: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 409 if checkpoint is already public
            HTTPException: 500 if there's an error publishing the checkpoint

        Example:
            >>> await rest_client.publish_checkpoint_from_tinker_path_async("tinker://run-id/weights/0001")
            >>> print("Checkpoint published successfully")
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        await self._publish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        )

    def _unpublish_checkpoint_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> AwaitableConcurrentFuture[None]:
        """Internal method to submit unpublish checkpoint request."""

        async def _unpublish_checkpoint_async() -> None:
            async def _send_request() -> None:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.delete(
                        f"/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}/publish",
                        cast_to=NoneType,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_unpublish_checkpoint_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def unpublish_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        """Unpublish a checkpoint referenced by a tinker path to make it private again.

        Only the exact owner of the training run can unpublish checkpoints.
        This reverses the effect of publishing a checkpoint.

        Args:
            tinker_path: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

        Returns:
            A Future that completes when the checkpoint is unpublished

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 409 if checkpoint is already private
            HTTPException: 500 if there's an error unpublishing the checkpoint

        Example:
            >>> future = rest_client.unpublish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
            >>> future.result()  # Wait for completion
            >>> print("Checkpoint unpublished successfully")
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._unpublish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def unpublish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        """Async version of unpublish_checkpoint_from_tinker_path.

        Only the exact owner of the training run can unpublish checkpoints.
        This reverses the effect of publishing a checkpoint.

        Args:
            tinker_path: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 409 if checkpoint is already private
            HTTPException: 500 if there's an error unpublishing the checkpoint

        Example:
            >>> await rest_client.unpublish_checkpoint_from_tinker_path_async("tinker://run-id/weights/0001")
            >>> print("Checkpoint unpublished successfully")
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        await self._unpublish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        )

    def _list_user_checkpoints_submit(
        self, limit: int = 100, offset: int = 0
    ) -> AwaitableConcurrentFuture[types.CheckpointsListResponse]:
        """Internal method to submit list user checkpoints request."""

        async def _list_user_checkpoints_async() -> types.CheckpointsListResponse:
            async def _send_request() -> types.CheckpointsListResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    params: dict[str, object] = {"limit": limit, "offset": offset}

                    return await client.get(
                        "/api/v1/checkpoints",
                        options={"params": params},
                        cast_to=types.CheckpointsListResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_list_user_checkpoints_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def list_user_checkpoints(
        self, limit: int = 100, offset: int = 0
    ) -> ConcurrentFuture[types.CheckpointsListResponse]:
        """List all checkpoints for the current user across all their training runs.

        This method retrieves checkpoints from all training runs owned by the authenticated user,
        sorted by time (newest first). It supports pagination for efficiently handling large
        numbers of checkpoints.

        Args:
            limit: Maximum number of checkpoints to return (default 100)
            offset: Offset for pagination (default 0)

        Returns:
            A Future containing the CheckpointsListResponse with checkpoints and cursor info

        Example:
            >>> future = rest_client.list_user_checkpoints(limit=50)
            >>> response = future.result()
            >>> print(f"Found {len(response.checkpoints)} checkpoints")
            >>> print(f"Total: {response.cursor.total_count if response.cursor else 'Unknown'}")
            >>> for checkpoint in response.checkpoints:
            ...     print(f"  {checkpoint.training_run_id}/{checkpoint.checkpoint_id}")
            >>> # Get next page if there are more checkpoints
            >>> if response.cursor and response.cursor.offset + response.cursor.limit < response.cursor.total_count:
            ...     next_page = rest_client.list_user_checkpoints(limit=50, offset=50)
        """
        return self._list_user_checkpoints_submit(limit, offset).future()

    @capture_exceptions(fatal=True)
    async def list_user_checkpoints_async(
        self, limit: int = 100, offset: int = 0
    ) -> types.CheckpointsListResponse:
        """Async version of list_user_checkpoints.

        This method retrieves checkpoints from all training runs owned by the authenticated user,
        sorted by time (newest first). It supports pagination for efficiently handling large
        numbers of checkpoints.

        Args:
            limit: Maximum number of checkpoints to return (default 100)
            offset: Offset for pagination (default 0)

        Returns:
            CheckpointsListResponse with checkpoints and cursor info

        Example:
            >>> response = await rest_client.list_user_checkpoints_async(limit=50)
            >>> print(f"Found {len(response.checkpoints)} checkpoints")
            >>> print(f"Total: {response.cursor.total_count if response.cursor else 'Unknown'}")
            >>> for checkpoint in response.checkpoints:
            ...     print(f"  {checkpoint.training_run_id}/{checkpoint.checkpoint_id}")
            >>> # Get next page if there are more checkpoints
            >>> if response.cursor and response.cursor.offset + response.cursor.limit < response.cursor.total_count:
            ...     next_page = await rest_client.list_user_checkpoints_async(limit=50, offset=50)
        """
        return await self._list_user_checkpoints_submit(limit, offset)
