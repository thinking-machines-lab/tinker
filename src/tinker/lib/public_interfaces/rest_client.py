"""RestClient for Tinker API REST operations."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future as ConcurrentFuture
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from tinker import types
from tinker._models import BaseModel
from tinker._types import NoneType
from tinker.lib._jwt_auth import jwt_claims
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..sync_only import sync_only

if TYPE_CHECKING:
    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)

# How often to poll the server while a session trace export is being built.
_TRACE_EXPORT_POLL_INTERVAL_SECONDS = 5.0


class _SessionTraceExportPollResponse(BaseModel):
    """Wire format of the trace_export poll endpoint; not part of the public API."""

    status: Literal["pending", "ready", "failed"]
    url: str | None = None
    error: str | None = None


def _whoami_response_from_jwt(jwt: str) -> types.WhoamiResponse:
    """Build a WhoamiResponse from the claims of an auth-exchange JWT.

    Non-user-backed principals get an empty email claim; report None.
    """
    claims = jwt_claims(jwt)
    return types.WhoamiResponse(
        user_urn=claims["sub"],
        email=claims.get("email") or None,
    )


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
    - set_checkpoint_ttl_from_tinker_path() - set or remove TTL on a checkpoint
    - assign_session_project() - move a session into a project
    - whoami() - get the calling principal's user URN and email
    - export_session_trace() - export a session's timeline as a Perfetto trace and get a signed download URL

    Args:
    - `holder`: Internal client managing HTTP connections and async operations

    Example:
    ```python
    rest_client = service_client.create_rest_client()
    training_run = rest_client.get_training_run("run-id").result()
    print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
    checkpoints = rest_client.list_checkpoints("run-id").result()
    print(f"Found {len(checkpoints.checkpoints)} checkpoints")
    for checkpoint in checkpoints.checkpoints:
        print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
    ```
    """

    def __init__(self, holder: InternalClientHolder):
        self.holder = holder

    def _get_training_run_submit(
        self,
        training_run_id: types.ModelID,
        access_scope: Literal["owned", "accessible"] = "owned",
    ) -> AwaitableConcurrentFuture[types.TrainingRun]:
        """Internal method to submit get model request."""

        async def _get_training_run_async() -> types.TrainingRun:
            async def _send_request() -> types.TrainingRun:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/training_runs/{training_run_id}",
                        options={"params": {"access_scope": access_scope}},
                        cast_to=types.TrainingRun,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_training_run_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_training_run(
        self,
        training_run_id: types.ModelID,
        access_scope: Literal["owned", "accessible"] = "owned",
    ) -> ConcurrentFuture[types.TrainingRun]:
        """Get training run info.

        Args:
        - `training_run_id`: The training run ID to get information for

        Returns:
        - A `Future` containing the training run information

        Example:
        ```python
        future = rest_client.get_training_run("run-id")
        response = future.result()
        print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
        ```
        """
        return self._get_training_run_submit(training_run_id, access_scope=access_scope).future()

    @capture_exceptions(fatal=True)
    async def get_training_run_async(
        self, training_run_id: types.ModelID, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.TrainingRun:
        """Async version of get_training_run."""
        return await self._get_training_run_submit(training_run_id, access_scope=access_scope)

    @sync_only
    @capture_exceptions(fatal=True)
    def get_training_run_by_tinker_path(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> ConcurrentFuture[types.TrainingRun]:
        """Get training run info.

        Args:
        - `tinker_path`: The tinker path to the checkpoint

        Returns:
        - A `Future` containing the training run information

        Example:
        ```python
        future = rest_client.get_training_run_by_tinker_path("tinker://run-id/weights/checkpoint-001")
        response = future.result()
        print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
        ```
        """
        parsed_checkpoint_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(
            tinker_path
        )
        return self.get_training_run(
            parsed_checkpoint_tinker_path.training_run_id,
            access_scope=access_scope,
        )

    @capture_exceptions(fatal=True)
    async def get_training_run_by_tinker_path_async(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.TrainingRun:
        """Async version of get_training_run_by_tinker_path."""
        parsed_checkpoint_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(
            tinker_path
        )
        return await self.get_training_run_async(
            parsed_checkpoint_tinker_path.training_run_id,
            access_scope=access_scope,
        )

    @capture_exceptions(fatal=True)
    def get_weights_info_by_tinker_path(
        self, tinker_path: str
    ) -> APIFuture[types.WeightsInfoResponse]:
        """Get checkpoint information from a tinker path.

        Args:
        - `tinker_path`: The tinker path to the checkpoint

        Returns:
        - An `APIFuture` containing the checkpoint information. The future is awaitable.

        Example:
        ```python
        future = rest_client.get_weights_info_by_tinker_path("tinker://run-id/weights/checkpoint-001")
        response = future.result()  # or await future
        print(f"Base Model: {response.base_model}, LoRA Rank: {response.lora_rank}")
        ```
        """

        async def _get_weights_info_async() -> types.WeightsInfoResponse:
            async def _send_request() -> types.WeightsInfoResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.post(
                        "/api/v1/weights_info",
                        body={"tinker_path": tinker_path},
                        cast_to=types.WeightsInfoResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_weights_info_async())

    def _list_training_runs_submit(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
        project_id: str | None = None,
    ) -> AwaitableConcurrentFuture[types.TrainingRunsResponse]:
        """Internal method to submit list training runs request."""

        async def _list_training_runs_async() -> types.TrainingRunsResponse:
            async def _send_request() -> types.TrainingRunsResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    params: dict[str, object] = {
                        "limit": limit,
                        "offset": offset,
                        "access_scope": access_scope,
                    }
                    if project_id is not None:
                        params["project_id"] = project_id

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
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
        project_id: str | None = None,
    ) -> ConcurrentFuture[types.TrainingRunsResponse]:
        """List training runs with pagination support.

        Args:
        - `limit`: Maximum number of training runs to return (default 20)
        - `offset`: Offset for pagination (default 0)
        - `project_id`: If provided, only return training runs in this project

        Returns:
        - A `Future` containing the `TrainingRunsResponse` with training runs and cursor info

        Example:
        ```python
        future = rest_client.list_training_runs(limit=50)
        response = future.result()
        print(f"Found {len(response.training_runs)} training runs")
        print(f"Total: {response.cursor.total_count}")
        # Get next page
        next_page = rest_client.list_training_runs(limit=50, offset=50)
        # Only runs in a given project
        project_runs = rest_client.list_training_runs(project_id="my-project-id").result()
        ```
        """
        return self._list_training_runs_submit(
            limit,
            offset,
            access_scope=access_scope,
            project_id=project_id,
        ).future()

    @capture_exceptions(fatal=True)
    async def list_training_runs_async(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
        project_id: str | None = None,
    ) -> types.TrainingRunsResponse:
        """Async version of list_training_runs."""
        return await self._list_training_runs_submit(
            limit,
            offset,
            access_scope=access_scope,
            project_id=project_id,
        )

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
        - `training_run_id`: The training run ID to list checkpoints for

        Returns:
        - A `Future` containing the `CheckpointsListResponse` with available checkpoints

        Example:
        ```python
        future = rest_client.list_checkpoints("run-id")
        response = future.result()
        for checkpoint in response.checkpoints:
            if checkpoint.checkpoint_type == "training":
                print(f"Training checkpoint: {checkpoint.checkpoint_id}")
            elif checkpoint.checkpoint_type == "sampler":
                print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
        ```
        """
        return self._list_checkpoints_submit(training_run_id).future()

    @capture_exceptions(fatal=True)
    async def list_checkpoints_async(
        self, training_run_id: types.ModelID
    ) -> types.CheckpointsListResponse:
        """Async version of list_checkpoints."""
        return await self._list_checkpoints_submit(training_run_id)

    def _get_checkpoint_archive_url_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> AwaitableConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        """Internal method to submit get checkpoint archive URL request."""

        async def _status_task():
            logger.warning(
                f"Creating checkpoint archive for run id: {training_run_id} checkpoint id: {checkpoint_id} (this may take a while)"
            )
            i = 0
            while True:
                await asyncio.sleep(30)
                i += 1
                if i > 5:
                    logger.warning("... still running")

        async def _get_checkpoint_archive_url_async() -> types.CheckpointArchiveUrlResponse:
            async def _send_request() -> types.CheckpointArchiveUrlResponse:
                with self.holder.aclient(ClientConnectionPoolType.REST_SUPPORT_REDIRECT) as client:
                    return await client.weights.get_checkpoint_archive_url(
                        model_id=training_run_id,
                        checkpoint_id=checkpoint_id,
                    )

            status_task = asyncio.create_task(_status_task())
            try:
                result = await self.holder.execute_with_retries(_send_request)
            finally:
                status_task.cancel()

            return result

        return self.holder.run_coroutine_threadsafe(_get_checkpoint_archive_url_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_checkpoint_archive_url(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        """Get signed URL to download checkpoint archive.

        Args:
        - `training_run_id`: The training run ID to download weights for
        - `checkpoint_id`: The checkpoint ID to download

        Returns:
        - A `Future` containing the `CheckpointArchiveUrlResponse` with signed URL and expiration

        Example:
        ```python
        future = rest_client.get_checkpoint_archive_url("run-id", "checkpoint-123")
        response = future.result()
        print(f"Download URL: {response.url}")
        print(f"Expires at: {response.expires_at}")
        # Use the URL to download the archive with your preferred HTTP client
        ```
        """
        return self._get_checkpoint_archive_url_submit(training_run_id, checkpoint_id).future()

    @capture_exceptions(fatal=True)
    async def get_checkpoint_archive_url_async(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> types.CheckpointArchiveUrlResponse:
        """Async version of get_checkpoint_archive_url."""
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

    @sync_only
    @capture_exceptions(fatal=True)
    def get_audit_log(
        self,
        event_type: Literal["all", "checkpoints"] = "all",
        day: date | None = None,
    ) -> ConcurrentFuture[types.AuditLogResponse]:
        """Get an audit log of events for the caller's organization.

        Requires the tinker-admin RBAC role (VIEW_AUDIT_LOG capability).

        Args:
        - `event_type`: Type of events to include. "all" and "checkpoints"
            are currently equivalent. Defaults to "all".
        - `day`: The date to query (default: today). The window covers
            midnight to midnight UTC.

        Returns:
        - A `Future` containing the `AuditLogResponse` with audit log entries

        Example:
        ```python
        from datetime import date

        future = rest_client.get_audit_log()
        response = future.result()
        print(f"Found {len(response.entries)} audit entries")
        for entry in response.entries:
            print(f"  {entry.timestamp}: {entry.event} ({entry.tinker_path})")

        # Query a specific day
        future = rest_client.get_audit_log(day=date(2025, 1, 15))
        ```
        """
        return self.holder.run_coroutine_threadsafe(
            self.get_audit_log_async(event_type, day)
        ).future()

    @capture_exceptions(fatal=True)
    async def get_audit_log_async(
        self,
        event_type: Literal["all", "checkpoints"] = "all",
        day: date | None = None,
    ) -> types.AuditLogResponse:
        """Async version of get_audit_log."""

        async def _send_request() -> types.AuditLogResponse:
            params: dict[str, object] = {"event_type": event_type}
            if day is not None:
                params["day"] = day.isoformat()
            with self.holder.aclient(ClientConnectionPoolType.REST_SUPPORT_REDIRECT) as client:
                return await client.get(
                    "/api/v1/audit",
                    options={"params": params},
                    cast_to=types.AuditLogResponse,
                )

        return await self.holder.execute_with_retries(_send_request)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()

    @sync_only
    @capture_exceptions(fatal=True)
    def get_checkpoint_archive_url_from_tinker_path(
        self, tinker_path: str
    ) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]:
        """Get signed URL to download checkpoint archive.

        Args:
        - `tinker_path`: The tinker path to the checkpoint

        Returns:
        - A `Future` containing the `CheckpointArchiveUrlResponse` with signed URL and expiration
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._get_checkpoint_archive_url_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def get_checkpoint_archive_url_from_tinker_path_async(
        self, tinker_path: str
    ) -> types.CheckpointArchiveUrlResponse:
        """Async version of get_checkpoint_archive_url_from_tinker_path."""
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
        - `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

        Returns:
        - A `Future` that completes when the checkpoint is published

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 409 if checkpoint is already public
            HTTPException: 500 if there's an error publishing the checkpoint

        Example:
        ```python
        future = rest_client.publish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
        future.result()  # Wait for completion
        print("Checkpoint published successfully")
        ```
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._publish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def publish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        """Async version of publish_checkpoint_from_tinker_path."""
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
        - `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

        Returns:
        - A `Future` that completes when the checkpoint is unpublished

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 409 if checkpoint is already private
            HTTPException: 500 if there's an error unpublishing the checkpoint

        Example:
        ```python
        future = rest_client.unpublish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
        future.result()  # Wait for completion
        print("Checkpoint unpublished successfully")
        ```
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._unpublish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        ).future()

    @capture_exceptions(fatal=True)
    async def unpublish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        """Async version of unpublish_checkpoint_from_tinker_path."""
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        await self._unpublish_checkpoint_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id
        )

    def _set_checkpoint_ttl_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str, ttl_seconds: int | None
    ) -> AwaitableConcurrentFuture[None]:
        """Internal method to submit set checkpoint TTL request."""

        async def _set_checkpoint_ttl_async() -> None:
            async def _send_request() -> None:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.put(
                        f"/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}/ttl",
                        body={"ttl_seconds": ttl_seconds},
                        cast_to=NoneType,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_set_checkpoint_ttl_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def set_checkpoint_ttl_from_tinker_path(
        self, tinker_path: str, ttl_seconds: int | None
    ) -> ConcurrentFuture[None]:
        """Set or remove the TTL on a checkpoint referenced by a tinker path.

        If ttl_seconds is provided, the checkpoint will expire after that many seconds from now.
        If ttl_seconds is None, any existing expiration will be removed.

        Args:
        - `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")
        - `ttl_seconds`: Number of seconds until expiration, or None to remove TTL

        Returns:
        - A `Future` that completes when the TTL is set

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid or ttl_seconds <= 0
            HTTPException: 404 if checkpoint not found or user doesn't own the training run
            HTTPException: 500 if there's an error setting the TTL

        Example:
        ```python
        future = rest_client.set_checkpoint_ttl_from_tinker_path("tinker://run-id/weights/0001", 86400)
        future.result()  # Wait for completion
        print("Checkpoint TTL set successfully")
        ```
        """
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._set_checkpoint_ttl_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id, ttl_seconds
        ).future()

    @capture_exceptions(fatal=True)
    async def set_checkpoint_ttl_from_tinker_path_async(
        self, tinker_path: str, ttl_seconds: int | None
    ) -> None:
        """Async version of set_checkpoint_ttl_from_tinker_path."""
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        await self._set_checkpoint_ttl_submit(
            parsed_tinker_path.training_run_id, parsed_tinker_path.checkpoint_id, ttl_seconds
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
        - `limit`: Maximum number of checkpoints to return (default 100)
        - `offset`: Offset for pagination (default 0)

        Returns:
        - A `Future` containing the `CheckpointsListResponse` with checkpoints and cursor info

        Example:
        ```python
        future = rest_client.list_user_checkpoints(limit=50)
        response = future.result()
        print(f"Found {len(response.checkpoints)} checkpoints")
        print(f"Total: {response.cursor.total_count if response.cursor else 'Unknown'}")
        for checkpoint in response.checkpoints:
            print(f"  {checkpoint.training_run_id}/{checkpoint.checkpoint_id}")
        # Get next page if there are more checkpoints
        if response.cursor and response.cursor.offset + response.cursor.limit < response.cursor.total_count:
            next_page = rest_client.list_user_checkpoints(limit=50, offset=50)
        ```
        """
        return self._list_user_checkpoints_submit(limit, offset).future()

    @capture_exceptions(fatal=True)
    async def list_user_checkpoints_async(
        self, limit: int = 100, offset: int = 0
    ) -> types.CheckpointsListResponse:
        """Async version of list_user_checkpoints."""
        return await self._list_user_checkpoints_submit(limit, offset)

    def _get_session_submit(
        self,
        session_id: str,
        access_scope: Literal["owned", "accessible"] = "owned",
    ) -> AwaitableConcurrentFuture[types.GetSessionResponse]:
        """Internal method to submit get session request."""

        async def _get_session_async() -> types.GetSessionResponse:
            async def _send_request() -> types.GetSessionResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/sessions/{session_id}",
                        options={"params": {"access_scope": access_scope}},
                        cast_to=types.GetSessionResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_session_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_session(
        self, session_id: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> ConcurrentFuture[types.GetSessionResponse]:
        """Get session information including all training runs and samplers.

        Args:
        - `session_id`: The session ID to get information for

        Returns:
        - A `Future` containing the `GetSessionResponse` with training_run_ids and sampler_ids

        Example:
        ```python
        future = rest_client.get_session("session-id")
        response = future.result()
        print(f"Training runs: {len(response.training_run_ids)}")
        print(f"Samplers: {len(response.sampler_ids)}")
        ```
        """
        return self._get_session_submit(session_id, access_scope=access_scope).future()

    @capture_exceptions(fatal=True)
    async def get_session_async(
        self, session_id: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.GetSessionResponse:
        """Async version of get_session."""
        return await self._get_session_submit(session_id, access_scope=access_scope)

    def _list_sessions_submit(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
    ) -> AwaitableConcurrentFuture[types.ListSessionsResponse]:
        """Internal method to submit list sessions request."""

        async def _list_sessions_async() -> types.ListSessionsResponse:
            async def _send_request() -> types.ListSessionsResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    params: dict[str, object] = {
                        "limit": limit,
                        "offset": offset,
                        "access_scope": access_scope,
                    }

                    return await client.get(
                        "/api/v1/sessions",
                        options={"params": params},
                        cast_to=types.ListSessionsResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_list_sessions_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
    ) -> ConcurrentFuture[types.ListSessionsResponse]:
        """List sessions with pagination support.

        Args:
        - `limit`: Maximum number of sessions to return (default 20)
        - `offset`: Offset for pagination (default 0)

        Returns:
        - A `Future` containing the `ListSessionsResponse` with list of session IDs

        Example:
        ```python
        future = rest_client.list_sessions(limit=50)
        response = future.result()
        print(f"Found {len(response.sessions)} sessions")
        # Get next page
        next_page = rest_client.list_sessions(limit=50, offset=50)
        ```
        """
        return self._list_sessions_submit(
            limit,
            offset,
            access_scope=access_scope,
        ).future()

    @capture_exceptions(fatal=True)
    async def list_sessions_async(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
    ) -> types.ListSessionsResponse:
        """Async version of list_sessions."""
        return await self._list_sessions_submit(
            limit,
            offset,
            access_scope=access_scope,
        )

    def _assign_session_project_submit(
        self, session_id: str, project_id: str
    ) -> AwaitableConcurrentFuture[None]:
        """Internal method to submit assign session project request."""

        async def _assign_session_project_async() -> None:
            async def _send_request() -> None:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.put(
                        f"/api/v1/sessions/{session_id}/project",
                        body={"project_id": project_id},
                        cast_to=NoneType,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_assign_session_project_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def assign_session_project(self, session_id: str, project_id: str) -> ConcurrentFuture[None]:
        """Move a session (and all of its training runs/samplers) into a project.

        Use this to attach a previously-created session to a project, or to move
        a session between projects. Clearing the project is not supported — sessions
        cannot be moved out of a project once placed.

        Args:
        - `session_id`: The session ID to move
        - `project_id`: The destination project ID

        Returns:
        - A `Future` that completes when the session has been moved

        Raises:
            HTTPException: 400 if `project_id` is missing
            HTTPException: 403 if the caller lacks access to the destination project
            HTTPException: 404 if the session is not found or not accessible

        Example:
        ```python
        future = rest_client.assign_session_project("session-id", "project-id")
        future.result()
        ```
        """
        return self._assign_session_project_submit(session_id, project_id).future()

    @capture_exceptions(fatal=True)
    async def assign_session_project_async(self, session_id: str, project_id: str) -> None:
        """Async version of assign_session_project."""
        await self._assign_session_project_submit(session_id, project_id)

    def _export_session_trace_submit(self, session_id: str) -> AwaitableConcurrentFuture[str]:
        """Internal method to poll the trace export job until it completes."""

        async def _export_session_trace_async() -> str:
            async def _send_request() -> _SessionTraceExportPollResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/sessions/{session_id}/trace_export",
                        cast_to=_SessionTraceExportPollResponse,
                    )

            first_poll = True
            while True:
                response = await self.holder.execute_with_retries(_send_request)
                if response.status == "ready":
                    assert response.url is not None  # guaranteed by the server when ready
                    return response.url
                if response.status == "failed":
                    raise RuntimeError(
                        f"Trace export failed for session {session_id}: {response.error}"
                    )
                if first_poll:
                    logger.warning(
                        f"Building trace export for session: {session_id} (this may take a while)"
                    )
                    first_poll = False
                await asyncio.sleep(_TRACE_EXPORT_POLL_INTERVAL_SECONDS)

        return self.holder.run_coroutine_threadsafe(_export_session_trace_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def export_session_trace(self, session_id: str) -> ConcurrentFuture[str]:
        """Export a session's timeline as a Perfetto trace and get a signed download URL.

        Kicks off (or reuses) an async export job on the server that builds a
        Perfetto trace (`.pftrace`) of the session's training and sampling
        requests, polls until the file is uploaded, and returns a signed
        download URL. The URL expires after about an hour; call this method
        again to get a fresh one (the trace is not rebuilt if it is already up
        to date).

        To download the trace, issue a plain HTTPS GET to the URL (no auth
        headers needed):
        ```python
        import urllib.request

        url = rest_client.export_session_trace("session-id").result()
        urllib.request.urlretrieve(url, "session-id.pftrace")
        ```
        or from a shell: `curl -o session.pftrace '<url>'` (quote the URL, it
        contains query parameters). Open the file in https://ui.perfetto.dev to
        view the timeline, or use `tinker session export-trace <session-id>`
        to do all of this from the CLI.

        Args:
        - `session_id`: The session ID to export a trace for

        Returns:
        - A `Future` containing the signed download URL for the `.pftrace` file

        Raises:
            RuntimeError: if the export job fails

        Example:
        ```python
        url = rest_client.export_session_trace("session-id").result()
        print(f"Download URL: {url}")
        ```
        """
        return self._export_session_trace_submit(session_id).future()

    @capture_exceptions(fatal=True)
    async def export_session_trace_async(self, session_id: str) -> str:
        """Async version of export_session_trace."""
        return await self._export_session_trace_submit(session_id)

    @capture_exceptions(fatal=True)
    def whoami(self) -> APIFuture[types.WhoamiResponse]:
        """Get the calling principal's identity.

        Returns the user URN associated with the credential in use,
        and the user's email when the principal is user-backed.

        The identity is read from the claims of the JWT the SDK already
        holds from auth (cached and refreshed in the background), so this
        makes no server requests.

        Returns:
        - An `APIFuture` containing the `WhoamiResponse`. The future is awaitable.

        Example:
        ```python
        # Sync usage
        response = rest_client.whoami().result()
        print(f"URN: {response.user_urn}, email: {response.email}")

        # Async usage
        response = await rest_client.whoami()
        ```
        """

        async def _whoami_async() -> types.WhoamiResponse:
            jwt = await self.holder.get_identity_jwt()
            if jwt is None:
                raise RuntimeError(
                    "whoami requires JWT auth, but the server has it disabled "
                    "and the credential is a plain API key."
                )
            return _whoami_response_from_jwt(jwt)

        return self.holder.run_coroutine_threadsafe(_whoami_async())

    @capture_exceptions(fatal=True)
    def get_sampler(self, sampler_id: str) -> APIFuture[types.GetSamplerResponse]:
        """Get sampler information.

        Args:
        - `sampler_id`: The sampler ID (sampling_session_id) to get information for

        Returns:
        - An `APIFuture` containing the `GetSamplerResponse` with sampler details

        Example:
        ```python
        # Sync usage
        future = rest_client.get_sampler("session-id:sample:0")
        response = future.result()
        print(f"Base model: {response.base_model}")
        print(f"Model path: {response.model_path}")

        # Async usage
        response = await rest_client.get_sampler("session-id:sample:0")
        print(f"Base model: {response.base_model}")
        ```
        """

        async def _get_sampler_async() -> types.GetSamplerResponse:
            async def _send_request() -> types.GetSamplerResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/samplers/{sampler_id}",
                        cast_to=types.GetSamplerResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_sampler_async())

    @capture_exceptions(fatal=True)
    async def get_sampler_async(self, sampler_id: str) -> types.GetSamplerResponse:
        """Async version of get_sampler."""
        return await self.get_sampler(sampler_id)

    def _get_billing_usage_submit(
        self,
        starting_on: datetime | str,
        ending_before: datetime | str,
    ) -> AwaitableConcurrentFuture[types.BillingUsageResponse]:
        """Internal method to submit the billing usage request."""
        # Typed request model: bad arguments fail here with a clear
        # validation error, and datetimes serialize to RFC 3339 via pydantic
        # (RFC 3339 strings are accepted and parsed too).
        request = types.GetBillingUsageRequest.model_validate(
            {
                "starting_on": starting_on,
                "ending_before": ending_before,
            }
        )

        async def _get_billing_usage_async() -> types.BillingUsageResponse:
            async def _send_request() -> types.BillingUsageResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        "/api/v1/billing/usage/events",
                        options={"params": request.model_dump(mode="json", exclude_none=True)},
                        cast_to=types.BillingUsageResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_billing_usage_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_billing_usage(
        self,
        starting_on: datetime | str,
        ending_before: datetime | str,
    ) -> ConcurrentFuture[types.BillingUsageResponse]:
        """Get detailed billing usage for your organization.

        Returns hourly-bucketed usage as a list of `BillingUsageEvent`
        envelopes: the shared attribution (bucket, base model, user, session,
        project) lives on the envelope, and the usage-kind-specific payload
        is `event_info` — a union discriminated on `.type` (training /
        sampling_prefill / sampling_sample / checkpoint / storage), each
        carrying exactly the fields that apply (token_count, gigabyte_hours,
        count, the prefill cached flag). Session user_metadata comes once
        per session in `response.sessions`, keyed by session_id. No dollar
        amounts. Data lags real time by up to a few hours. Requires billing
        view access in your organization.

        Args:
        - `starting_on`: Inclusive window start (RFC 3339 string or datetime),
          aligned to a UTC hour boundary
        - `ending_before`: Exclusive window end, aligned to a UTC hour
          boundary; at most 14 days after `starting_on`; must not start in the future

        Returns:
        - A `Future` containing the `BillingUsageResponse`

        Example:
        ```python
        future = rest_client.get_billing_usage(
            "2026-07-13T00:00:00Z", "2026-07-14T00:00:00Z"
        )
        for event in future.result().data:
            match event.event_info:
                case types.StorageBillingEvent() as info:
                    print(event.bucket_start, "storage", info.gigabyte_hours, "GB-h")
                case types.CheckpointBillingEvent() as info:
                    print(event.bucket_start, "checkpoints", info.count)
                case info:
                    print(event.bucket_start, info.type, event.base_model, info.token_count)
        ```
        """
        return self._get_billing_usage_submit(starting_on, ending_before).future()

    @capture_exceptions(fatal=True)
    async def get_billing_usage_async(
        self,
        starting_on: datetime | str,
        ending_before: datetime | str,
    ) -> types.BillingUsageResponse:
        """Async version of get_billing_usage."""
        return await self._get_billing_usage_submit(starting_on, ending_before)
