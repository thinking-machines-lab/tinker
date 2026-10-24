"""RestClient for Tinker API REST operations."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import Future as ConcurrentFuture
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from typing_extensions import deprecated

from tinker import types
from tinker._compat import model_dump
from tinker._models import BaseModel
from tinker._types import NoneType
from tinker.lib._jwt_auth import jwt_claims
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
from tinker.lib.experimental import ExperimentalWarning, experimental
from tinker.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from tinker.lib.public_interfaces.rest_client_common import rest_client_holder, sdk_headers
from tinker.lib.telemetry import Telemetry, capture_exceptions
from tinker.lib.telemetry_provider import TelemetryProvider

from ..internal_client_holder import InternalClientHolder
from ..sync_only import sync_only

if TYPE_CHECKING:
    from .admin_client import AdminClient

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
        org_urn=f"tml:organization:{claims['org_id']}",
        email=claims.get("email") or None,
    )


class RestClient(TelemetryProvider):
    """Client for REST API operations like listing checkpoints and metadata.

    The RestClient provides access to various REST endpoints for querying
    model information, checkpoints, and other resources. Construct one directly
    with `RestClient()`; no ServiceClient or training session is needed.

    Key methods:
    - list_checkpoints() - list available model checkpoints (both training and sampler)
    - list_user_checkpoints() - list all checkpoints across all user's training runs
    - get_training_run() - get model information and metadata as ModelEntry
    - delete_checkpoint() - delete an existing checkpoint for a training run
    - get_billing_usage(), get_current_checkpoint_storage_usage(), get_audit_log() -
      (deprecated) admin-only operations, now on `AdminClient`
    - get_external_weights_urls() - get per-file signed URLs for an external weights checkpoint
    - copy_weights() - copy a checkpoint into a selected project
    - publish_checkpoint_from_tinker_path() - publish a checkpoint to make it public
    - unpublish_checkpoint_from_tinker_path() - unpublish a checkpoint to make it private
    - set_checkpoint_ttl_from_tinker_path() - set or remove TTL on a checkpoint
    - assign_session_project() - move a session into a project
    - whoami() - get the calling principal's user URN and email
    - export_session_trace() - export a session's timeline as a Perfetto trace and get a signed download URL
    - search_sessions() - (experimental) search your organization's sessions, newest first

    A RestClient constructed directly talks to the REST API only: no session is
    created on the server, so no project is needed and nothing shows up in the
    sessions list. Its connections are shared by every client constructed this
    way with the same base URL and API key, stay open for the life of the
    process, and need no closing.

    Operations that need an organization admin or billing-admin role live on
    `AdminClient`, which is constructed the same way.

    Args:
    - `base_url`: The Tinker API URL. Defaults to the `TINKER_BASE_URL`
      environment variable, then to the production API.
    - `api_key`: The Tinker API key. Defaults to the `TINKER_API_KEY`
      environment variable, then to the stored credentials.
    - `_holder`: Internal use only. The client managing HTTP connections and
      async operations that a `ServiceClient` hands over; it cannot be combined
      with `base_url` or `api_key`.

    Example:
    ```python
    rest_client = tinker.RestClient()
    checkpoints = rest_client.list_user_checkpoints().result()
    print(f"Found {len(checkpoints.checkpoints)} checkpoints")

    training_run = rest_client.get_training_run("run-id").result()
    print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
    checkpoints = rest_client.list_checkpoints("run-id").result()
    print(f"Found {len(checkpoints.checkpoints)} checkpoints")
    for checkpoint in checkpoints.checkpoints:
        print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
    ```
    """

    def __init__(
        self,
        _holder: InternalClientHolder | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.holder = rest_client_holder("RestClient", _holder, base_url=base_url, api_key=api_key)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()

    def _admin_client(self) -> AdminClient:
        """An AdminClient over this client's holder, backing the deprecated admin methods."""
        from .admin_client import AdminClient

        return AdminClient(self.holder)

    def _get_training_run_submit(
        self,
        training_run_id: types.ModelID,
        access_scope: Literal["owned", "accessible"] = "owned",
        *,
        sdk_method: str,
    ) -> AwaitableConcurrentFuture[types.TrainingRun]:
        """Internal method to submit get model request."""

        async def _get_training_run_async() -> types.TrainingRun:
            async def _send_request() -> types.TrainingRun:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.get(
                        f"/api/v1/training_runs/{training_run_id}",
                        options={
                            "params": {"access_scope": access_scope},
                            "headers": sdk_headers(sdk_method),
                        },
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
        return self._get_training_run_submit(
            training_run_id, access_scope=access_scope, sdk_method="get_training_run"
        ).future()

    @capture_exceptions(fatal=True)
    async def get_training_run_async(
        self, training_run_id: types.ModelID, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.TrainingRun:
        """Async version of get_training_run."""
        return await self._get_training_run_submit(
            training_run_id, access_scope=access_scope, sdk_method="get_training_run"
        )

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
        return self._get_training_run_submit(
            parsed_checkpoint_tinker_path.training_run_id,
            access_scope=access_scope,
            sdk_method="get_training_run_by_tinker_path",
        ).future()

    @capture_exceptions(fatal=True)
    async def get_training_run_by_tinker_path_async(
        self, tinker_path: str, access_scope: Literal["owned", "accessible"] = "owned"
    ) -> types.TrainingRun:
        """Async version of get_training_run_by_tinker_path."""
        parsed_checkpoint_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(
            tinker_path
        )
        return await self._get_training_run_submit(
            parsed_checkpoint_tinker_path.training_run_id,
            access_scope=access_scope,
            sdk_method="get_training_run_by_tinker_path",
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
                        options={"headers": sdk_headers("get_weights_info_by_tinker_path")},
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
                        options={"params": params, "headers": sdk_headers("list_training_runs")},
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
                    return await client.weights.list(
                        training_run_id, extra_headers=sdk_headers("list_checkpoints")
                    )

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

    def _get_external_weights_urls_submit(
        self, tinker_path: str
    ) -> AwaitableConcurrentFuture[types.ExternalWeightsUrlsResponse]:
        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        if not parsed_tinker_path.checkpoint_id.startswith("external_weights/"):
            raise ValueError(
                "get_external_weights_urls only accepts external weights checkpoints "
                f"(tinker://<training_run_id>/external_weights/<name>), got {tinker_path!r}"
            )

        async def _get_external_weights_urls_async() -> types.ExternalWeightsUrlsResponse:
            async def _send_request() -> types.ExternalWeightsUrlsResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.get_external_weights_urls(
                        model_id=parsed_tinker_path.training_run_id,
                        checkpoint_id=parsed_tinker_path.checkpoint_id,
                        extra_headers=sdk_headers("get_external_weights_urls"),
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_external_weights_urls_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_external_weights_urls(
        self, tinker_path: str
    ) -> ConcurrentFuture[types.ExternalWeightsUrlsResponse]:
        """Get signed download URLs, one per file, for an external weights checkpoint.

        Args:
        - `tinker_path`: The checkpoint's tinker path, as returned by `save_weights_external`
          (`tinker://<training_run_id>/external_weights/<name>`)

        Returns:
        - A `Future` containing the `ExternalWeightsUrlsResponse`: `urls` maps each file path
          (relative to the checkpoint root) to a signed URL, valid until `expires`
        """
        return self._get_external_weights_urls_submit(tinker_path).future()

    @capture_exceptions(fatal=True)
    async def get_external_weights_urls_async(
        self, tinker_path: str
    ) -> types.ExternalWeightsUrlsResponse:
        """Async version of get_external_weights_urls."""
        return await self._get_external_weights_urls_submit(tinker_path)

    def _delete_checkpoint_submit(
        self, training_run_id: types.ModelID, checkpoint_id: str, *, sdk_method: str
    ) -> AwaitableConcurrentFuture[None]:
        """Internal method to submit delete checkpoint request."""

        async def _delete_checkpoint_async() -> None:
            async def _send_request() -> None:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    await client.delete(
                        f"/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}",
                        options={"headers": sdk_headers(sdk_method)},
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

        return self._delete_checkpoint_submit(
            training_run_id, checkpoint_id, sdk_method="delete_checkpoint"
        ).future()

    @capture_exceptions(fatal=True)
    async def delete_checkpoint_async(
        self, training_run_id: types.ModelID, checkpoint_id: str
    ) -> None:
        """Async version of delete_checkpoint."""

        await self._delete_checkpoint_submit(
            training_run_id, checkpoint_id, sdk_method="delete_checkpoint"
        )

    @sync_only
    @capture_exceptions(fatal=True)
    def delete_checkpoint_from_tinker_path(self, tinker_path: str) -> ConcurrentFuture[None]:
        """Delete a checkpoint referenced by a tinker path."""

        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self._delete_checkpoint_submit(
            parsed_tinker_path.training_run_id,
            parsed_tinker_path.checkpoint_id,
            sdk_method="delete_checkpoint_from_tinker_path",
        ).future()

    @capture_exceptions(fatal=True)
    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        """Async version of delete_checkpoint_from_tinker_path."""

        parsed_tinker_path = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        await self._delete_checkpoint_submit(
            parsed_tinker_path.training_run_id,
            parsed_tinker_path.checkpoint_id,
            sdk_method="delete_checkpoint_from_tinker_path",
        )

    @deprecated(
        "DEPRECATED: RestClient.get_audit_log() is deprecated and will be removed in a future "
        "tinker SDK version. Use AdminClient.get_audit_log() instead; AdminClient is constructed "
        "the same way as RestClient.",
        category=FutureWarning,
    )
    def get_audit_log(
        self,
        event_type: Literal["all", "checkpoints", "projects", "teams", "organizations"] = "all",
        day: date | None = None,
    ) -> ConcurrentFuture[types.AuditLogResponse]:
        """Deprecated: use `AdminClient.get_audit_log()` instead. This method will be removed in a
        subsequent release."""
        return self._admin_client().get_audit_log(event_type, day)

    @deprecated(
        "DEPRECATED: RestClient.get_audit_log_async() is deprecated and will be removed in a "
        "future tinker SDK version. Use AdminClient.get_audit_log_async() instead; AdminClient is "
        "constructed the same way as RestClient.",
        category=FutureWarning,
    )
    async def get_audit_log_async(
        self,
        event_type: Literal["all", "checkpoints", "projects", "teams", "organizations"] = "all",
        day: date | None = None,
    ) -> types.AuditLogResponse:
        """Deprecated: use `AdminClient.get_audit_log_async()` instead. This method will be removed
        in a subsequent release."""
        return await self._admin_client().get_audit_log_async(event_type, day)

    def _copy_weights_submit(
        self,
        request: types.CopyWeightsRequest,
        weights_access_token: str | None,
    ) -> AwaitableConcurrentFuture[str]:
        # Reuse one key across HTTP-client and holder retries to avoid duplicate copies.
        headers = {**sdk_headers("copy_weights"), "X-Idempotency-Key": str(uuid.uuid4())}
        headers_secret = (
            {"X-Tinker-Weights-Access-Token": weights_access_token}
            if weights_access_token is not None
            else None
        )

        async def _copy_weights_async() -> str:
            async def _send_request() -> types.CopyWeightsResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.models.copy_weights(
                        request=request,
                        extra_headers=headers,
                        extra_headers_secret=headers_secret,
                    )

            response = await self.holder.execute_with_retries(_send_request)
            logger.info(f"Copied {request.source_path} to {response.tinker_path}")
            return response.tinker_path

        return self.holder.run_coroutine_threadsafe(_copy_weights_async())

    def copy_weights(
        self,
        path: str,
        destination_project_id: str | None = None,
        *,
        ttl_seconds: int | None = None,
        weights_access_token: str | None = None,
    ) -> AwaitableConcurrentFuture[str]:
        """Copy a checkpoint into a selected project.

        Storage is shared with the source, so no bytes are duplicated. The copy
        keeps the source's checkpoint kind and belongs to a new training run
        that cannot be trained on.

        Args:
        - `path`: Tinker path of the checkpoint to copy
        - `destination_project_id`: Project that will own the copy. If omitted,
          defaults to the `TINKER_PROJECT_ID` environment variable.
        - `ttl_seconds`: Seconds until the copy expires, between 1 hour (3600)
          and 10 years, or None for no expiry
        - `weights_access_token`: Optional access token for copying weights readable
          under a different account

        Returns:
        - A future for the tinker path of the copy. Await it, or call `.result()`.

        Raises:
        - `ValueError`: If neither a destination project nor `TINKER_PROJECT_ID` is set

        Example:
        ```python
        copied_path = rest_client.copy_weights(
            "tinker://run-id/weights/step-400", "proj-archive"
        ).result()
        print(copied_path)
        ```
        """
        if destination_project_id is None:
            destination_project_id = os.environ.get("TINKER_PROJECT_ID") or None
        if destination_project_id is None:
            raise ValueError("Provide destination_project_id or set TINKER_PROJECT_ID")
        request = types.CopyWeightsRequest(
            source_path=path,
            destination_project_id=destination_project_id,
            ttl_seconds=ttl_seconds,
        )
        return self._copy_weights_submit(request, weights_access_token)

    @capture_exceptions(fatal=True)
    async def copy_weights_async(
        self,
        path: str,
        destination_project_id: str | None = None,
        *,
        ttl_seconds: int | None = None,
        weights_access_token: str | None = None,
    ) -> str:
        """Async version of copy_weights."""
        return await self.copy_weights(
            path,
            destination_project_id,
            ttl_seconds=ttl_seconds,
            weights_access_token=weights_access_token,
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
                        options={"headers": sdk_headers("publish_checkpoint_from_tinker_path")},
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
                        options={"headers": sdk_headers("unpublish_checkpoint_from_tinker_path")},
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
                        options={"headers": sdk_headers("set_checkpoint_ttl_from_tinker_path")},
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
        It must be between 1 hour (3600) and 10 years.
        If ttl_seconds is None, any existing expiration will be removed.

        Args:
        - `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")
        - `ttl_seconds`: Seconds until expiration (1 hour to 10 years), or None to remove TTL

        Returns:
        - A `Future` that completes when the TTL is set

        Raises:
            HTTPException: 400 if checkpoint identifier is invalid or ttl_seconds is out of range
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
                        options={
                            "params": params,
                            "headers": sdk_headers("list_user_checkpoints"),
                        },
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
                        options={
                            "params": {"access_scope": access_scope},
                            "headers": sdk_headers("get_session"),
                        },
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
        - A `Future` containing the `GetSessionResponse` with training_run_ids, sampler_ids,
          and user_metadata

        Example:
        ```python
        future = rest_client.get_session("session-id")
        response = future.result()
        print(f"Training runs: {len(response.training_run_ids)}")
        print(f"Samplers: {len(response.sampler_ids)}")
        print(f"User metadata: {response.user_metadata}")
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
                        options={"params": params, "headers": sdk_headers("list_sessions")},
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
                        options={"headers": sdk_headers("assign_session_project")},
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
                        options={"headers": sdk_headers("export_session_trace")},
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

        Returns the user URN and organization URN associated with the
        credential in use, and the user's email when the principal is
        user-backed.

        The identity is read from the claims of the JWT the SDK already
        holds from auth (cached and refreshed in the background), so this
        makes no server requests.

        Returns:
        - An `APIFuture` containing the `WhoamiResponse`. The future is awaitable.

        Example:
        ```python
        # Sync usage
        response = rest_client.whoami().result()
        print(f"user: {response.user_urn}, org: {response.org_urn}, email: {response.email}")

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
                        options={"headers": sdk_headers("get_sampler")},
                        cast_to=types.GetSamplerResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_sampler_async())

    @capture_exceptions(fatal=True)
    async def get_sampler_async(self, sampler_id: str) -> types.GetSamplerResponse:
        """Async version of get_sampler."""
        return await self.get_sampler(sampler_id)

    @deprecated(
        "DEPRECATED: RestClient.get_billing_usage() is deprecated and will be removed in a future "
        "tinker SDK version. Use AdminClient.get_billing_usage() instead; AdminClient is "
        "constructed the same way as RestClient.",
        category=FutureWarning,
    )
    def get_billing_usage(
        self,
        starting_on: datetime | str,
        ending_before: datetime | str,
    ) -> ConcurrentFuture[types.BillingUsageResponse]:
        """Deprecated: use `AdminClient.get_billing_usage()` instead. This method will be removed in
        a subsequent release."""
        return self._admin_client().get_billing_usage(starting_on, ending_before)

    @deprecated(
        "DEPRECATED: RestClient.get_billing_usage_async() is deprecated and will be removed in a "
        "future tinker SDK version. Use AdminClient.get_billing_usage_async() instead; AdminClient "
        "is constructed the same way as RestClient.",
        category=FutureWarning,
    )
    async def get_billing_usage_async(
        self,
        starting_on: datetime | str,
        ending_before: datetime | str,
    ) -> types.BillingUsageResponse:
        """Deprecated: use `AdminClient.get_billing_usage_async()` instead. This method will be
        removed in a subsequent release."""
        return await self._admin_client().get_billing_usage_async(starting_on, ending_before)

    @deprecated(
        "DEPRECATED: RestClient.get_current_checkpoint_storage_usage() is deprecated and will be "
        "removed in a future tinker SDK version. Use "
        "AdminClient.get_current_checkpoint_storage_usage() instead; AdminClient is constructed "
        "the same way as RestClient.",
        category=FutureWarning,
    )
    def get_current_checkpoint_storage_usage(
        self,
        project_id: str | None = None,
    ) -> ConcurrentFuture[types.CurrentCheckpointStorageUsageResponse]:
        """Deprecated: use `AdminClient.get_current_checkpoint_storage_usage()` instead. This method
        will be removed in a subsequent release."""
        return self._admin_client().get_current_checkpoint_storage_usage(project_id)

    @deprecated(
        "DEPRECATED: RestClient.get_current_checkpoint_storage_usage_async() is deprecated and "
        "will be removed in a future tinker SDK version. Use "
        "AdminClient.get_current_checkpoint_storage_usage_async() instead; AdminClient is "
        "constructed the same way as RestClient.",
        category=FutureWarning,
    )
    async def get_current_checkpoint_storage_usage_async(
        self,
        project_id: str | None = None,
    ) -> types.CurrentCheckpointStorageUsageResponse:
        """Deprecated: use `AdminClient.get_current_checkpoint_storage_usage_async()` instead. This
        method will be removed in a subsequent release."""
        return await self._admin_client().get_current_checkpoint_storage_usage_async(project_id)

    def _search_sessions_submit(
        self, request: types.SessionSearchRequest
    ) -> AwaitableConcurrentFuture[types.SessionSearchResponse]:
        """Internal method to submit the session search request."""

        async def _search_sessions_async() -> types.SessionSearchResponse:
            async def _send_request() -> types.SessionSearchResponse:
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.post(
                        "/api/v1/sessions/search",
                        body=model_dump(request, mode="json", exclude_none=True),
                        options={"headers": sdk_headers("search_sessions")},
                        cast_to=types.SessionSearchResponse,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_search_sessions_async())

    @experimental(
        "RestClient.search_sessions is experimental: it may change or be removed in "
        "any release without a deprecation period.",
        category=ExperimentalWarning,
    )
    @sync_only
    @capture_exceptions(fatal=True)
    def search_sessions(
        self, request: types.SessionSearchRequest
    ) -> ConcurrentFuture[types.SessionSearchResponse]:
        """Search your organization's sessions, newest first.

        Experimental: this API may change or be removed in any release.

        Results are restricted to projects you may list. Page through them by
        passing each response's `next_cursor` as the next request's `cursor`.

        Args:
        - `request`: The search filters, page size and cursor

        Returns:
        - A `Future` containing one `SessionSearchResponse` page

        Example:
        ```python
        from tinker.types import AccessibleSearchOwner, SessionSearchRequest

        request = SessionSearchRequest(display_name="nightly", owner=AccessibleSearchOwner())
        page = rest_client.search_sessions(request).result()
        while True:
            for session in page.sessions:
                print(session.session_id, session.display_name)
            if page.next_cursor is None:
                break
            request = request.model_copy(update={"cursor": page.next_cursor})
            page = rest_client.search_sessions(request).result()
        ```
        """
        return self._search_sessions_submit(request).future()

    @experimental(
        "RestClient.search_sessions_async is experimental: it may change or be removed in "
        "any release without a deprecation period.",
        category=ExperimentalWarning,
    )
    @capture_exceptions(fatal=True)
    async def search_sessions_async(
        self, request: types.SessionSearchRequest
    ) -> types.SessionSearchResponse:
        """Async version of search_sessions."""
        return await self._search_sessions_submit(request)
