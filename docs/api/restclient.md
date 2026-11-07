RestClient for Tinker API REST operations.

## `RestClient` Objects

```python
@with_sdk_method_telemetry
class RestClient(TelemetryProvider)
```

Client for REST API operations like listing checkpoints and metadata.

The RestClient provides access to various REST endpoints for querying
model information, checkpoints, and other resources. Construct one directly
with `RestClient()`; no ServiceClient or training session is needed.

Key methods:
- get_server_capabilities() - get the models the caller can use and the server's features
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
- search_checkpoints() - (experimental) search your organization's checkpoints, newest first

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

#### `get_server_capabilities`

```python
def get_server_capabilities(
) -> APIFuture[types.GetServerCapabilitiesResponse]
```

Query the server's supported features and capabilities.

Returns:
- An `APIFuture` containing the `GetServerCapabilitiesResponse` with the models
    the caller can use, features, and limits. Await it, or call `.result()`.

Example:
```python
capabilities = rest_client.get_server_capabilities().result()
for model in capabilities.supported_models:
    print(model.model_name)
```

#### `get_training_run`

```python
def get_training_run(
    training_run_id: types.ModelID,
    access_scope: Literal["owned", "accessible"] = "owned"
) -> APIFuture[types.TrainingRun]
```

Get training run info.

Args:
- `training_run_id`: The training run ID to get information for

Returns:
- An `APIFuture` containing the training run information. Await it, or call `.result()`.

Example:
```python
future = rest_client.get_training_run("run-id")
response = future.result()  # or `await future`
print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

#### `get_training_run_async`

```python
async def get_training_run_async(
    training_run_id: types.ModelID,
    access_scope: Literal["owned",
                          "accessible"] = "owned") -> types.TrainingRun
```

Deprecated: use `await rest_client.get_training_run(...)`; `get_training_run()` returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent release.

#### `get_training_run_by_tinker_path`

```python
def get_training_run_by_tinker_path(
    tinker_path: str,
    access_scope: Literal["owned", "accessible"] = "owned"
) -> APIFuture[types.TrainingRun]
```

Get training run info.

Args:
- `tinker_path`: The tinker path to the checkpoint

Returns:
- An `APIFuture` containing the training run information. Await it, or call `.result()`.

Example:
```python
future = rest_client.get_training_run_by_tinker_path("tinker://run-id/weights/checkpoint-001")
response = future.result()  # or `await future`
print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

#### `get_training_run_by_tinker_path_async`

```python
async def get_training_run_by_tinker_path_async(
    tinker_path: str,
    access_scope: Literal["owned",
                          "accessible"] = "owned") -> types.TrainingRun
```

Deprecated: use `await rest_client.get_training_run_by_tinker_path(...)`;
`get_training_run_by_tinker_path()` returns an `APIFuture` that can be awaited. This method
will be removed in a subsequent release.

#### `get_weights_info_by_tinker_path`

```python
def get_weights_info_by_tinker_path(
        tinker_path: str) -> APIFuture[types.WeightsInfoResponse]
```

Get checkpoint information from a tinker path.

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

#### `list_training_runs`

```python
def list_training_runs(
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
        project_id: str | None = None
) -> APIFuture[types.TrainingRunsResponse]
```

List training runs with pagination support.

Args:
- `limit`: Maximum number of training runs to return (default 20)
- `offset`: Offset for pagination (default 0)
- `project_id`: If provided, only return training runs in this project

Returns:
- An `APIFuture` containing the `TrainingRunsResponse` with training runs and cursor info.
    Await it, or call `.result()`.

Example:
```python
future = rest_client.list_training_runs(limit=50)
response = future.result()  # or `await future`
print(f"Found {len(response.training_runs)} training runs")
print(f"Total: {response.cursor.total_count}")
# Get next page
next_page = rest_client.list_training_runs(limit=50, offset=50)
# Only runs in a given project
project_runs = rest_client.list_training_runs(project_id="my-project-id").result()
```

#### `list_training_runs_async`

```python
async def list_training_runs_async(
        limit: int = 20,
        offset: int = 0,
        access_scope: Literal["owned", "accessible"] = "owned",
        project_id: str | None = None) -> types.TrainingRunsResponse
```

Deprecated: use `await rest_client.list_training_runs(...)`; `list_training_runs()` returns
an `APIFuture` that can be awaited. This method will be removed in a subsequent release.

#### `list_checkpoints`

```python
def list_checkpoints(
    training_run_id: types.ModelID
) -> APIFuture[types.CheckpointsListResponse]
```

List available checkpoints (both training and sampler).

Args:
- `training_run_id`: The training run ID to list checkpoints for

Returns:
- An `APIFuture` containing the `CheckpointsListResponse` with available checkpoints.
    Await it, or call `.result()`.

Example:
```python
future = rest_client.list_checkpoints("run-id")
response = future.result()  # or `await future`
for checkpoint in response.checkpoints:
    if checkpoint.checkpoint_type == "training":
        print(f"Training checkpoint: {checkpoint.checkpoint_id}")
    elif checkpoint.checkpoint_type == "sampler":
        print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
```

#### `list_checkpoints_async`

```python
async def list_checkpoints_async(
        training_run_id: types.ModelID) -> types.CheckpointsListResponse
```

Deprecated: use `await rest_client.list_checkpoints(...)`; `list_checkpoints()` returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent release.

#### `get_external_weights_urls`

```python
def get_external_weights_urls(
        tinker_path: str) -> APIFuture[types.ExternalWeightsUrlsResponse]
```

Get signed download URLs, one per file, for an external weights checkpoint.

Args:
- `tinker_path`: The checkpoint's tinker path, as returned by `save_weights_external`
  (`tinker://<training_run_id>/external_weights/<name>`)

Returns:
- An `APIFuture` containing the `ExternalWeightsUrlsResponse`: `urls` maps each file path
  (relative to the checkpoint root) to a signed URL, valid until `expires`.
  Await it, or call `.result()`.

#### `get_external_weights_urls_async`

```python
async def get_external_weights_urls_async(
        tinker_path: str) -> types.ExternalWeightsUrlsResponse
```

Deprecated: use `await rest_client.get_external_weights_urls(...)`;
`get_external_weights_urls()` returns an `APIFuture` that can be awaited. This method will
be removed in a subsequent release.

#### `external_weights_upload_start`

```python
def external_weights_upload_start(
    base_model: str,
    lora_config: types.LoraConfig,
    destination_project_id: str | None = None
) -> APIFuture[types.ExternalWeightsUploadStartResult]
```

Start uploading weights to become an external weights checkpoint of a new
training run, which cannot be trained on.

Experimental: this API may change or be removed in any release.

Upload each file in the result's `urls` with an HTTP PUT of its bytes, then call
`external_weights_upload_finish`. The files must be exactly what
`save_weights_external` exports for `base_model` and `lora_config`; finishing
rejects anything else.

Args:
- `base_model`: Base model the weights were trained for
- `lora_config`: LoRA configuration the weights were trained with
- `destination_project_id`: Project that will own the checkpoint. If omitted,
  defaults to the `TINKER_PROJECT_ID` environment variable, and without it to your
  organization's Default project.

Returns:
- An `APIFuture` containing the `ExternalWeightsUploadStartResult`. Await it, or call
  `.result()`.

#### `external_weights_upload_finish`

```python
def external_weights_upload_finish(
        upload_id: str) -> APIFuture[types.ExternalWeightsUploadFinishResult]
```

Validate an upload whose files were all put to their signed URLs, and register it
as an external weights checkpoint. Retrying returns the same checkpoint.

Experimental: this API may change or be removed in any release.

Load the checkpoint into a training client of the same base model and LoRA config
with `load_state` (weights only).

Args:
- `upload_id`: The `upload_id` of `external_weights_upload_start`'s result

Returns:
- An `APIFuture` containing the `ExternalWeightsUploadFinishResult`. Await it, or call
  `.result()`.

#### `delete_checkpoint`

```python
def delete_checkpoint(training_run_id: types.ModelID,
                      checkpoint_id: str) -> APIFuture[None]
```

Delete a checkpoint for a training run.

#### `delete_checkpoint_async`

```python
async def delete_checkpoint_async(training_run_id: types.ModelID,
                                  checkpoint_id: str) -> None
```

Deprecated: use `await rest_client.delete_checkpoint(...)`; `delete_checkpoint()` returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent release.

#### `delete_checkpoint_from_tinker_path`

```python
def delete_checkpoint_from_tinker_path(tinker_path: str) -> APIFuture[None]
```

Delete a checkpoint referenced by a tinker path.

#### `delete_checkpoint_from_tinker_path_async`

```python
async def delete_checkpoint_from_tinker_path_async(tinker_path: str) -> None
```

Deprecated: use `await rest_client.delete_checkpoint_from_tinker_path(...)`;
`delete_checkpoint_from_tinker_path()` returns an `APIFuture` that can be awaited. This
method will be removed in a subsequent release.

#### `get_audit_log`

```python
def get_audit_log(
        event_type: Literal["all", "checkpoints", "projects", "teams",
                            "organizations"] = "all",
        day: date | None = None) -> APIFuture[types.AuditLogResponse]
```

Deprecated: use `AdminClient.get_audit_log()` instead. This method will be removed in a
subsequent release.

#### `get_audit_log_async`

```python
async def get_audit_log_async(
        event_type: Literal["all", "checkpoints", "projects", "teams",
                            "organizations"] = "all",
        day: date | None = None) -> types.AuditLogResponse
```

Deprecated: use `await AdminClient.get_audit_log(...)` instead. This method will be
removed in a subsequent release.

#### `copy_weights`

```python
def copy_weights(path: str,
                 destination_project_id: str | None = None,
                 *,
                 ttl_seconds: int | None = None,
                 weights_access_token: str | None = None) -> APIFuture[str]
```

Copy a checkpoint into a selected project.

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

#### `copy_weights_async`

```python
async def copy_weights_async(path: str,
                             destination_project_id: str | None = None,
                             *,
                             ttl_seconds: int | None = None,
                             weights_access_token: str | None = None) -> str
```

Deprecated: use `await rest_client.copy_weights(...)`; `copy_weights()` returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent release.

#### `publish_checkpoint_from_tinker_path`

```python
def publish_checkpoint_from_tinker_path(tinker_path: str) -> APIFuture[None]
```

Publish a checkpoint referenced by a tinker path to make it publicly accessible.

Only the exact owner of the training run can publish checkpoints.
Published checkpoints can be unpublished using the unpublish_checkpoint_from_tinker_path method.

Args:
- `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

Returns:
- An `APIFuture` that completes when the checkpoint is published.
    Await it, or call `.result()`.

Raises:
    HTTPException: 400 if checkpoint identifier is invalid
    HTTPException: 404 if checkpoint not found or user doesn't own the training run
    HTTPException: 409 if checkpoint is already public
    HTTPException: 500 if there's an error publishing the checkpoint

Example:
```python
future = rest_client.publish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()  # or `await future`
print("Checkpoint published successfully")
```

#### `publish_checkpoint_from_tinker_path_async`

```python
async def publish_checkpoint_from_tinker_path_async(tinker_path: str) -> None
```

Deprecated: use `await rest_client.publish_checkpoint_from_tinker_path(...)`;
`publish_checkpoint_from_tinker_path()` returns an `APIFuture` that can be awaited. This
method will be removed in a subsequent release.

#### `unpublish_checkpoint_from_tinker_path`

```python
def unpublish_checkpoint_from_tinker_path(tinker_path: str) -> APIFuture[None]
```

Unpublish a checkpoint referenced by a tinker path to make it private again.

Only the exact owner of the training run can unpublish checkpoints.
This reverses the effect of publishing a checkpoint.

Args:
- `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

Returns:
- An `APIFuture` that completes when the checkpoint is unpublished.
    Await it, or call `.result()`.

Raises:
    HTTPException: 400 if checkpoint identifier is invalid
    HTTPException: 404 if checkpoint not found or user doesn't own the training run
    HTTPException: 409 if checkpoint is already private
    HTTPException: 500 if there's an error unpublishing the checkpoint

Example:
```python
future = rest_client.unpublish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()  # or `await future`
print("Checkpoint unpublished successfully")
```

#### `unpublish_checkpoint_from_tinker_path_async`

```python
async def unpublish_checkpoint_from_tinker_path_async(
        tinker_path: str) -> None
```

Deprecated: use `await rest_client.unpublish_checkpoint_from_tinker_path(...)`;
`unpublish_checkpoint_from_tinker_path()` returns an `APIFuture` that can be awaited. This
method will be removed in a subsequent release.

#### `set_checkpoint_ttl_from_tinker_path`

```python
def set_checkpoint_ttl_from_tinker_path(
        tinker_path: str, ttl_seconds: int | None) -> APIFuture[None]
```

Set or remove the TTL on a checkpoint referenced by a tinker path.

If ttl_seconds is provided, the checkpoint will expire after that many seconds from now.
It must be between 1 hour (3600) and 10 years.
If ttl_seconds is None, any existing expiration will be removed.

Args:
- `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")
- `ttl_seconds`: Seconds until expiration (1 hour to 10 years), or None to remove TTL

Returns:
- An `APIFuture` that completes when the TTL is set. Await it, or call `.result()`.

Raises:
    HTTPException: 400 if checkpoint identifier is invalid or ttl_seconds is out of range
    HTTPException: 404 if checkpoint not found or user doesn't own the training run
    HTTPException: 500 if there's an error setting the TTL

Example:
```python
future = rest_client.set_checkpoint_ttl_from_tinker_path("tinker://run-id/weights/0001", 86400)
future.result()  # or `await future`
print("Checkpoint TTL set successfully")
```

#### `set_checkpoint_ttl_from_tinker_path_async`

```python
async def set_checkpoint_ttl_from_tinker_path_async(
        tinker_path: str, ttl_seconds: int | None) -> None
```

Deprecated: use `await rest_client.set_checkpoint_ttl_from_tinker_path(...)`;
`set_checkpoint_ttl_from_tinker_path()` returns an `APIFuture` that can be awaited. This
method will be removed in a subsequent release.

#### `list_user_checkpoints`

```python
def list_user_checkpoints(
        limit: int = 100,
        offset: int = 0) -> APIFuture[types.CheckpointsListResponse]
```

List all checkpoints for the current user across all their training runs.

This method retrieves checkpoints from all training runs owned by the authenticated user,
sorted by time (newest first). It supports pagination for efficiently handling large
numbers of checkpoints.

Args:
- `limit`: Maximum number of checkpoints to return (default 100)
- `offset`: Offset for pagination (default 0)

Returns:
- An `APIFuture` containing the `CheckpointsListResponse` with checkpoints and cursor info.
    Await it, or call `.result()`.

Example:
```python
future = rest_client.list_user_checkpoints(limit=50)
response = future.result()  # or `await future`
print(f"Found {len(response.checkpoints)} checkpoints")
print(f"Total: {response.cursor.total_count if response.cursor else 'Unknown'}")
for checkpoint in response.checkpoints:
    print(f"  {checkpoint.training_run_id}/{checkpoint.checkpoint_id}")
# Get next page if there are more checkpoints
if response.cursor and response.cursor.offset + response.cursor.limit < response.cursor.total_count:
    next_page = rest_client.list_user_checkpoints(limit=50, offset=50)
```

#### `list_user_checkpoints_async`

```python
async def list_user_checkpoints_async(limit: int = 100,
                                      offset: int = 0
                                      ) -> types.CheckpointsListResponse
```

Deprecated: use `await rest_client.list_user_checkpoints(...)`; `list_user_checkpoints()`
returns an `APIFuture` that can be awaited. This method will be removed in a subsequent
release.

#### `get_session`

```python
def get_session(
    session_id: str,
    access_scope: Literal["owned", "accessible"] = "owned"
) -> APIFuture[types.GetSessionResponse]
```

Get session information including all training runs and samplers.

Args:
- `session_id`: The session ID to get information for

Returns:
- An `APIFuture` containing the `GetSessionResponse` with training_run_ids, sampler_ids,
  and user_metadata. Await it, or call `.result()`.

Example:
```python
future = rest_client.get_session("session-id")
response = future.result()  # or `await future`
print(f"Training runs: {len(response.training_run_ids)}")
print(f"Samplers: {len(response.sampler_ids)}")
print(f"User metadata: {response.user_metadata}")
```

#### `get_session_async`

```python
async def get_session_async(
    session_id: str,
    access_scope: Literal["owned", "accessible"] = "owned"
) -> types.GetSessionResponse
```

Deprecated: use `await rest_client.get_session(...)`; `get_session()` returns an `APIFuture`
that can be awaited. This method will be removed in a subsequent release.

#### `list_sessions`

```python
def list_sessions(
    limit: int = 20,
    offset: int = 0,
    access_scope: Literal["owned", "accessible"] = "owned"
) -> APIFuture[types.ListSessionsResponse]
```

List sessions with pagination support.

Args:
- `limit`: Maximum number of sessions to return (default 20)
- `offset`: Offset for pagination (default 0)

Returns:
- An `APIFuture` containing the `ListSessionsResponse` with list of session IDs.
    Await it, or call `.result()`.

Example:
```python
future = rest_client.list_sessions(limit=50)
response = future.result()  # or `await future`
print(f"Found {len(response.sessions)} sessions")
# Get next page
next_page = rest_client.list_sessions(limit=50, offset=50)
```

#### `list_sessions_async`

```python
async def list_sessions_async(
    limit: int = 20,
    offset: int = 0,
    access_scope: Literal["owned", "accessible"] = "owned"
) -> types.ListSessionsResponse
```

Deprecated: use `await rest_client.list_sessions(...)`; `list_sessions()` returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent release.

#### `assign_session_project`

```python
def assign_session_project(session_id: str,
                           project_id: str) -> APIFuture[None]
```

Move a session (and all of its training runs/samplers) into a project.

Use this to attach a previously-created session to a project, or to move
a session between projects. Clearing the project is not supported — sessions
cannot be moved out of a project once placed.

Args:
- `session_id`: The session ID to move
- `project_id`: The destination project ID

Returns:
- An `APIFuture` that completes when the session has been moved.
    Await it, or call `.result()`.

Raises:
    HTTPException: 400 if `project_id` is missing
    HTTPException: 403 if the caller lacks access to the destination project
    HTTPException: 404 if the session is not found or not accessible

Example:
```python
future = rest_client.assign_session_project("session-id", "project-id")
future.result()
```

#### `assign_session_project_async`

```python
async def assign_session_project_async(session_id: str,
                                       project_id: str) -> None
```

Deprecated: use `await rest_client.assign_session_project(...)`; `assign_session_project()`
returns an `APIFuture` that can be awaited. This method will be removed in a subsequent
release.

#### `export_session_trace`

```python
def export_session_trace(session_id: str) -> APIFuture[str]
```

Export a session's timeline as a Perfetto trace and get a signed download URL.

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
- An `APIFuture` containing the signed download URL for the `.pftrace` file.
    Await it, or call `.result()`.

Raises:
    RuntimeError: if the export job fails

Example:
```python
url = rest_client.export_session_trace("session-id").result()
print(f"Download URL: {url}")
```

#### `export_session_trace_async`

```python
async def export_session_trace_async(session_id: str) -> str
```

Deprecated: use `await rest_client.export_session_trace(...)`; `export_session_trace()`
returns an `APIFuture` that can be awaited. This method will be removed in a subsequent
release.

#### `whoami`

```python
def whoami() -> APIFuture[types.WhoamiResponse]
```

Get the calling principal's identity.

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

#### `get_sampler`

```python
def get_sampler(sampler_id: str) -> APIFuture[types.GetSamplerResponse]
```

Get sampler information.

Args:
- `sampler_id`: The sampler ID (sampling_session_id) to get information for

Returns:
- An `APIFuture` containing the `GetSamplerResponse` with sampler details

Example:
```python
# Sync usage
future = rest_client.get_sampler("session-id:sample:0")
response = future.result()  # or `await future`
print(f"Base model: {response.base_model}")
print(f"Model path: {response.model_path}")

# Async usage
response = await rest_client.get_sampler("session-id:sample:0")
print(f"Base model: {response.base_model}")
```

#### `get_sampler_async`

```python
async def get_sampler_async(sampler_id: str) -> types.GetSamplerResponse
```

Deprecated: use `await rest_client.get_sampler(...)`; `get_sampler()` returns an `APIFuture`
that can be awaited. This method will be removed in a subsequent release.

#### `get_billing_usage`

```python
def get_billing_usage(
        starting_on: datetime | str, ending_before: datetime | str
) -> APIFuture[types.BillingUsageResponse]
```

Deprecated: use `AdminClient.get_billing_usage()` instead. This method will be removed in
a subsequent release.

#### `get_billing_usage_async`

```python
async def get_billing_usage_async(
        starting_on: datetime | str,
        ending_before: datetime | str) -> types.BillingUsageResponse
```

Deprecated: use `await AdminClient.get_billing_usage(...)` instead. This method will be
removed in a subsequent release.

#### `get_current_checkpoint_storage_usage`

```python
def get_current_checkpoint_storage_usage(
    project_id: str | None = None
) -> APIFuture[types.CurrentCheckpointStorageUsageResponse]
```

Deprecated: use `AdminClient.get_current_checkpoint_storage_usage()` instead. This method
will be removed in a subsequent release.

#### `get_current_checkpoint_storage_usage_async`

```python
async def get_current_checkpoint_storage_usage_async(
    project_id: str | None = None
) -> types.CurrentCheckpointStorageUsageResponse
```

Deprecated: use `await AdminClient.get_current_checkpoint_storage_usage(...)` instead.
This method will be removed in a subsequent release.

#### `search_sessions`

```python
def search_sessions(
    request: types.SessionSearchRequest
) -> APIFuture[types.SessionSearchResponse]
```

Search your organization's sessions, newest first.

Experimental: this API may change or be removed in any release.

Results are restricted to projects you may list. Page through them by
passing each response's `next_cursor` as the next request's `cursor`.

Args:
- `request`: The search filters, page size and cursor

Returns:
- An `APIFuture` containing one `SessionSearchResponse` page. Await it, or call `.result()`.

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

#### `search_checkpoints`

```python
def search_checkpoints(
    request: types.CheckpointSearchRequest
) -> APIFuture[types.CheckpointSearchResponse]
```

Search your organization's checkpoints, newest first.

Experimental: this API may change or be removed in any release.

Results are restricted to projects you may list. Page through them by
passing each response's `next_cursor` as the next request's `cursor`.

Args:
- `request`: The search filters, page size and cursor

Returns:
- An `APIFuture` containing one `CheckpointSearchResponse` page. Await it, or call `.result()`.

Example:
```python
from tinker.types import AccessibleSearchOwner, CheckpointSearchRequest

request = CheckpointSearchRequest(
    checkpoint_type="sampler", owner=AccessibleSearchOwner()
)
page = rest_client.search_checkpoints(request).result()
while True:
    for checkpoint in page.checkpoints:
        print(checkpoint.tinker_path, checkpoint.time)
    if page.next_cursor is None:
        break
    request = request.model_copy(update={"cursor": page.next_cursor})
    page = rest_client.search_checkpoints(request).result()
```
