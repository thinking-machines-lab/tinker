## `Datum` Objects

```python
class Datum(StrictBase)
```

#### `loss_fn_inputs`

Dictionary mapping field names to tensor data

#### `convert_tensors`

```python
def convert_tensors(cls, data: Any) -> Any
```

Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction.

## `ForwardBackwardInput` Objects

```python
class ForwardBackwardInput(StrictBase)
```

#### `data`

Array of input data for the forward/backward pass

#### `loss_fn`

Fully qualified function path for the loss function

#### `loss_fn_config`

Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)

## `ModelData` Objects

```python
class ModelData(BaseModel)
```

Metadata about a model's architecture and configuration.

#### `arch`

The model architecture identifier.

#### `model_name`

The human-readable model name.

#### `tokenizer_id`

The identifier of the tokenizer used by this model.

## `GetInfoResponse` Objects

```python
class GetInfoResponse(BaseModel)
```

Response containing information about a training client's model.

#### `type`

Response type identifier.

#### `model_data`

Detailed metadata about the model.

#### `model_id`

Unique identifier for the model.

#### `is_lora`

Whether this is a LoRA fine-tuned model.

#### `lora_rank`

The rank of the LoRA adaptation, if applicable.

#### `model_name`

The name of the model.

## `LoadWeightsResponse` Objects

```python
class LoadWeightsResponse(BaseModel)
```

#### `path`

A tinker URI for model weights at a specific step

#### `model_id`

Canonical id of the model the weights were loaded onto.

## `ModelInput` Objects

```python
class ModelInput(StrictBase)
```

#### `chunks`

Sequence of input chunks (formerly TokenSequence)

#### `from_ints`

```python
def from_ints(cls, tokens: List[int]) -> "ModelInput"
```

Create a ModelInput from a list of ints (tokens).

#### `to_ints`

```python
def to_ints() -> List[int]
```

Convert the ModelInput to a list of ints (tokens)
Throws exception if there are any non-token chunks

#### `length`

```python
def length() -> int
```

Return the total context length used by this ModelInput.

#### `empty`

```python
def empty(cls) -> "ModelInput"
```

Create an empty ModelInput.

#### `append`

```python
def append(chunk: ModelInputChunk) -> "ModelInput"
```

Add a new chunk, return a new ModelInput.

#### `append_int`

```python
def append_int(token: int) -> "ModelInput"
```

Add a new token, return a new ModelInput.

## `OptimStepResponse` Objects

```python
class OptimStepResponse(BaseModel)
```

#### `metrics`

Optimization step metrics as key-value pairs

## `SamplingParams` Objects

```python
class SamplingParams(BaseModel)
```

#### `max_tokens`

Maximum number of tokens to generate

#### `seed`

Random seed for reproducible generation

#### `stop`

Stop sequences for generation. ``None`` keeps implicit EOS stopping;
``[]`` disables all stop tokens, including EOS, and generation continues
until ``max_tokens``.

#### `temperature`

Sampling temperature

#### `top_k`

Top-k sampling parameter (-1 for no limit)

#### `top_p`

Nucleus sampling probability

## `SaveWeightsExternalResponseInternal` Objects

```python
class SaveWeightsExternalResponseInternal(BaseModel)
```

#### `path`

A tinker URI for the external-format model weights

#### `size_bytes`

Size of the saved external weights in bytes

## `SaveWeightsExternalResponse` Objects

```python
class SaveWeightsExternalResponse(BaseModel)
```

#### `path`

A tinker URI for the external-format model weights

## `SaveWeightsForSamplerResponseInternal` Objects

```python
class SaveWeightsForSamplerResponseInternal(BaseModel)
```

#### `path`

A tinker URI for model weights for sampling at a specific step

#### `sampling_session_id`

The generated sampling session ID

## `SaveWeightsForSamplerResponse` Objects

```python
class SaveWeightsForSamplerResponse(BaseModel)
```

#### `path`

A tinker URI for model weights for sampling at a specific step

#### `get_console_url`

```python
def get_console_url() -> str
```

Return the Tinker Console URL for this sampler checkpoint.

#### `get_playground_url`

```python
def get_playground_url() -> str
```

Return a Tinker Playground URL configured with this checkpoint.

## `TensorData` Objects

```python
class TensorData(StrictBase)
```

#### `data`

Flattened tensor data as array of numbers.

#### `shape`

Optional.

The shape of the tensor (see PyTorch tensor.shape). The shape of a
one-dimensional list of length N is `(N,)`. Can usually be inferred if not
provided, and is generally inferred as a 1D tensor.

#### `sparse_crow_indices`

Optional CSR compressed row pointers. When set, this tensor is sparse CSR:
- data contains only the listed values (flattened); every other entry takes
  the `pad_value` given when converting to or from a dense tensor
- sparse_crow_indices contains the row pointers (length = nrows + 1)
- sparse_col_indices contains the column indices (length = nnz)
- shape is required and specifies the dense shape

#### `sparse_col_indices`

Optional CSR column indices. Must be set together with sparse_crow_indices.

#### `from_torch_sparse`

```python
def from_torch_sparse(cls,
                      tensor: "torch.Tensor",
                      pad_value: int = 0) -> "TensorData"
```

Create a sparse CSR TensorData from a dense 2-D torch tensor.

Entries equal to `pad_value` are left out; the rest are stored as CSR
values. Automatically detects sparsity and encodes as CSR when it saves
space. Falls back to dense if the tensor is 1-D or mostly non-pad.

`pad_value` must be an integer: the tensor is shifted by it so torch's
zero-based CSR conversion can be reused, and an integer shift is exact
for `int64` tensors. For `float32` tensors with a non-zero pad it may
round values whose magnitude is far below `pad_value`.

#### `to_numpy`

```python
def to_numpy(pad_value: int = 0) -> npt.NDArray[Any]
```

Convert TensorData to numpy array.

A sparse CSR tensor is densified with `pad_value` in every unlisted entry.

#### `to_torch`

```python
def to_torch(pad_value: int = 0) -> "torch.Tensor"
```

Convert TensorData to torch tensor.

A sparse CSR tensor is densified with `pad_value` in every unlisted
entry. `pad_value` must be an integer so the shift around torch's
zero-based densification is exact for `int64` tensors.

TensorContainer header readers used by public SDK types.

#### `tensor_container_shape`

```python
def tensor_container_shape(data: bytes) -> tuple[int, ...]
```

Read a TensorContainer shape from bytes without deserializing payload.

#### `tensor_container_dim`

```python
def tensor_container_dim(data: bytes, dim: int) -> int
```

Read one dimension from a TensorContainer shape.

## `AuditLogEntry` Objects

```python
class AuditLogEntry(BaseModel)
```

A single entry in the audit log.

#### `timestamp`

When the event occurred.

#### `event`

`<resource>_<action>`, e.g. `checkpoint_read`, `project_grant_set`.

#### `event_details`

Who did it, what it was about, and whatever else this event records.

#### `model_id`

Deprecated: read `event_details`. Set on checkpoint events only.

#### `tinker_path`

Deprecated: read `event_details`. Set on checkpoint events only.

#### `purpose`

Deprecated: read `event_details`. Set on checkpoint reads only.

## `AuditLogResponse` Objects

```python
class AuditLogResponse(BaseModel)
```

Audit log response containing a list of entries.

#### `entries`

List of audit log entries, sorted by timestamp.

## `GetBillingUsageRequest` Objects

```python
class GetBillingUsageRequest(StrictBase)
```

Query parameters for GET /api/v1/billing/usage/events.

#### `starting_on`

Inclusive window start, aligned to a UTC hour boundary; must not be
in the future

#### `ending_before`

Exclusive window end, aligned to a UTC hour boundary; at most 14 days
after `starting_on`

## `TrainingBillingEvent` Objects

```python
class TrainingBillingEvent(BaseModel)
```

Training tokens processed by forward/backward passes.

#### `token_count`

Training token count for the bucket

## `SamplingPrefillBillingEvent` Objects

```python
class SamplingPrefillBillingEvent(BaseModel)
```

Prompt (prefill) tokens processed while sampling.

#### `cached`

True when the tokens were served from the prefill cache (billed at
the discounted cached rate); False for full-rate prefill

#### `token_count`

Prefill token count for the bucket

## `SamplingSampleBillingEvent` Objects

```python
class SamplingSampleBillingEvent(BaseModel)
```

Tokens generated while sampling.

#### `token_count`

Sampled token count for the bucket

## `CheckpointBillingEvent` Objects

```python
class CheckpointBillingEvent(BaseModel)
```

Checkpoint operations (billed per checkpoint).

#### `count`

Number of checkpoints in the bucket

## `StorageBillingEvent` Objects

```python
class StorageBillingEvent(BaseModel)
```

Checkpoint storage in the public HTTP/JSON usage response.

#### `free`

True for free storage; False for paid storage. Defaults to False for
compatibility with responses from before the free-storage flag was added.

#### `gigabyte_hours`

Storage quantity for the bucket

## `BillingUsageEvent` Objects

```python
class BillingUsageEvent(BaseModel)
```

One hourly bucket of billing usage: shared attribution dimensions on
the envelope, with the usage-kind-specific payload in `event_info` (a
tagged union discriminated on `.type`).

#### `bucket_start`

Inclusive start of the UTC hour bucket

#### `bucket_end`

Exclusive end of the UTC hour bucket

#### `base_model`

Base model the usage was billed against (None when unknown, e.g.
storage from before base-model stamping)

#### `user_id`

Organization-user urn of the user that created the session the usage
is attributed to

#### `user_name`

Display name of `user_id`, as self-reported during onboarding

#### `session_id`

The session the usage is attributed to: for sampling the session that
issued the requests, for training the session that created the training
run. None for events without a session (storage, usage from before
session tagging). Session user_metadata is available in
`BillingUsageResponse.sessions` — keyed by this id.

#### `project_id`

Project this usage belongs to, resolved from the current project
associated with the session identified by `session_id`

#### `estimated_cost_usd`

Estimated gross USD usage cost before credits and commits, not invoice
amount due. For a completed UTC day, each model/usage-category group is
priced only after its full-day quantity reconciles with usage line items
from the finalized invoice export, or the latest draft when no
finalized invoice is available. Other groups on the same day may remain
unpriced. The current incomplete UTC day instead uses the published Tinker
rate-card snapshot; that estimate is not invoice-reconciled and can change
when the day completes.
For token events, this equals `effective_rate_usd_per_million_tokens *
token_count / 1_000_000`. For storage events, this equals
`effective_rate_usd_per_gigabyte_month * gigabyte_hours / 720`. None for
checkpoint operations, when the applicable completed-day group has not
reconciled, or when the current rate card has no applicable token or
storage rate.

#### `effective_rate_usd_per_million_tokens`

Effective gross USD rate per million tokens used to calculate
`estimated_cost_usd`, before credits and commits. This is not an amount
due. None for non-token events or whenever token cost cannot be
attributed.

#### `effective_rate_usd_per_gigabyte_month`

Effective gross USD rate per storage GB-month used to calculate
`estimated_cost_usd`, before credits and commits. One GB-month is 720
GB-hours for this calculation. None for non-storage events or whenever
storage cost cannot be attributed.

#### `event_info`

What kind of usage this is and its quantity; dispatch on
`event_info.type`

## `BillingUsageSession` Objects

```python
class BillingUsageSession(BaseModel)
```

Session-level attributes tied to a session.

#### `user_metadata`

The session's customer-supplied user_metadata (as passed to
CreateSessionRequest.user_metadata); None when the session has none.

## `BillingUsageResponse` Objects

```python
class BillingUsageResponse(BaseModel)
```

#### `data`

Hourly usage events, ordered by bucket then descending token count

#### `sessions`

session_id -> that session's attributes, for every distinct session
appearing in `data`

#### `cost_data_through`

Exclusive UTC boundary between the two pricing sources used for
estimated costs. Non-null estimates before it use invoice-derived effective
rates, while newer non-null estimates use fixed published rates. A cost or
rate may still be None when the data required to price that row is
unavailable. None means the response contains no completed day with invoice
pricing data.

## `Checkpoint` Objects

```python
class Checkpoint(BaseModel)
```

#### `checkpoint_id`

The checkpoint ID

#### `checkpoint_type`

The type of checkpoint (training, sampler, or external)

#### `time`

The time when the checkpoint was created

#### `tinker_path`

The tinker path to the checkpoint

#### `size_bytes`

The size of the checkpoint in bytes

#### `public`

Whether the checkpoint is publicly accessible

#### `user_metadata`

Optional user-provided metadata attached to the checkpoint

#### `expires_at`

When this checkpoint expires (None = never expires)

#### `base_model`

The base model the checkpoint was trained from

## `ParsedCheckpointTinkerPath` Objects

```python
class ParsedCheckpointTinkerPath(BaseModel)
```

#### `tinker_path`

The tinker path to the checkpoint

#### `training_run_id`

The training run ID

#### `checkpoint_type`

The type of checkpoint (training, sampler, or external)

#### `checkpoint_id`

The checkpoint ID

#### `from_tinker_path`

```python
def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath"
```

Parse a tinker path to an instance of ParsedCheckpointTinkerPath

## `CheckpointSearchCursor` Objects

```python
class CheckpointSearchCursor(BaseModel)
```

Where the next page starts; pass a response's `next_cursor` back unchanged.

## `CheckpointSearchRequest` Objects

```python
class CheckpointSearchRequest(StrictBase)
```

A checkpoint search. Results are restricted to projects the caller may list.

#### `project_ids`

Projects to search; None searches every listable project. Naming a
project the caller cannot list is an error

#### `user_metadata`

Exact user_metadata key/value matches; every pair must hold

#### `checkpoint_type`

Checkpoint type to match; None matches every type

#### `owner`

Whose checkpoints to search; defaults to the caller's own

#### `base_models`

Match any listed base model; None matches every base model. An empty
list matches no checkpoints

#### `limit`

Page size, 1 to 200

#### `cursor`

`next_cursor` of the previous page; None for the first page

## `CheckpointSearchResponse` Objects

```python
class CheckpointSearchResponse(BaseModel)
```

One page of search results, newest first.

#### `next_cursor`

Pass as the next request's `cursor`; None once the results are exhausted

## `CheckpointsListResponse` Objects

```python
class CheckpointsListResponse(BaseModel)
```

#### `checkpoints`

List of available model checkpoints for the model

#### `cursor`

Pagination cursor information (None for unpaginated responses)

## `ClientConfigRequest` Objects

```python
class ClientConfigRequest(StrictBase)
```

#### `sdk_version`

The SDK version string for flag resolution.

## `ClientConfigResponse` Objects

```python
class ClientConfigResponse(BaseModel)
```

Server-side feature flags resolved for this caller.

Uses BaseModel (extra="ignore") so new flags from the server are
silently dropped until the SDK adds fields for them.

#### `proto_compress_fwdbwd`

When true, the SDK zstd-compresses the proto fwd/bwd request body and
sets Content-Encoding: zstd. Real fwd/bwd payloads compress >10× — the API
server decompresses transparently via an ASGI middleware.

#### `fwdbwd_max_chunk_len`

Maximum number of datums per fwd/bwd chunk request.

#### `fwdbwd_max_chunk_bytes_count`

Maximum estimated request payload bytes (serialized proto, before zstd
compression) per fwd/bwd chunk request.

#### `fwdbwd_dispatch_bytes_semaphore_size`

Maximum estimated fwd/bwd request payload bytes being dispatched at the
same time (same pre-compression estimate as fwdbwd_max_chunk_bytes_count).
Bounds client memory for serialized bodies and smooths submission bursts.

#### `sample_enable_stuck_detection`

When true, the SDK runs the retry handler's progress timeout check that
raises ``APIConnectionError("...Requests appear to be stuck.")`` when no
progress is made within ``RetryConfig.progress_timeout``.

#### `sample_max_concurrent_requests`

Maximum number of in-flight sampling requests a SamplingClient will allow
(the size of its retry handler's semaphore). Always applied as the sampling
``RetryConfig.max_connections``, overriding any value in a caller-provided
``retry_config``.

#### `use_pyqwest_transport`

When true, the SDK builds its default httpx async client on top of the
pyqwest (reqwest/hyper-based) transport adapter. Set to false server-side
to force every client to fall back to httpx's default transport.

#### `sample_use_retrieve_futures`

When true, each SamplingClient runs a single background task that polls
``/api/v1/retrieve_futures`` for its whole sampling session; a sample's
result future waits for that poller to signal its request complete (and
carries the completion metadata) before doing the normal payload fetch,
instead of polling ``/api/v1/retrieve_future`` per request. Requires a
server exposing the retrieve_futures endpoint.

#### `custom_loss_server_side`

When true, ``ServiceClient.create_custom_loss`` uploads the traced loss
(``/api/v1/create_custom_loss``) and ``forward_backward_custom`` runs it on the
server by its loss id. When false, custom losses run client-side. Requires a
server that stores and runs custom losses.

#### `sample_join_sampling_session`

When true, an unpickled SamplingClient asks the server for its clone id
(``/api/v1/join_sampling_session``) before its first sample, instead of
picking a random one, so clones of a sampling session never collide on
request seq_ids. Requires a server exposing the join_sampling_session
endpoint.

## `ClientDynamicConfigResponse` Objects

```python
class ClientDynamicConfigResponse(BaseModel)
```

Server-side flags re-fetched periodically by a live client.

Unlike ClientConfigResponse (fetched once at client creation), these
flags are refreshed in the background at refresh_interval_sec cadence,
so server-side changes take effect on the next request without
recreating the client. Uses BaseModel (extra="ignore") so new flags
from the server are silently dropped until the SDK adds fields for
them.

#### `refresh_interval_sec`

How often the SDK re-fetches this config from the server.

#### `sample_cancel_enabled`

When true, abandoning a sampling future (timeout or local cancel) sends
an explicit cancel_future request to the server so the in-flight sampling
work is stopped promptly. When false, cancellation falls back to the
server's SDK-heartbeat-expiry path.

#### `sample_cancel_max_batch_size`

Maximum number of queued sampling cancellations the client dispatches in
parallel per drain iteration.

## `CopyWeightsRequest` Objects

```python
class CopyWeightsRequest(StrictBase)
```

Provide a destination project, or both a session ID and its model sequence ID.

#### `session_id`

Session that will own the copy; the copy inherits its project.

#### `model_seq_id`

Training run slot within the session, as CreateModelRequest and LoadWeightsRequest use.

#### `destination_project_id`

Project that will own the copy when no session_id is provided.

#### `source_path`

A tinker URI for the weights to copy. Either kind is accepted.

#### `ttl_seconds`

Seconds until the copy expires, 1 hour (3600) to 10 years. Source expiry is not inherited.

#### `weights_access_token`

Optional access token for copying weights under a different account.

## `CopyWeightsResponse` Objects

```python
class CopyWeightsResponse(BaseModel)
```

#### `tinker_path`

Path of the newly created copy.

## `CreateCustomLossRequest` Objects

```python
class CreateCustomLossRequest(StrictBase)
```

#### `inputs`

The graph's inputs in order, ``logprobs`` first, with their declared ranks.

## `CreateCustomLossResponse` Objects

```python
class CreateCustomLossResponse(BaseModel)
```

#### `loss_id`

Names the stored loss in ``forward_backward`` requests of the same session.

## `CreateModelRequest` Objects

```python
class CreateModelRequest(StrictBase)
```

#### `base_model`

The name of the base model to fine-tune (e.g., 'Qwen/Qwen3-8B').

#### `user_metadata`

Optional metadata about this model/training run, set by the end-user.

#### `lora_config`

LoRA configuration

#### `optimizer_config`

Optimizer identity fixed for the lifetime of the training model.

## `CreateSamplingSessionRequest` Objects

```python
class CreateSamplingSessionRequest(StrictBase)
```

#### `session_id`

The session ID to create the sampling session within

#### `sampling_session_seq_id`

Sequence ID for the sampling session within the session

#### `base_model`

Optional base model name to sample from.

Is inferred from model_path, if provided. If sampling against a base model, this
is required.

#### `model_path`

Optional tinker:// path to your model weights or LoRA weights.

If not provided, samples against the base model.

## `CreateSamplingSessionResponse` Objects

```python
class CreateSamplingSessionResponse(BaseModel)
```

#### `sampling_session_id`

The generated sampling session ID

## `CreateSessionRequest` Objects

```python
class CreateSessionRequest(StrictBase)
```

#### `display_name`

Optional human-readable display name for the session. Does not need to be unique. Only used for display and search purposes.

## `GetCurrentCheckpointStorageUsageRequest` Objects

```python
class GetCurrentCheckpointStorageUsageRequest(StrictBase)
```

HTTP query parameters sent by the SDK and CLI to the API server.

Used by GET /api/v1/billing/usage/checkpoints/current; no JSON request body.
With no parameters, returns checkpoint storage usage for the entire
authenticated organization.

#### `project_id`

Only return checkpoints attributed to this project; omit for the entire org

## `CurrentCheckpointStorageUsageItem` Objects

```python
class CurrentCheckpointStorageUsageItem(BaseModel)
```

HTTP/JSON response item returned by the API server to SDK and CLI consumers.

Contains current active checkpoint storage attributed through its session.

#### `project_id`

Project currently associated with the checkpoint's session

#### `org_user_urn`

Organization-user URN of the checkpoint's session owner

#### `user_email`

Current email of the checkpoint's session owner

#### `user_name`

Current display name of the checkpoint's session owner

#### `checkpoint_count`

Number of active checkpoints

#### `size_bytes`

Current stored bytes

#### `size_gigabytes`

Current stored bytes converted to the storage billing unit

#### `estimated_monthly_cost_usd`

Projected gross 720-hour cost at the current storage rate

## `CurrentCheckpointStorageUsageResponse` Objects

```python
class CurrentCheckpointStorageUsageResponse(BaseModel)
```

HTTP/JSON response returned by the API server to SDK and CLI consumers.

Contains current checkpoint storage usage for the authenticated organization.
Data can lag by 1-2 hours.

#### `effective_rate_usd_per_gigabyte_month`

Current gross storage rate used for projected costs

#### `data`

Storage grouped by project and session owner

## `Cursor` Objects

```python
class Cursor(BaseModel)
```

#### `offset`

The offset used for pagination

#### `limit`

The maximum number of items requested

#### `total_count`

The total number of items available

## `CustomLossNode` Objects

```python
class CustomLossNode(StrictBase)
```

One node of a traced custom loss, as torch.fx records it.

#### `name`

Unique within the graph; later nodes refer to this node by it.

#### `target`

The parameter name (placeholder), the function's qualified name (call_function,
e.g. ``torch.exp`` or ``operator.mul``), the method or attribute name, or ``output``.

#### `args`

Encoded arguments; an output node's one argument is what the loss returns.
JSON scalars and strings stand for themselves and lists for lists. Anything else
is a one-key object: ``{"ref": name}`` for a node, ``{"tuple": [...]}``,
``{"dict": {...}}``, ``{"slice": [start, stop, step]}``, ``{"ellipsis": true}``,
``{"float": "inf"}`` (also ``"-inf"`` and ``"nan"``), ``{"dtype": "float32"}``,
``{"device": "cpu"}``, ``{"layout": "strided"}`` or
``{"memory_format": "contiguous_format"}``.

#### `kwargs`

Encoded keyword arguments, in the same form as ``args``.

## `CustomLossGraph` Objects

```python
class CustomLossGraph(StrictBase)
```

A traced custom loss as plain data: its nodes in execution order. Nothing in it
is executable until the server rebuilds it from its own list of allowed
operations.

#### `CustomLossInputRank`

A custom loss input's rank: 0 is a scalar, 1 is ``[T]`` and 2 is ``[T, K]``.

## `CustomLossInput` Objects

```python
class CustomLossInput(StrictBase)
```

One input of a custom loss, with the rank every datum's value has.

#### `name`

``logprobs``, or the ``loss_fn_inputs`` key the loss reads.

## `Datum` Objects

```python
@dataclass(frozen=True)
class Datum()
```

#### `model_input_spans`

Provenance of the input tokens as assembled: consecutive runs tiling
``model_input`` (lengths sum to its token length; position comes from
order). Attach via ``with_provenance``.

#### `loss_fn_input_spans`

Attribution of the loss rows: consecutive runs tiling
``loss_fn_inputs["target_tokens"]`` (lengths sum to its length). Attach
via ``with_provenance``.

#### `with_provenance`

```python
def with_provenance(model_input: ProvenanceSpans,
                    loss_fn_inputs: ProvenanceSpans) -> Self
```

Return a copy carrying provenance for both span-annotated fields.

Each keyword names the Datum field whose positions its runs tile —
``model_input`` spans tile the input tokens, ``loss_fn_inputs``
spans tile ``loss_fn_inputs["target_tokens"]``. Tiling is validated
here, so a miscounted partition fails at the call site rather than
as a request error.

## `DmelChunk` Objects

```python
class DmelChunk(StrictBase)
```

#### `dmel`

Serialized TensorContainer bytes holding the DMel model token tensor.

#### `validate_dmel`

```python
def validate_dmel(cls, value: Union[bytes, str]) -> bytes
```

Deserialize base64 string to bytes if needed.

#### `serialize_dmel`

```python
def serialize_dmel(value: bytes) -> str
```

Serialize bytes to base64 string for JSON.

## `EncodedTextChunk` Objects

```python
class EncodedTextChunk(StrictBase)
```

#### `tokens`

Array of token IDs

## `ExternalWeightsUploadStartResult` Objects

```python
@dataclass(frozen=True)
class ExternalWeightsUploadStartResult()
```

An upload started by `RestClient.external_weights_upload_start`.

#### `upload_id`

Identifies the upload when finishing it

#### `urls`

Signed URL per file to upload, keyed by file name; upload each file with an HTTP PUT

#### `expires`

When the signed URLs expire

## `ExternalWeightsUploadFinishResult` Objects

```python
@dataclass(frozen=True)
class ExternalWeightsUploadFinishResult()
```

The checkpoint `RestClient.external_weights_upload_finish` made from an upload.

#### `tinker_path`

Tinker path of the external weights checkpoint holding the uploaded weights

## `ExternalWeightsUploadStartRequest` Objects

```python
class ExternalWeightsUploadStartRequest(StrictBase)
```

Transport: HTTP/JSON request body of /api/v1/external_weights_upload_start, from the SDK
to the API server, which also stores it with the upload; never stored by the SDK.

## `ExternalWeightsUploadStartResponse` Objects

```python
class ExternalWeightsUploadStartResponse(BaseModel)
```

Transport: HTTP/JSON response of /api/v1/external_weights_upload_start, from the API
server to the SDK; never stored.

## `ExternalWeightsUploadFinishRequest` Objects

```python
class ExternalWeightsUploadFinishRequest(StrictBase)
```

Transport: HTTP/JSON request body of /api/v1/external_weights_upload_finish, from the SDK
to the API server; never stored.

## `ExternalWeightsUploadFinishResponse` Objects

```python
class ExternalWeightsUploadFinishResponse(BaseModel)
```

Transport: HTTP/JSON response of /api/v1/external_weights_upload_finish, from the API
server to the SDK; never stored.

## `ExternalWeightsUrlsResponse` Objects

```python
class ExternalWeightsUrlsResponse(BaseModel)
```

#### `urls`

Signed download URL per file, keyed by the file path relative to the checkpoint root

#### `expires`

When the signed URLs expire

## `ForwardBackwardInput` Objects

```python
@dataclass(frozen=True)
class ForwardBackwardInput()
```

#### `data`

Array of input data for the forward/backward pass

#### `loss_fn`

Fully qualified function path for the loss function

#### `loss_fn_config`

Optional configuration parameters for the loss function (e.g., PPO clip thresholds, StableReinforce kl_coeff)

## `ForwardBackwardOutput` Objects

```python
@dataclass(frozen=True)
class ForwardBackwardOutput()
```

#### `loss_fn_output_type`

The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord').

#### `loss_fn_outputs`

List of per-datum dicts mapping field names to ``TensorData``.

#### `metrics`

Training metrics as key-value pairs.

The following metrics are recorded only during MoE (Mixture of Experts) training.

- ``e_frac_with_tokens:mean``: Fraction of experts that received at least one token,
  averaged across layers. A value of 1.0 means every expert got work; 0.5 means half
  were idle. Decreasing over time is concerning (routing collapse).

- ``e_frac_oversubscribed:mean``: Fraction of experts receiving more tokens than
  perfect balance, averaged across layers. Increasing over time is concerning.

- ``e_max_violation:mean``: How much the most overloaded expert exceeds perfect
  balance, as a fraction of perfect balance, averaged across layers. Computed as
  ``(max_tokens - perfect_balance) / perfect_balance``. A value of 2.0 means the
  busiest expert got 3x the fair share. Increasing over time is concerning.

- ``e_max_violation:max``: Same as ``e_max_violation:mean`` but takes the max
  across layers instead of the mean.

- ``e_min_violation:mean``: How much the least loaded expert is below perfect
  balance, as a fraction of perfect balance, averaged across layers. Typically
  negative; decreasing (more negative) is concerning.

## `FutureFinished` Objects

```python
class FutureFinished(BaseModel)
```

A finished request reported by ``/api/v1/retrieve_futures``.

Carries the uncompressed response-payload size (as retrieve_future's
metadata-only response does) so the SDK can reserve the inflight-bytes budget
before fetching the payload.

## `FutureFailed` Objects

```python
class FutureFailed(BaseModel)
```

A failed request reported by ``/api/v1/retrieve_futures`` (no payload).

## `FutureRetrieveRequest` Objects

```python
class FutureRetrieveRequest(StrictBase)
```

#### `request_id`

The ID of the request to retrieve

#### `allow_metadata_only`

When True, the server may return only response metadata (status and size)
instead of the full payload if the response exceeds the server's inline size limit.

## `SamplingSessionFuturesTarget` Objects

```python
class SamplingSessionFuturesTarget(StrictBase)
```

Poll target: one (possibly cloned) sampling session.

``cloned_sampler_id`` is ``seq_id // 1_000_000_000`` for cloned
SamplingClients (0 for the original).

## `FuturesRetrieveRequest` Objects

```python
class FuturesRetrieveRequest(StrictBase)
```

#### `prev_cursor`

Opaque cursor from the previous response; events at or before it are
considered seen and get trimmed server-side. 0 starts from the beginning.

#### `retrieved_request_ids`

Requests of the target whose results have been fully retrieved, so the
server may delete their payloads.

## `FuturesRetrieveResponse` Objects

```python
class FuturesRetrieveResponse(BaseModel)
```

#### `cursor`

Opaque cursor to pass back as ``prev_cursor`` on the next poll.

## `GenericEvent` Objects

```python
class GenericEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `event_name`

Low-cardinality event name

#### `severity`

Log severity level

#### `event_data`

Arbitrary structured JSON payload

## `SupportedModel` Objects

```python
class SupportedModel(BaseModel)
```

Information about a model supported by the server.

#### `model_name`

The name of the supported model.

#### `max_context_length`

The maximum context length (in tokens) supported by this model.

#### `trainable`

Whether training runs can be created on this model.

#### `sampleable`

Whether this model has sampling capacity.

## `GetServerCapabilitiesResponse` Objects

```python
class GetServerCapabilitiesResponse(BaseModel)
```

Response containing the server's supported models and capabilities.

#### `supported_models`

List of models available on the server.

## `GetSessionResponse` Objects

```python
class GetSessionResponse(BaseModel)
```

#### `user_metadata`

Optional metadata attached to this session by the user.

#### `display_name`

Optional human-readable display name for the session. Does not need to be unique. Only used for display and search purposes.

## `ImageAssetPointerChunk` Objects

```python
class ImageAssetPointerChunk(StrictBase)
```

#### `format`

Image format

#### `location`

Path or URL to the image asset

#### `expected_tokens`

Expected number of tokens this image represents.
This is only advisory: the tinker backend will compute the number of tokens
from the image, and we can fail requests quickly if the tokens does not
match expected_tokens.

## `ImageChunk` Objects

```python
class ImageChunk(StrictBase)
```

#### `data`

Image data as bytes

#### `format`

Image format

#### `expected_tokens`

Expected number of tokens this image represents.
This is only advisory: the tinker backend will compute the number of tokens
from the image, and we can fail requests quickly if the tokens does not
match expected_tokens.

#### `validate_data`

```python
def validate_data(cls, value: Union[bytes, str]) -> bytes
```

Deserialize base64 string to bytes if needed.

#### `serialize_data`

```python
def serialize_data(value: bytes) -> str
```

Serialize bytes to base64 string for JSON.

## `JoinSamplingSessionRequest` Objects

```python
class JoinSamplingSessionRequest(StrictBase)
```

#### `sampling_session_id`

The existing sampling session to join

## `JoinSamplingSessionResponse` Objects

```python
class JoinSamplingSessionResponse(BaseModel)
```

#### `client_counter`

Server-allocated id, unique among the sampling session's clients (the original client is 0)

## `LoadWeightsRequest` Objects

```python
class LoadWeightsRequest(StrictBase)
```

#### `model_id`

Legacy addressing: a load right after create_model (with seq_id 1).
Set together with seq_id, and mutually exclusive with session_id.

#### `session_id`

Create-via-load addressing: the load is the model's first request
(seq id 0, no create_model). Set together with model_seq_id; mirrors
CreateModelRequest minus lora_config, which the server derives from the
checkpoint.

#### `base_model`

Optional base model override for create-via-load; must be compatible
with the checkpoint's base model (e.g. a different context length).

#### `user_metadata`

Optional metadata about this model/training run, set by the end-user.

#### `path`

A tinker URI for model weights at a specific step

#### `optimizer`

Whether to load optimizer state along with model weights

#### `optimizer_config`

Optimizer for a new weights-only model. Defaults to Adam; cannot override
restored optimizer state or an existing model.

#### `weights_access_token`

Optional access token for loading checkpoints under a different account.

## `LoraConfig` Objects

```python
class LoraConfig(StrictBase)
```

#### `rank`

LoRA rank (dimension of low-rank matrices)

#### `seed`

Seed used for initialization of LoRA weights.

Useful if you need deterministic or reproducible initialization of weights.

#### `train_unembed`

Whether to add lora to the unembedding layer

#### `train_mlp`

Whether to add loras to the MLP layers (including MoE layers)

#### `train_attn`

Whether to add loras to the attention layers

Optimizer selection and per-step hyperparameters for training models.

## `OptimizerConfigBase` Objects

```python
class OptimizerConfigBase(StrictBase)
```

Optimizer family selected at model creation, identified by `type`; keeps family fields.

## `DimuonOptimizerConfig` Objects

```python
class DimuonOptimizerConfig(OptimizerConfigBase)
```

Selects the Dimuon optimizer for a training model.

## `AdamParams` Objects

```python
class AdamParams(StrictBase)
```

#### `learning_rate`

Learning rate for the optimizer

#### `beta1`

Coefficient used for computing running averages of gradient

#### `beta2`

Coefficient used for computing running averages of gradient square

#### `eps`

Term added to the denominator to improve numerical stability

#### `weight_decay`

Weight decay for the optimizer. Uses decoupled weight decay.

#### `grad_clip_norm`

Maximum global gradient norm. If the global gradient norm is greater than this value, it will be clipped to this value. 0.0 means no clipping.

## `OptimParamsBase` Objects

```python
class OptimParamsBase(StrictBase)
```

Per-step hyperparameters of a non-Adam optimizer family, identified by `type`.

## `DimuonParams` Objects

```python
class DimuonParams(OptimParamsBase)
```

Hyperparameters for one Dimuon optimizer step.

#### `type`

Optimizer family for this step.

#### `learning_rate`

Sets the step size for the optimizer.

#### `grad_clip_norm`

Maximum global gradient norm. Nonpositive values disable gradient clipping.

#### `beta1`

EMA coefficient for the momentum. Should be between zero and one.

## `PromptAltTokens` Objects

```python
@dataclass(frozen=True)
class PromptAltTokens()
```

Independent draws from the model's next-token distribution at every prompt
position after the first, taken in the prefill that served the request
(``SampleRequest.prompt_alt_tokens_k``).

Both tensors are dense ``[len(prompt) - 1, k]``. Row ``i`` holds ``k`` draws,
with replacement and at the request's temperature, from the distribution
over prompt token ``i + 1`` (position 0 has no preceding context), so
``tokens[i]`` are alternatives to ``prompt[i + 1]``.

#### `tokens`

int64 token ids, shape ``[len(prompt) - 1, k]``.

#### `logprobs`

float32 logprobs, shape ``[len(prompt) - 1, k]``: ``logprobs[i][j]`` is the
model's logprob of ``tokens[i][j]`` at prompt position ``i + 1``.

## `PromptProvenanceSpan` Objects

```python
@dataclass(frozen=True, kw_only=True)
class PromptProvenanceSpan()
```

A run of tokens that were provided as sampling input.

Claims that the covered datum positions are positions
``[offset, offset + length)`` of the named sequence's request prompt.
Runs tile the field they annotate (see ``Datum.with_provenance``):
every position is in exactly one run and position comes from order.

#### `sequence_id`

``SampledSequence.sequence_id`` of a request whose prompt contained
this run.

#### `offset`

Position of the run within that request's prompt.

#### `length`

Number of consecutive tokens in this run (must be >= 0).

## `SampledProvenanceSpan` Objects

```python
@dataclass(frozen=True, kw_only=True)
class SampledProvenanceSpan()
```

A run of tokens produced by one sampled sequence.

Claims that the covered datum positions are positions
``[offset, offset + length)`` of the named sequence's sampled tokens.
Two sampled sequences back to back are two adjacent runs, each naming
its own sequence.

#### `sequence_id`

``SampledSequence.sequence_id`` of the sequence this run reproduces.

#### `offset`

Position of the run within that sequence's sampled tokens.

#### `length`

Number of covered tokens (must be >= 0).

#### `ProvenanceSpans`

Consecutive runs tiling one Datum field; see ``Datum.with_provenance``.

## `SampleRequest` Objects

```python
class SampleRequest(StrictBase)
```

#### `num_samples`

Number of samples to generate

#### `base_model`

Optional base model name to sample from.

Is inferred from model_path, if provided. If sampling against a base model, this
is required.

#### `model_path`

Optional tinker:// path to your model weights or LoRA weights.

If not provided, samples against the base model.

#### `sampling_session_id`

Optional sampling session ID to use instead of model_path/base_model.

If provided along with seq_id, the model configuration will be loaded from the
sampling session. This is useful for multi-turn conversations.

#### `seq_id`

Sequence ID within the sampling session.

Required when sampling_session_id is provided. Used to generate deterministic
request IDs for the sampling request.

#### `prompt_logprobs`

If set to `true`, computes and returns logprobs on the prompt tokens.

Defaults to false.

#### `topk_prompt_logprobs`

If set to a positive integer, returns the top-k logprobs for each prompt token.

#### `topk_sample_logprobs`

If set to a positive integer, returns the top-k logprobs for each sampled token
(see ``SampledSequence.topk_logprobs``).

#### `target_prompt_logprobs`

Token ids whose prompt logprobs to return, as an int64 tensor of shape
``[len(prompt) - 1, K]``: cell ``[i, j]`` is scored at prompt position
``i + 1`` (position 0 has no preceding context). A cell of ``-1`` requests
nothing and gets no logprob. May be sparse CSR, in which case only the
listed cells request anything. ``SampleResponse.target_prompt_logprobs``
has the same shape and layout.

#### `prompt_alt_tokens_k`

If set to a positive integer, draws that many tokens, independently and at
the request's temperature, from the model's next-token distribution at every
prompt position after the first, in the same prefill that serves the request
(see ``SampleResponse.prompt_alt_tokens``).

#### `prompt_logprobs_last_n`

Score only the last N prompt tokens: ``prompt_logprobs[-N:]`` are scored and
earlier entries are ``None``, and the server can serve the prefix before them from
its prefix cache. Requires ``prompt_logprobs``, and must be between 1 and
``len(prompt) - 1``; also applies to ``topk_prompt_logprobs``. None scores the whole
prompt.

#### `record_stability_info`

If set to `true`, the server records stability information for each
sampled sequence, referenced from training requests by the sequence's
``sequence_id`` (StableReinforce training).

Defaults to false.

## `SampleResponse` Objects

```python
@dataclass(frozen=True)
class SampleResponse()
```

Response from a sampling request.

Contains generated sequences and optional prompt-level log probabilities.
Numpy fields provide direct array access without format conversion.
The corresponding Python-list properties convert lazily on first access.

#### `sequences`

Generated sequences. Each contains token IDs, optional logprobs, and stop reason.

#### `prompt_logprobs_np`

Per-token log probabilities for the prompt as a 1-D float32 numpy array,
shape ``(prompt_length,)``. ``NaN`` at positions where logprobs were not
computed (e.g. the first prompt token).
None if prompt logprobs were not requested.

#### `topk_prompt_logprobs_np`

Top-k prompt logprobs as a pair of dense matrices
(see ``TopkLogprobs``).
None if top-k was not requested.

#### `target_prompt_logprobs`

Logprobs of the ids in ``SampleRequest.target_prompt_logprobs``: a float32
tensor of the same shape and layout, 0.0 where the request had ``-1``.
None if not requested.

#### `prompt_cache_hit_tokens`

Number of prompt tokens billed as prefix-cache hits.

Counted on the prompt itself: for ``num_samples > 1`` the prompt is
shared, so this is not multiplied across samples. Prefill on the shared prompts
samples (the remaining ``num_samples - 1``) is billed as cache hits.

#### `prompt_alt_tokens`

The draws ``SampleRequest.prompt_alt_tokens_k`` asked for: ``k`` token ids
and their logprobs at every prompt position after the first, as two
``[len(prompt) - 1, k]`` tensors (see ``PromptAltTokens``).
None if not requested.

#### `prompt_logprobs`

```python
def prompt_logprobs() -> Optional[List[Optional[float]]]
```

Per-token log probabilities for the prompt as a Python list.

If prompt_logprobs was set to true in the request, logprobs are
computed for every token in the prompt, or for the last
``SampleRequest.prompt_logprobs_last_n`` tokens if set. Each entry
is a float, or
``None`` for positions where logprobs were not computed (e.g. the
first prompt token). Returns ``None`` if prompt logprobs were not
requested.

Converted from ``prompt_logprobs_np`` on first access (cached afterwards).

#### `topk_prompt_logprobs`

```python
def topk_prompt_logprobs(
) -> Optional[list[Optional[list[tuple[int, float]]]]]
```

Top-k prompt logprobs as nested Python lists.

If topk_prompt_logprobs was set to a positive integer k in the request,
the top-k logprobs are computed for every token in the prompt, or for the last
``SampleRequest.prompt_logprobs_last_n`` tokens if set.
For each prompt position: a list of up to k ``(token_id, logprob)``
tuples, or ``None`` for positions where logprobs were not computed.
Returns ``None`` if top-k was not requested.

Converted from ``topk_prompt_logprobs_np`` on first access (cached afterwards).

## `SampledSequence` Objects

```python
@dataclass(frozen=True)
class SampledSequence()
```

A single sampled sequence from the model.

Provides two ways to access token data:

- **Numpy arrays** (``tokens_np``, ``logprobs_np``): As numpy arrays
  without format conversion.
- **Python lists** (``tokens``, ``logprobs``): Standard Python lists,
  converted lazily on first access.

#### `stop_reason`

Reason why sampling stopped.

#### `sequence_id`

Session-scoped identifier, distinct per sequence; use it wherever a
later request needs to reference this exact sequence.

#### `tokens_np`

Generated token IDs as a 1-D int32 numpy array, shape ``(num_tokens,)``.

#### `logprobs_np`

Log probabilities for each generated token as a 1-D float32 numpy array,
shape ``(num_tokens,)``. None if logprobs were not requested.

#### `topk_logprobs_np`

Top-k logprobs at each generated position as a pair of dense
``(num_tokens, k)`` matrices (see ``TopkLogprobs``).
None if ``SampleRequest.topk_sample_logprobs`` was not requested.

#### `tokens`

```python
def tokens() -> List[int]
```

Generated token IDs as a Python list.

Converted from ``tokens_np`` on first access (cached afterwards).

#### `logprobs`

```python
def logprobs() -> Optional[List[float]]
```

Log probabilities for each generated token (optional).

None if logprobs were not requested. Converted from ``logprobs_np``
on first access (cached afterwards).

#### `topk_logprobs`

```python
def topk_logprobs() -> Optional[List[Optional[List[tuple[int, float]]]]]
```

Top-k logprobs at each generated position as nested Python lists.

If ``SampleRequest.topk_sample_logprobs`` was set to a positive integer k in
the request, each generated position gets a list of up to k
``(token_id, logprob)`` tuples, or ``None`` where the engine reported
nothing. Returns ``None`` if top-k was not requested.

Converted from ``topk_logprobs_np`` on first access (cached afterwards).

#### `SamplingClientKind`

Which weights a sampling client uses, sent in sampler-save requests to the API.

- `snapshot` (default) captures fixed weights. Use it when inference must stay on
  one saved set of weights.
- `inflight` publishes mutable weights shared by this training model's inflight
  sampling clients.

Inflight sampling contract:

- All inflight sampling clients for this training model refer to the same mutable
  weights target. Later inflight saves update that target for both existing and
  newly returned clients. Updates propagate asynchronously across serving replicas.
- Save completion means the checkpoint is saved and the shared target is updated.
  It does not wait for serving replicas to finish loading weights.
- Samples generated with the returned client are guaranteed to run on samplers
  using the weights saved when the client was created, or newer weights.
- Each returned client represents a sampling session with a fresh KV-cache
  namespace. Requests using that session, including serialized copies sent to
  worker processes, can reuse matching cached prefixes when available. Different
  inflight sessions do not share KV entries. A new save creates a new namespace
  even if no optimizer step occurred between saves.
- Publishing new weights does not clear an existing session's KV or change its
  namespace. Active generation may span updates and use KV computed under earlier
  weights. Outputs can depend on update timing even with the same prompt and seed.
  Responses do not identify which saved weights produced each token.

## `SaveWeightsExternalRequest` Objects

```python
class SaveWeightsExternalRequest(StrictBase)
```

#### `path`

A file/directory name for the external weights (required)

#### `ttl_seconds`

TTL in seconds for this checkpoint, 1 hour (3600) to 10 years (None = never expires)

#### `age_encryption_recipients`

age (filippo.io/age) recipients to encrypt every exported file to: native "age1..."
keys or SSH public keys. Files are stored with a ".age" suffix (None = unencrypted).

## `SaveWeightsForSamplerRequest` Objects

```python
class SaveWeightsForSamplerRequest(StrictBase)
```

#### `path`

A file/directory name for the weights

#### `kind`

See `SamplingClientKind` for the snapshot and inflight sampling contracts.

#### `ttl_seconds`

TTL in seconds for this checkpoint, 1 hour (3600) to 10 years (None = never expires).

Inflight saves omit this field.

#### `user_metadata`

Optional user-provided metadata to attach to the checkpoint

## `SaveWeightsRequest` Objects

```python
class SaveWeightsRequest(StrictBase)
```

#### `path`

A file/directory name for the weights

#### `ttl_seconds`

TTL in seconds for this checkpoint, 1 hour (3600) to 10 years (None = never expires)

#### `overwrite`

If True, overwrite any existing checkpoint with the same name

#### `user_metadata`

Optional user-provided metadata to attach to the checkpoint

## `SaveWeightsResponse` Objects

```python
class SaveWeightsResponse(BaseModel)
```

#### `path`

A tinker URI for model weights at a specific step

#### `get_console_url`

```python
def get_console_url() -> str
```

Return the Tinker Console URL for this checkpoint.

## `OwnSearchOwner` Objects

```python
class OwnSearchOwner(StrictBase)
```

The caller's own sessions or checkpoints.

## `AccessibleSearchOwner` Objects

```python
class AccessibleSearchOwner(StrictBase)
```

Every owner's sessions or checkpoints.

## `UserSearchOwner` Objects

```python
class UserSearchOwner(StrictBase)
```

The listed users' sessions or checkpoints.

#### `owner_urns`

Owner URNs to match (1 to 10)

## `SessionEndEvent` Objects

```python
class SessionEndEvent(BaseModel)
```

#### `duration`

ISO 8601 duration string

#### `event`

Telemetry event type

#### `severity`

Log severity level

## `SessionSearchCursor` Objects

```python
class SessionSearchCursor(BaseModel)
```

Where the next page starts; pass a response's `next_cursor` back unchanged.

## `SessionSearchRequest` Objects

```python
class SessionSearchRequest(StrictBase)
```

A session search. Results are restricted to projects the caller may list.

#### `session_id`

Exact session ID to match; None searches every session. With
`display_name`, a session matching either one matches

#### `project_ids`

Projects to search; None searches every listable project. Naming a
project the caller cannot list is an error

#### `user_metadata`

Exact user_metadata key/value matches; every pair must hold

#### `display_name`

Case-insensitive substring match on the display name; with several
words, each must appear somewhere in the name. With `session_id`, a session
matching either one matches

#### `owner`

Whose sessions to search; defaults to the caller's own

#### `statuses`

Match any listed status; None matches every status

#### `limit`

Page size, 1 to 200

#### `cursor`

`next_cursor` of the previous page; None for the first page

## `SessionSearchResponse` Objects

```python
class SessionSearchResponse(BaseModel)
```

One page of search results, newest first.

#### `next_cursor`

Pass as the next request's `cursor`; None once the results are exhausted

## `SessionStartEvent` Objects

```python
class SessionStartEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `severity`

Log severity level

## `SessionFailed` Objects

```python
class SessionFailed(BaseModel)
```

The session was failed by one of its requests, training runs or sampling sessions.

## `ClientSessionFinished` Objects

```python
class ClientSessionFinished(BaseModel)
```

The session's client reported it was done.

## `ActiveSessionStatusQuery` Objects

```python
class ActiveSessionStatusQuery(StrictBase)
```

Sessions still running.

## `CancelledSessionStatusQuery` Objects

```python
class CancelledSessionStatusQuery(StrictBase)
```

Sessions cancelled from the console.

## `FailedSessionStatusQuery` Objects

```python
class FailedSessionStatusQuery(StrictBase)
```

Sessions failed by a request, training run or sampling session.

## `ClientFinishedSessionStatusQuery` Objects

```python
class ClientFinishedSessionStatusQuery(StrictBase)
```

Sessions whose client reported it was done.

#### `reasons`

Match any listed client-reported finish reason; None matches every reason

## `SessionSummary` Objects

```python
class SessionSummary(BaseModel)
```

#### `status`

None when the session neither finished nor is known to be running

## `UntypedAPIFuture` Objects

```python
class UntypedAPIFuture(BaseModel)
```

#### `sample_sequence_ids`

Sampling promises only: one id per requested sample, aligned with the
final response's sequence order; each becomes the corresponding
``SampledSequence.sequence_id``.

## `TelemetryBatch` Objects

```python
class TelemetryBatch(BaseModel)
```

#### `platform`

Host platform name

#### `sdk_version`

SDK version string

#### `process_uuid`

UUID identifying the client process, shared by all its sessions

## `TelemetrySendRequest` Objects

```python
class TelemetrySendRequest(StrictBase)
```

#### `platform`

Host platform name

#### `sdk_version`

SDK version string

#### `process_uuid`

UUID identifying the client process, shared by all its sessions

## `TensorData` Objects

```python
@dataclass(frozen=True, eq=False, init=False)
class TensorData()
```

#### `shape`

Optional.

The shape of the tensor (see PyTorch tensor.shape). The shape of a
one-dimensional list of length N is `(N,)`. Can usually be inferred if not
provided, and is generally inferred as a 1D tensor.

#### `sparse_crow_indices`

Optional CSR compressed row pointers. When set, this tensor is sparse CSR:
- data contains only the listed values (flattened); every other entry takes
  the `pad_value` given when converting to or from a dense tensor
- sparse_crow_indices contains the row pointers (length = nrows + 1)
- sparse_col_indices contains the column indices (length = nnz)
- shape is required and specifies the dense shape

#### `sparse_col_indices`

Optional CSR column indices. Must be set together with sparse_crow_indices.

#### `data`

```python
def data() -> Union[List[int], List[float]]
```

Flattened tensor data as array of numbers.

#### `from_torch_sparse`

```python
def from_torch_sparse(cls,
                      tensor: torch.Tensor,
                      pad_value: int = 0) -> TensorData
```

Create a sparse CSR TensorData from a dense 2-D torch tensor.

Entries equal to `pad_value` are left out; the rest are stored as CSR
values. Automatically detects sparsity and encodes as CSR when it saves
space. Falls back to dense if the tensor is 1-D or mostly non-pad.

`pad_value` must be an integer: the tensor is shifted by it so torch's
zero-based CSR conversion can be reused, and an integer shift is exact
for `int64` tensors. For `float32` tensors with a non-zero pad it may
round values whose magnitude is far below `pad_value`.

#### `to_numpy`

```python
def to_numpy(pad_value: int = 0) -> npt.NDArray[Any]
```

Convert TensorData to numpy array.

A sparse CSR tensor is densified with `pad_value` in every unlisted entry.

#### `to_torch`

```python
def to_torch(pad_value: int = 0) -> torch.Tensor
```

Convert TensorData to torch tensor.

A sparse CSR tensor is densified with `pad_value` in every unlisted
entry. `pad_value` must be an integer so the shift around torch's
zero-based densification is exact for `int64` tensors.

## `TopkLogprobs` Objects

```python
@dataclass(frozen=True, slots=True)
class TopkLogprobs()
```

Top-k most likely tokens at each position, as dense numpy matrices.

Both matrices have shape ``(length, k)`` where ``k`` is the number of top
tokens requested. Empty positions are filled with sentinel values
(``token_id=0``, ``logprob=-99999.0``).

#### `token_ids`

int32 matrix of token IDs, shape ``(length, k)``.

#### `logprobs`

float32 matrix of log probabilities, shape ``(length, k)``.

#### `topk_to_lists`

```python
def topk_to_lists(topk: TopkLogprobs) -> list[list[tuple[int, float]] | None]
```

Convert TopkLogprobs matrices to Python list format.

Each position becomes a list of up to k ``(token_id, logprob)`` tuples,
or ``None`` for positions filled entirely with the sentinel.

## `TrainingRun` Objects

```python
class TrainingRun(BaseModel)
```

#### `training_run_id`

The unique identifier for the training run

#### `base_model`

The base model name this model is derived from

#### `model_owner`

The owner/creator of this model

#### `is_lora`

Whether this model uses LoRA (Low-Rank Adaptation)

#### `corrupted`

Whether the model is in a corrupted state

#### `lora_rank`

The LoRA rank if this is a LoRA model, null otherwise

#### `last_request_time`

The timestamp of the last request made to this model

#### `last_checkpoint`

The most recent training checkpoint, if available

#### `last_sampler_checkpoint`

The most recent sampler checkpoint, if available

#### `user_metadata`

Optional metadata about this training run, set by the end-user

## `TrainingRunsResponse` Objects

```python
class TrainingRunsResponse(BaseModel)
```

#### `training_runs`

List of training runs

#### `cursor`

Pagination cursor information

## `TryAgainResponse` Objects

```python
class TryAgainResponse(BaseModel)
```

#### `request_id`

Request ID that is still pending

## `UnhandledExceptionEvent` Objects

```python
class UnhandledExceptionEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `severity`

Log severity level

#### `traceback`

Optional Python traceback string

## `WeightsInfoResponse` Objects

```python
class WeightsInfoResponse(BaseModel)
```

Minimal information for loading public checkpoints.
