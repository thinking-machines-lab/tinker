ServiceClient for Tinker API.

## `ServiceClient` Objects

```python
@with_sdk_method_telemetry
class ServiceClient(TelemetryProvider)
```

The ServiceClient is the main entry point for the Tinker API. It provides methods to:
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

#### `holder`

```python
def holder() -> InternalClientHolder
```

The sessionful holder. Deprecated: kept for backwards compatibility
with callers that reach into ServiceClient internals.

#### `get_console_url`

```python
def get_console_url() -> str
```

Return the Tinker Console URL for this session.

#### `get_server_capabilities`

```python
def get_server_capabilities() -> types.GetServerCapabilitiesResponse
```

Deprecated: use `tinker.RestClient().get_server_capabilities().result()`, which
needs no session. This method will be removed in a subsequent release.

#### `get_server_capabilities_async`

```python
async def get_server_capabilities_async(
) -> types.GetServerCapabilitiesResponse
```

Deprecated: use `await tinker.RestClient().get_server_capabilities()`. This method will
be removed in a subsequent release.

#### `create_lora_training_client`

```python
def create_lora_training_client(
    base_model: str,
    rank: int = 32,
    seed: int | None = None,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
    user_metadata: dict[str, str] | None = None,
    optimizer: types.OptimizerConfig | None = None
) -> BackcompatTrainingClientFuture
```

Create a TrainingClient for LoRA fine-tuning.

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

#### `create_lora_training_client_async`

```python
async def create_lora_training_client_async(
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
        optimizer: types.OptimizerConfig | None = None) -> TrainingClient
```

Deprecated: use `await service_client.create_lora_training_client(...)`, which returns a
future of the client that can be awaited. This method will be removed in a
subsequent release.

#### `copy_weights`

```python
def copy_weights(path: str,
                 *,
                 ttl_seconds: int | None = None,
                 weights_access_token: str | None = None) -> APIFuture[str]
```

Copy weights into this client's project, using a new session.

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

#### `create_training_client_from_state`

```python
def create_training_client_from_state(
    path: str,
    base_model: str | None = None,
    user_metadata: dict[str, str] | None = None,
    weights_access_token: str | None = None,
    optimizer: types.OptimizerConfig | None = None
) -> BackcompatTrainingClientFuture
```

Create a TrainingClient from saved model weights.

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

#### `create_training_client_from_state_async`

```python
async def create_training_client_from_state_async(
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
        optimizer: types.OptimizerConfig | None = None) -> TrainingClient
```

Deprecated: use `await service_client.create_training_client_from_state(...)`, which
returns a future of the client that can be awaited. This method will be removed
in a subsequent release.

#### `create_training_client_from_state_with_optimizer`

```python
def create_training_client_from_state_with_optimizer(
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None
) -> BackcompatTrainingClientFuture
```

Create a TrainingClient from saved model weights and optimizer state.

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

#### `create_training_client_from_state_with_optimizer_async`

```python
async def create_training_client_from_state_with_optimizer_async(
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None) -> TrainingClient
```

Deprecated: use `await
service_client.create_training_client_from_state_with_optimizer(...)`, which
returns a future of the client that can be awaited. This method will be removed
in a subsequent release.

#### `create_sampling_client`

```python
def create_sampling_client(
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False) -> BackcompatSamplingClientFuture
```

Create a SamplingClient for text generation.

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

#### `create_custom_loss`

```python
def create_custom_loss(
    fn: PerDatumLossFn,
    input_ranks: Mapping[str,
                         types.CustomLossInputRank]) -> APIFuture[LossHandle]
```

Create a custom loss function for ``TrainingClient.forward_backward_custom``.

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

#### `create_sampling_client_async`

```python
async def create_sampling_client_async(
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False) -> SamplingClient
```

Deprecated: use `await service_client.create_sampling_client(...)`, which returns a
future of the client that can be awaited. This method will be removed in a
subsequent release.

#### `create_rest_client`

```python
def create_rest_client() -> RestClient
```

Deprecated: construct `RestClient()` directly instead. This method will be
removed in a subsequent release.

Returns:
- `RestClient` for accessing REST API endpoints

#### `close`

```python
def close(status: Literal["success", "errored", "interrupted"],
          detail: str | None = None) -> AwaitableConcurrentFuture[None]
```

Finish the session and release local client resources.

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
