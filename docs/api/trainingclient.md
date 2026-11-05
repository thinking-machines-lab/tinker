TrainingClient for Tinker API.

## `TrainingClient` Objects

```python
@with_sdk_method_telemetry
class TrainingClient(TelemetryProvider)
```

Client for training ML models with forward/backward passes and optimization.

The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
You typically get one by calling `service_client.create_lora_training_client()`.
Key methods:
- forward_backward() - compute gradients for training
- optim_step() - update model parameters with Adam optimizer
- save_weights_and_get_sampling_client() - export trained model for inference

Args:
- `holder`: Internal client managing HTTP connections and async operations
- `model_id`: Unique identifier for the model to train. Required for training operations.

Example:
```python
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B").result()
fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
fwdbwd_result = fwdbwd_future.result()  # Wait for gradients
optim_result = optim_future.result()    # Wait for parameter update
sampling_client = training_client.save_weights_and_get_sampling_client().result()
```

#### `get_console_url`

```python
def get_console_url() -> str
```

Return the Tinker Console URL for this training run.

#### `forward`

```python
def forward(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Mapping[str, float | str] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Compute forward pass without gradients.

Args:
- `data`: List of training data samples
- `loss_fn`: Loss function type (e.g., "cross_entropy")
- `loss_fn_config`: Optional configuration for the loss function

Returns:
- `APIFuture` containing the forward pass outputs and loss

Example:
```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
    loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
)]
future = training_client.forward(data, "cross_entropy")
result = await future
print(f"Loss: {result.loss}")
```

#### `forward_async`

```python
async def forward_async(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Mapping[str, float | str] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Deprecated: use `training_client.forward(...)`, which returns an `APIFuture` at once;
submit the requests that follow, then await the future or call `.result()` on it.
This method will be removed in a subsequent release.

#### `forward_backward`

```python
def forward_backward(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Mapping[str, float | str] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Compute forward pass and backward pass to calculate gradients.

Args:
- `data`: List of training data samples
- `loss_fn`: Loss function type (e.g., "cross_entropy")
- `loss_fn_config`: Optional configuration for the loss function

Returns:
- `APIFuture` containing the forward/backward outputs, loss, and gradients

Example:
```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
    loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
)]

# Compute gradients
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

# Update parameters
optim_future = training_client.optim_step(
    types.AdamParams(learning_rate=1e-4)
)

fwdbwd_result = await fwdbwd_future
print(f"Loss: {fwdbwd_result.loss}")
```

#### `forward_backward_async`

```python
async def forward_backward_async(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Mapping[str, float | str] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Deprecated: use `training_client.forward_backward(...)`, which returns an `APIFuture`
at once. Submit the optimizer step before waiting:
```python
fwd_bwd_future = training_client.forward_backward(data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
await fwd_bwd_future  # or fwd_bwd_future.result()
await optim_future  # or optim_future.result()
```
This method will be removed in a subsequent release.

#### `forward_backward_custom`

```python
def forward_backward_custom(
    data: List[types.Datum],
    loss_fn: CustomLossFnV1 | LossHandle,
    *,
    loss_type_input: Literal["logprobs"] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Compute forward/backward with a custom loss function.

Allows you to define custom loss functions that operate on log probabilities.
A callable, or a `LossHandle` that runs client-side, receives every
`loss_fn_inputs` field, but only `target_tokens` and `weights` are sent to the
server. A `LossHandle` the server runs (one with a `loss_id`) is sent with its
data unchanged and runs in a single server-side forward/backward.

Returns a future without waiting for server submission or processing. Requests on
this client execute in submission order, so you can submit an optimizer step before
waiting for this future. If a client-side custom loss operation fails or is cancelled,
this client rejects queued and subsequent training requests.

Args:
- `data`: List of training data samples
- `loss_fn`: Custom loss function that takes (data, logprobs) and returns (loss, metrics),
  or a `LossHandle` from `ServiceClient.create_custom_loss`
- `loss_type_input`: Deprecated; the loss always receives logprobs. Will be removed in
  a future release.

Returns:
- `APIFuture` containing the forward/backward outputs with custom loss

Example:
```python
def custom_loss(data, logprobs_list):
    # Custom loss computation
    loss = torch.mean(torch.stack([torch.mean(lp) for lp in logprobs_list]))
    metrics = {"custom_metric": loss.item()}
    return loss, metrics

future = training_client.forward_backward_custom(data, custom_loss)
result = future.result()
print(f"Custom loss: {result.loss}")
print(f"Metrics: {result.metrics}")
```

For example, to provide a target distribution for a custom loss:
```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints([1]),
    loss_fn_inputs={
        "target_tokens": [[2, 3]],
        "target_distribution": types.TensorData(
            data=[0.25, 0.75], dtype="float32", shape=[1, 2],
        ),
    },
)]

def custom_loss(data, logprobs_list):
    loss = sum(
        -(datum.loss_fn_inputs["target_distribution"].to_torch() * logprobs).sum()
        for datum, logprobs in zip(data, logprobs_list, strict=True)
    )
    return loss, {}

future = training_client.forward_backward_custom(data, custom_loss)
```

#### `forward_backward_custom_async`

```python
async def forward_backward_custom_async(
    data: List[types.Datum],
    loss_fn: CustomLossFnV1 | LossHandle,
    *,
    loss_type_input: Literal["logprobs"] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Deprecated: use `training_client.forward_backward_custom(...)`, which returns an
`APIFuture` at once. Submit the optimizer step before waiting:
```python
fwd_bwd_future = training_client.forward_backward_custom(data, custom_loss)
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
await fwd_bwd_future  # or fwd_bwd_future.result()
await optim_future  # or optim_future.result()
```
This method will be removed in a subsequent release.

#### `optim_step`

```python
def optim_step(
    optim_params: types.OptimParams | None = None,
    *,
    adam_params: types.AdamParams | None = None
) -> APIFuture[types.OptimStepResponse]
```

Update model parameters using the optimizer selected at model creation.

The Adam optimizer used by tinker is identical
to [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
Note that unlike PyTorch, Tinker's default weight decay value is 0.0 (no weight decay).


Args:
- `optim_params`: AdamParams or DimuonParams matching the optimizer selected at model creation.
- `adam_params`: Legacy keyword alias for AdamParams; cannot be combined with optim_params.

Returns:
- `APIFuture` containing optimizer step response

Example:
```python
# First compute gradients
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

# Then update parameters
optim_future = training_client.optim_step(
    types.AdamParams(
        learning_rate=1e-4,
        weight_decay=0.01
    )
)

# Wait for both to complete
fwdbwd_result = await fwdbwd_future
optim_result = await optim_future
```

#### `optim_step_async`

```python
async def optim_step_async(
    optim_params: types.OptimParams | None = None,
    *,
    adam_params: types.AdamParams | None = None
) -> APIFuture[types.OptimStepResponse]
```

Deprecated: use `training_client.optim_step(...)`, which returns an `APIFuture` at
once. Submit it right after `forward_backward`, then wait for both:
```python
fwd_bwd_future = training_client.forward_backward(data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
await fwd_bwd_future  # or fwd_bwd_future.result()
await optim_future  # or optim_future.result()
```
This method will be removed in a subsequent release.

#### `save_state`

```python
def save_state(
    name: str,
    ttl_seconds: int | None = None,
    overwrite: bool = False,
    user_metadata: dict[str, str] | None = None
) -> APIFuture[types.SaveWeightsResponse]
```

Save model weights to persistent storage.

Args:
- `name`: Name for the saved checkpoint
- `ttl_seconds`: Optional TTL in seconds for the checkpoint, between 1 hour (3600) and
  10 years (None = never expires)
- `overwrite`: If True, overwrite any existing checkpoint with the same name. This
  replaces the entire existing `user_metadata` mapping; if `user_metadata` is not
  provided, the previous user metadata is deleted.
- `user_metadata`: Optional user-provided metadata to attach to the checkpoint

Returns:
- `APIFuture` containing the save response with checkpoint path

Example:
```python
# Save after training
save_future = training_client.save_state("checkpoint-001")
result = await save_future
print(f"Saved to: {result.path}")
```

#### `save_state_async`

```python
async def save_state_async(
    name: str,
    ttl_seconds: int | None = None,
    overwrite: bool = False,
    user_metadata: dict[str, str] | None = None
) -> APIFuture[types.SaveWeightsResponse]
```

Deprecated: use `training_client.save_state(...)`, which returns an `APIFuture` that can
be awaited. This method will be removed in a subsequent release.

#### `load_state`

```python
def load_state(
    path: str,
    weights_access_token: str | None = None
) -> APIFuture[types.LoadWeightsResponse]
```

Load model weights from a saved checkpoint.

This loads only the model weights, not optimizer state (e.g., Adam momentum).
To also restore optimizer state, use load_state_with_optimizer.

Deprecated: will not be supported in future SDK versions. Use
`ServiceClient.create_training_client_from_state()` instead.

Args:
- `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
- `weights_access_token`: Optional access token for loading checkpoints under a different account.

Returns:
- `APIFuture` containing the load response

Example:
```python
# Load checkpoint to continue training (weights only, optimizer resets)
load_future = training_client.load_state("tinker://run-id/weights/checkpoint-001")
await load_future
# Continue training from loaded state
```

#### `load_state_async`

```python
async def load_state_async(
    path: str,
    weights_access_token: str | None = None
) -> APIFuture[types.LoadWeightsResponse]
```

Async version of load_state.

#### `load_state_with_optimizer`

```python
def load_state_with_optimizer(
    path: str,
    weights_access_token: str | None = None
) -> APIFuture[types.LoadWeightsResponse]
```

Load model weights and optimizer state from a checkpoint.

Deprecated: will not be supported in future SDK versions. Use
`ServiceClient.create_training_client_from_state_with_optimizer()` instead.

Args:
- `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
- `weights_access_token`: Optional access token for loading checkpoints under a different account.

Returns:
- `APIFuture` containing the load response

Example:
```python
# Resume training with optimizer state
load_future = training_client.load_state_with_optimizer(
    "tinker://run-id/weights/checkpoint-001"
)
await load_future
# Continue training with restored optimizer momentum
```

#### `load_state_with_optimizer_async`

```python
async def load_state_with_optimizer_async(
    path: str,
    weights_access_token: str | None = None
) -> APIFuture[types.LoadWeightsResponse]
```

Async version of load_state_with_optimizer.

#### `save_weights_for_sampler`

```python
def save_weights_for_sampler(
    name: str,
    ttl_seconds: int | None = None,
    user_metadata: dict[str, str] | None = None
) -> APIFuture[types.SaveWeightsForSamplerResponse]
```

Save model weights for use with a SamplingClient.

Args:
- `name`: Name for the saved sampler weights
- `ttl_seconds`: Optional TTL in seconds for the checkpoint, between 1 hour (3600) and
  10 years (None = never expires)
- `user_metadata`: Optional user-provided metadata to attach to the checkpoint

Returns:
- `APIFuture` containing the save response with sampler path

Example:
```python
# Save weights for inference
save_future = training_client.save_weights_for_sampler("sampler-001")
result = await save_future
print(f"Sampler weights saved to: {result.path}")

# Use the path to create a sampling client
sampling_client = service_client.create_sampling_client(
    model_path=result.path
)
```

#### `save_weights_for_sampler_async`

```python
async def save_weights_for_sampler_async(
    name: str,
    ttl_seconds: int | None = None,
    user_metadata: dict[str, str] | None = None
) -> APIFuture[types.SaveWeightsForSamplerResponse]
```

Deprecated: use `training_client.save_weights_for_sampler(...)`, which returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent
release.

#### `save_weights_external`

```python
def save_weights_external(
    name: str,
    ttl_seconds: int | None = None,
    age_encryption_recipients: list[str] | None = None
) -> APIFuture[types.SaveWeightsExternalResponse]
```

Save model weights in an external (e.g. HuggingFace) format.

The saved checkpoint is stored in the external-weights bucket and can be
managed (listed, deleted) like any other checkpoint.

Args:
- `name`: Name for the saved external weights
- `ttl_seconds`: Optional TTL in seconds for the checkpoint, between 1 hour (3600) and
  10 years (None = never expires)
- `age_encryption_recipients`: Optional age (filippo.io/age) recipients (at most 10),
  native `age1...` keys or SSH public keys. When set, every exported file is
  encrypted to them and stored with a `.age` suffix; decrypt with
  `age -d -i <identity>`.

Returns:
- `APIFuture` containing the save response with the external weights path

#### `save_weights_external_async`

```python
async def save_weights_external_async(
    name: str,
    ttl_seconds: int | None = None,
    age_encryption_recipients: list[str] | None = None
) -> APIFuture[types.SaveWeightsExternalResponse]
```

Deprecated: use `training_client.save_weights_external(...)`, which returns an
`APIFuture` that can be awaited. This method will be removed in a subsequent
release.

#### `get_info`

```python
def get_info() -> types.GetInfoResponse
```

Information about the model: its base model, LoRA configuration and optimizer.

Fetched once when the client was created, so this never waits on the server.

Example:
```python
info = training_client.get_info()
print(f"Model ID: {info.model_id}")
print(f"Base model: {info.model_data.model_name}")
print(f"LoRA rank: {info.lora_rank}")
```

#### `get_info_async`

```python
async def get_info_async() -> types.GetInfoResponse
```

Deprecated: use `training_client.get_info()`, which no longer waits on the server.
This method will be removed in a subsequent release.

#### `get_tokenizer`

```python
def get_tokenizer() -> PreTrainedTokenizer
```

Get the tokenizer for the current model.

Returns:
- `PreTrainedTokenizer` compatible with the model

Example:
```python
tokenizer = training_client.get_tokenizer()
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

#### `create_sampling_client`

```python
def create_sampling_client(
        model_path: str,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False) -> BackcompatSamplingClientFuture
```

Create a SamplingClient from saved weights.

Args:
- `model_path`: Tinker path to saved weights
- `retry_config`: Optional configuration for retrying failed requests
- `record_stability_info`: When true, every sample from this client
    records stability information for its sequences (StableReinforce
    training); recorded data is referenced from training requests by
    `sequence_id`, is session-scoped, and expires after inactivity.

Returns:
- A future of the `SamplingClient` configured with the specified weights.
    Await it, or call `.result()`.

Example:
```python
sampling_client = training_client.create_sampling_client(
    "tinker://run-id/weights/checkpoint-001"
).result()
# Use sampling_client for inference
```

#### `create_sampling_client_async`

```python
async def create_sampling_client_async(
        model_path: str,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False) -> SamplingClient
```

Deprecated: use `await training_client.create_sampling_client(...)`, which returns a
future of the client that can be awaited. This method will be removed in a
subsequent release.

#### `save_weights_and_get_sampling_client`

```python
def save_weights_and_get_sampling_client(
    name: str | None = None,
    retry_config: RetryConfig | None = None,
    record_stability_info: bool = False,
    kind: types.SamplingClientKind = "snapshot"
) -> BackcompatSamplingClientFuture
```

Save current weights and create a SamplingClient for inference.

Args:
- `name`: Deprecated, has no effect. Will be removed in a future release.
- `retry_config`: Optional configuration for retrying failed requests
- `record_stability_info`: When true, every sample from this client
    records stability information for its sequences (StableReinforce
    training); recorded data is referenced from training requests by
    `sequence_id`, is session-scoped, and expires after inactivity.
- `kind`: `snapshot` (default) captures fixed weights. `inflight` publishes
  mutable weights shared by this training model's inflight sampling clients.

See `types.SamplingClientKind` for the snapshot and inflight sampling contracts.

Returns:
- A future of the `SamplingClient` using the saved snapshot or shared inflight
    weights. Await it, or call `.result()`.

Example:
```python
# After training, create a sampling client directly
sampling_client = training_client.save_weights_and_get_sampling_client().result()

# Now use it for inference
prompt = types.ModelInput.from_ints(tokenizer.encode("Hello"))
params = types.SamplingParams(max_tokens=20)
result = sampling_client.sample(prompt, 1, params).result()
```

Inflight example:
```python
# Publish W1 and use sampler_a for the rollouts starting now.
sampler_a = training_client.save_weights_and_get_sampling_client(kind="inflight")

# ... train further ...

# Publish W2 and give sampler_b to newly starting rollouts.
sampler_b = training_client.save_weights_and_get_sampling_client(kind="inflight")

# sampler_b requires W2 or newer and has a fresh KV namespace.
# sampler_a keeps its namespace and can adopt W2 as it propagates.
# Until then, sampler_a can still use a replica serving W1.
```

#### `save_weights_and_get_sampling_client_async`

```python
async def save_weights_and_get_sampling_client_async(
        name: str | None = None,
        retry_config: RetryConfig | None = None,
        record_stability_info: bool = False,
        kind: types.SamplingClientKind = "snapshot") -> SamplingClient
```

Deprecated: use `await training_client.save_weights_and_get_sampling_client(...)`, which
returns a future of the client that can be awaited. This method will be removed
in a subsequent release.

#### `fetch_training_info`

```python
async def fetch_training_info(
        holder: InternalClientHolder,
        model_id: types.ModelID) -> types.GetInfoResponse
```

The model's `GetInfoResponse`, fetched once by the paths that create a `TrainingClient`.
