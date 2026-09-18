SamplingClient for Tinker API.

## `SamplingClient` Objects

```python
class SamplingClient(TelemetryProvider, QueueStateObserver)
```

Client for text generation and inference from trained or base models.

The SamplingClient lets you generate text tokens from either a base model or from weights
you've saved using a TrainingClient. You typically get one by calling
`service_client.create_sampling_client()` or `training_client.save_weights_and_get_sampling_client()`.

Key methods:
- sample() - generate text completions with customizable parameters
- compute_logprobs() - get log probabilities for prompt tokens

Create method parameters:
- `model_path`: Path to saved model weights (starts with 'tinker://')
- `base_model`: Name of base model to use for inference (e.g., 'Qwen/Qwen3-8B')
- `retry_config`: Configuration for retrying failed requests

Example:
```python
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
params = types.SamplingParams(max_tokens=20, temperature=0.7)
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()
```

Multi-processing support:
This class is picklable, so it can be passed to a separate process/worker to sample. It is also
safe to pass the same instance of SamplingClient to multiple processes/workers.

If you are using Tinker SDK with more than one process you should always create SamplingClient from
the main process and then pass it to the other processes/workers.
ServiceClient and TrainingClient should always be managed from the main process.

#### `sample`

```python
def sample(
    prompt: types.ModelInput,
    num_samples: int,
    sampling_params: types.SamplingParams,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
    topk_sample_logprobs: int = 0,
    target_prompt_logprobs: types.TensorData | None = None
) -> ConcurrentFuture[types.SampleResponse]
```

Generate text completions from the model.

Args:
- `prompt`: The input tokens as ModelInput
- `num_samples`: Number of independent samples to generate
- `sampling_params`: Parameters controlling generation (temperature, max_tokens, etc.)
- `include_prompt_logprobs`: Whether to include log probabilities for prompt tokens
- `topk_prompt_logprobs`: Number of top token log probabilities to return per prompt position
- `topk_sample_logprobs`: Number of top token log probabilities to return per sampled position
- `target_prompt_logprobs`: Token ids whose log probabilities to return at each prompt
    position, as an int64 `TensorData` of shape `[len(prompt) - 1, K]`:
    `target_prompt_logprobs[i][j]` is scored at prompt position `i + 1` (position 0
    has no preceding context). Use `-1` for cells you don't need; no logprob is
    computed for them. Dense (`TensorData.from_torch(ids)`) or sparse CSR
    (`TensorData.from_torch_sparse(ids, pad_value=-1)`), which sends and returns
    only the cells you name. The server requires exactly `len(prompt) - 1` rows and
    at least one id, and bounds the cost, `len(prompt) * distinct ids`, the way it
    bounds a top-k width. Rows before the first one that names an id are not scored.

Returns:
- A `Future` containing the `SampleResponse` with generated text and other logprob information.

Example:
```python
prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
params = types.SamplingParams(max_tokens=20, temperature=0.7)
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()
for sequence in result.sequences:
    print(tokenizer.decode(sequence.tokens))
```

Example: log probabilities of chosen token ids at chosen prompt positions.
`max_tokens=1` makes the request a single prefill of the prompt (one token is still
generated, and can be ignored); `target_prompt_logprobs` names the ids to score. Here,
we score one candidate token at the last position in the prompt and send in a sparse tensor:
```python
tokens = tokenizer.encode("Hello world")
position = len(tokens) - 1
ids = torch.full((len(tokens) - 1, 1), -1, dtype=torch.int64)
ids[position - 1, 0] = candidate_token_id  # row i - 1 scores prompt position i
target = types.TensorData.from_torch_sparse(ids, pad_value=-1)
future = sampling_client.sample(
    prompt=types.ModelInput.from_ints(tokens),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    target_prompt_logprobs=target,
)
result = future.result()
prompt_logprobs = result.prompt_logprobs  # [len(tokens)], None at position 0
actual_token_logprob = prompt_logprobs[position]
target_logprobs = result.target_prompt_logprobs.to_torch()  # [len(tokens) - 1, 1]
candidate_token_logprob = target_logprobs[position - 1, 0]
```

#### `sample_async`

```python
async def sample_async(
    prompt: types.ModelInput,
    num_samples: int,
    sampling_params: types.SamplingParams,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
    topk_sample_logprobs: int = 0,
    target_prompt_logprobs: types.TensorData | None = None
) -> types.SampleResponse
```

Async version of sample.

#### `compute_logprobs`

```python
def compute_logprobs(
        prompt: types.ModelInput) -> ConcurrentFuture[list[float | None]]
```

Compute log probabilities for prompt tokens.

Args:
- `prompt`: The input tokens as ModelInput

Returns:
- A `Future` containing a list of log probabilities for each token in the prompt.
    None values indicate tokens where log probabilities couldn't be computed.

Example:
```python
prompt = types.ModelInput.from_ints(tokenizer.encode("Hello world"))
future = sampling_client.compute_logprobs(prompt)
logprobs = future.result()
for i, logprob in enumerate(logprobs):
    if logprob is not None:
        print(f"Token {i}: logprob = {logprob:.4f}")
```

#### `compute_logprobs_async`

```python
async def compute_logprobs_async(
        prompt: types.ModelInput) -> list[float | None]
```

Async version of compute_logprobs.

#### `get_tokenizer`

```python
def get_tokenizer() -> PreTrainedTokenizer
```

Get the tokenizer for the current model.

Returns:
- `PreTrainedTokenizer` compatible with the model

#### `get_base_model`

```python
def get_base_model() -> str
```

Get the base model name for the current sampling session.

#### `get_base_model_async`

```python
async def get_base_model_async() -> str
```

Async version of get_base_model.

#### `__reduce__`

```python
def __reduce__() -> tuple[Any, tuple[_SamplingClientPickleState]]
```

Enable pickling of SamplingClient for multi-process use.

Serializes into a ``_SamplingClientPickleState`` dataclass.
