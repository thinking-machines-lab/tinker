# Tinker Cookbook — Recipe Deep-Dives

This reference covers the cookbook's architecture, every
recipe, renderer selection, dataset builders, RL
environments, and hyperparameter guidance.

Cookbook repo:
https://github.com/thinking-machines-lab/tinker-cookbook

SDK repo:
https://github.com/thinking-machines-lab/tinker

For detailed source code, browse or clone these repos.

## Table of Contents

1. [Blueprint / chz Pattern](#blueprint--chz-pattern)
2. [SFT Recipes](#sft-recipes)
3. [RL Recipes](#rl-recipes)
4. [Renderer Selection](#renderer-selection)
5. [Dataset Builder Interface](#dataset-builder-interface)
6. [RL Environment Pattern](#rl-environment-pattern)
7. [Pipelined Async Training](#pipelined-async-training)
8. [Checkpoint & Resume](#checkpoint--resume)
9. [Logging](#logging)
10. [Hyperparameter Guidance](#hyperparameter-guidance)

---

## Blueprint / chz Pattern

Every recipe uses `chz.Blueprint` for typed, CLI-
overridable configuration:

```python
import chz
from tinker_cookbook.rl import train

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Build a Blueprint with defaults that can be
    overridden from the command line."""
    return chz.Blueprint(train.Config).apply({
        "model_name": "meta-llama/Llama-3.1-8B",
        "learning_rate": 2e-4,
        "log_path": "/tmp/my-run",
        ...
    })

if __name__ == "__main__":
    bp = build_config_blueprint()
    # Parse CLI overrides: --field value
    bp.make_from_argv(sys.argv[1:])
    config = bp.make()
    main(config)
```

**CLI override syntax:**
- Flat: `--model_name "Qwen/Qwen3-8B"`
- Nested: `--dataset_builder.batch_size 64`
- Check fields: `python -m <recipe> --help`

---

## SFT Recipes

### sl_basic (Minimal SFT)

Entry point: `tinker_cookbook/recipes/sl_basic.py`

Default config:
- Model: `meta-llama/Llama-3.1-8B`
- Dataset: NoRobots (HuggingFace Hub)
- LR: 2e-4, schedule: linear
- Epochs: 1, eval every 8 batches
- Batch size: 128, max length: 32768

```bash
python -m tinker_cookbook.recipes.sl_basic \
  --model_name "meta-llama/Llama-3.1-8B" \
  --learning_rate 2e-4
```

### chat_sl (Conversational SFT)

Entry point: `tinker_cookbook/recipes/chat_sl/train.py`

Multi-dataset support (NoRobots, Tulu3). Uses flexible
chat template rendering. Key config:
- `train_on_what`: `ALL_ASSISTANT_MESSAGES` (default)
- Supports custom conversation JSONL files

### prompt_distillation

Entry point:
`tinker_cookbook/recipes/prompt_distillation/train.py`

Internalizes long system prompts into model parameters.
Teacher-student framework with task-specific data
generation.

### vlm_classifier (Vision-Language SFT)

Entry point:
`tinker_cookbook/recipes/vlm_classifier/train.py`

Image + text classification using vision-language models.
Example: Caltech101 dataset.

### Custom Dataset for SFT

Use `FromConversationFileBuilder` with a JSONL file:

```python
from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
)

common = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer="meta-llama/Llama-3.1-8B",
    renderer_name="llama3",
    max_length=32768,
    batch_size=128,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
dataset = FromConversationFileBuilder(
    common_config=common,
    file_path="/path/to/conversations.jsonl",
)
```

JSONL format: see `example_data/conversations.jsonl`.
Each line is a JSON object with a `messages` array of
`{"role": "user"|"assistant", "content": "..."}`.

---

## RL Recipes

### rl_basic (Minimal RL)

Entry point: `tinker_cookbook/recipes/rl_basic.py`

Default config:
- Model: `meta-llama/Llama-3.1-8B`
- Dataset: GSM8K via `Gsm8kDatasetBuilder`
- LR: 4e-5, max tokens: 256
- Batch size: 128, group size: 16

```bash
python -m tinker_cookbook.recipes.rl_basic \
  --learning_rate 4e-5 --max_tokens 256
```

### math_rl (Math Reasoning)

Entry point: `tinker_cookbook/recipes/math_rl/train.py`

Trains on GSM8K / MATH / Arithmetic with custom grading
functions. Structured answer extraction with regex.

### code_rl (Code Reasoning)

Entry point: `tinker_cookbook/recipes/code_rl/train.py`

DeepCoder-like competitive programming. Sandboxed code
execution via SandboxFusion or Modal. Test-driven rewards.

### search_tool (Tool Use RL)

Entry point: `tinker_cookbook/recipes/search_tool/train.py`

Multi-hop QA with tool-calling framework. Vector DB
integration (ChromaDB) for retrieval. Multi-turn
interaction.

### preference (RLHF)

3-stage pipeline:
1. SFT on reference data
2. Train reward model on preference pairs
3. RL against reward model

Entry: `tinker_cookbook/recipes/preference/`

### distillation

On-policy & off-policy teacher→student distillation.
Multi-dataset support, teacher model loading.

Entry: `tinker_cookbook/recipes/distillation/`

### multiplayer_rl (Multi-Agent / Self-Play)

Environments: tic-tac-toe, 20 Questions, guess-the-number.
Self-play and multi-agent training.

Entry: `tinker_cookbook/recipes/multiplayer_rl/`

### rubric (LLM Grader)

LLM-based reward via structured rubrics. Regex extraction,
Prometheus dataset support.

Entry: `tinker_cookbook/recipes/rubric/train.py`

### verifiers_rl (Community Envs)

Prime Intellect Environments Hub integration. Generic
environment interface for community-contributed envs.

Entry: `tinker_cookbook/recipes/verifiers_rl/train.py`

### harbor_rl (Terminal/SWE Tasks)

Harbor task format standardization. Multi-turn bash tool
use with sandboxed execution.

Entry: `tinker_cookbook/recipes/harbor_rl/train.py`

---

## Renderer Selection

Each model requires a specific chat format renderer.
The registry in `model_info.py` maps model names to
recommended renderers.

```python
from tinker_cookbook import model_info

name = model_info.get_recommended_renderer_name(
    "meta-llama/Llama-3.1-8B"
)
# Returns: "llama3"
```

**Supported models and renderers:**

| Model Family | Example Model | Renderer |
|-------------|---------------|----------|
| Llama 3.x | `meta-llama/Llama-3.1-8B` | `llama3` |
| Qwen 3 | `Qwen/Qwen3-8B` | `qwen3` |
| Qwen 3.5 | `Qwen/Qwen3.5-*` | `qwen3_5` |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3.1` | `deepseek_v3` |
| Nemotron 3 | `nvidia/Nemotron-3-*` | `nemotron3` |
| Kimi K2 | `moonshotai/Kimi-K2` | `kimi_k2` |
| Kimi K2.5 | `moonshotai/Kimi-K2.5` | `kimi_k25` |
| GPT-OSS | `openai/gpt-oss-*` | `gpt_oss` |

Renderer files: `tinker_cookbook/renderers/`

Features vary by renderer:
- Tool calling support
- Thinking/reasoning mode (Qwen3, DeepSeek, Kimi)
- Stop sequences
- Image token counting (VLM renderers)

**Warning system:** If you use a non-recommended renderer,
`warn_if_renderer_not_recommended()` logs a warning.

---

## Dataset Builder Interface

### SFT Datasets

Inherit from the builder pattern:
- `SupervisedDatasetFromHFDataset` — HuggingFace Hub
- `FromConversationFileBuilder` — local JSONL file

Common config fields:
- `model_name_for_tokenizer`: tokenizer model name
- `renderer_name`: chat format renderer
- `max_length`: max sequence length
- `batch_size`: training batch size
- `train_on_what`: which messages to train on
  (`ALL_ASSISTANT_MESSAGES`, etc.)

### RL Datasets

Inherit from `RLDatasetBuilder`:

```python
class MyEnvBuilder(RLDatasetBuilder):
    def build_dataset(self):
        """Return batches of prompts + reward fn."""
        ...
```

Each builder provides:
- Prompt generation
- Reward computation
- Episode structure (single or multi-turn)

---

## RL Environment Pattern

RL recipes define environments that provide prompts and
compute rewards. Example from `math_rl/math_env.py`:

```python
class MathEnvironment:
    def get_prompt(self, problem):
        """Format the math problem as a prompt."""
        ...

    def compute_reward(self, response, answer):
        """Grade the response against ground truth."""
        ...
```

**Multi-turn environments** (search_tool, multiplayer_rl)
use `message_env.py`:
- Token-level trajectories from message-level episodes
- Multiple interaction turns with the environment
- Terminal rewards after conversation ends

---

## Pipelined Async Training

The training loop uses pipelined async requests for
throughput:

```python
# Pipeline: overlap compute with data loading
fwd_bwd_future = client.forward_backward(batch, loss)
optim_future = client.optim_step(adam_params)

# While GPU computes, prepare next batch
next_batch = dataset.next()

# Collect results
fwd_bwd_result = fwd_bwd_future.result()
optim_result = optim_future.result()
```

This overlaps:
- Data loading on CPU
- Forward/backward pass on GPU (via API)
- Optimizer step on GPU (via API)

---

## Checkpoint & Resume

```python
from tinker_cookbook import checkpoint_utils

# Save checkpoint
checkpoint_utils.save_checkpoint(
    client, log_path, batch_num, metrics
)

# Resume from last checkpoint
resume = checkpoint_utils.get_last_checkpoint(log_path)
if resume:
    client = service_client\
        .create_training_client_from_state_with_optimizer(
            resume.state_path
        )
    start_batch = resume.batch
```

All recipes support `--log_path` for recovery. Re-run
with the same path and choose "resume" when prompted.

Artifacts in log_path:
- `metrics.jsonl` — training metrics per batch
- `checkpoints.jsonl` — checkpoint metadata

---

## Logging

### Local Logging
```python
from tinker_cookbook.utils.ml_log import MLLog

logger = MLLog(log_path="/tmp/my-run")
logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
```

### Weights & Biases
```bash
python -m tinker_cookbook.recipes.sl_basic \
  --wandb_project "my-project"
```

Requires: `pip install wandb` and `wandb login`.

---

## Hyperparameter Guidance

### SFT Defaults

| Param | Small (1-3B) | Medium (7-8B) | Large (70B) |
|-------|-------------|---------------|-------------|
| LR | 2e-4 | 2e-4 | 1e-4 |
| Batch | 64-128 | 128 | 128-256 |
| Epochs | 1-3 | 1-2 | 1 |
| Max len | 4096-8192 | 8192-32768 | 32768 |

### RL Defaults

| Param | Small (1-3B) | Medium (7-8B) | Large (70B) |
|-------|-------------|---------------|-------------|
| LR | 4e-5 | 4e-5 | 2e-5 |
| Group | 8-16 | 16 | 16-32 |
| Max tok | 256-512 | 256-1024 | 512-2048 |

### Hyperparameter Utilities

```python
from tinker_cookbook import hyperparam_utils

# Estimate LoRA parameter count
count = hyperparam_utils.estimate_lora_params(
    model_name, lora_rank=32
)

# Get LR suggestion scaled by model size
lr = hyperparam_utils.suggest_learning_rate(model_name)
```

The `hyperparam_utils` module has a registry of known
hidden sizes per model for accurate parameter estimation.
