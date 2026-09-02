---
name: tinker
description: >
  Use this skill whenever the user mentions Tinker, tinker CLI,
  training runs, checkpoints, model fine-tuning with Tinker,
  tinker-cookbook, tinker recipes, or any Thinking Machines AI
  SDK operations. Also trigger when users ask about
  listing/inspecting/downloading/deleting training checkpoints,
  pushing checkpoints to HuggingFace, managing checkpoint TTL,
  configuring post-training pipelines (SFT, RL, math RL,
  code RL, distillation, preference learning, RLHF, tool use
  training, multi-agent RL, prompt distillation, rubric
  grading, VLM classification, Harbor RL), or working with
  tinker:// paths. Use this skill even if the user just
  mentions "tinker" in passing — it covers the full Tinker
  ecosystem including CLI, Python SDK, and cookbook recipes.
---

# Tinker SDK Skill

Tinker is an ML platform SDK by Thinking Machines AI for
managing training runs, model checkpoints, and fine-tuning
workflows. This skill covers the CLI, Python SDK, and the
tinker-cookbook training recipes.

**Source repos:**
- SDK: https://github.com/thinking-machines-lab/tinker
- Cookbook: https://github.com/thinking-machines-lab/tinker-cookbook
- This skill: https://github.com/zjrwtx/max_skills

**IMPORTANT — Always use the latest version:**
Before running any Tinker command or cookbook recipe,
ensure the latest version is installed:
```bash
uv pip install --upgrade tinker
# For cookbook, pull latest and reinstall:
cd <cookbook-dir> && git pull && uv pip install -e .
```
When you need more detailed information about API
internals, recipe implementations, or SDK source code,
always check the latest code from these repos — do NOT
rely on cached or outdated knowledge. Clone or browse
the repos directly to get up-to-date APIs and options.

## Quick Start

### Authentication

```bash
# Option 1: Environment variable (preferred)
export TINKER_API_KEY="your-api-key"

# Option 2: Config file (~/.tinker/config.json)
mkdir -p ~/.tinker
echo '{"api_key": "your-api-key"}' > ~/.tinker/config.json
```

### Verify Installation

```bash
tinker version
tinker run list --limit 3
```

### Tinker Path Format

All checkpoint operations use **tinker paths**:

```
tinker://<RUN_ID>/<TYPE>/<STEP>
```

- `TYPE`: `weights` (training) or `sampler_weights` (sampler)
- Example: `tinker://run-abc123/weights/00040`

---

## CLI Commands

### Global Options

- `--format [table|json]` or `-f` — output format
  (default: table)
- `-h` / `--help` — help on any command

### Run Commands

```bash
# List training runs (default: 20, use --limit=0 for all)
tinker run list [--limit N] [-c COLUMNS]

# Available columns:
#   id, model, owner, lora, updated, status,
#   checkpoint, checkpoint_time
# Default columns: id, model, lora, updated, status

# Show detailed info for a specific run
tinker run info <RUN_ID>
```

### Checkpoint Commands

```bash
# List checkpoints (all runs, or filter by --run-id)
tinker checkpoint list [--run-id ID] [--limit N]

# Show checkpoint details
tinker checkpoint info <TINKER_PATH>

# Download and extract checkpoint locally
tinker checkpoint download <TINKER_PATH> \
  [-o OUTPUT_DIR] [--force]

# Toggle public access
tinker checkpoint publish <TINKER_PATH>
tinker checkpoint unpublish <TINKER_PATH>

# Set or remove expiration (TTL in seconds)
tinker checkpoint set-ttl <TINKER_PATH> --ttl 604800
tinker checkpoint set-ttl <TINKER_PATH> --remove

# Delete checkpoints (by path or by filters)
tinker checkpoint delete <PATH1> [PATH2 ...] [-y]
tinker checkpoint delete --run-id <ID> \
  [--type weights|sampler_weights] \
  [--before DATE] [--after DATE] [-y]

# Push checkpoint to HuggingFace Hub
tinker checkpoint push-hf <TINKER_PATH> \
  [-r REPO_ID] [--public] [--revision REV] \
  [--commit-message MSG] [--create-pr] \
  [--allow-pattern PAT] [--ignore-pattern PAT] \
  [--no-model-card]
```

> For full flag details and output format examples,
> read `references/cli-reference.md`.

---

## Common Workflows

### 1. Find and Download a Checkpoint

```bash
# Step 1: Find your training run
tinker run list

# Step 2: Inspect the run
tinker run info <RUN_ID>

# Step 3: List available checkpoints
tinker checkpoint list --run-id <RUN_ID>

# Step 4: Download
tinker checkpoint download \
  tinker://<RUN_ID>/weights/<STEP> \
  -o ./models/ --force
```

### 2. Push a Checkpoint to HuggingFace

```bash
# Prerequisite: authenticate with HF
# pip install huggingface_hub && hf auth login

# Push as public PEFT adapter
tinker checkpoint push-hf \
  tinker://<RUN_ID>/sampler_weights/<STEP> \
  -r myorg/my-lora --public

# Or create a PR instead of direct push
tinker checkpoint push-hf \
  tinker://<RUN_ID>/sampler_weights/<STEP> \
  -r myorg/my-lora --create-pr
```

### 3. Clean Up Old Checkpoints

```bash
# Delete checkpoints older than a date
tinker checkpoint delete --run-id <RUN_ID> \
  --type weights --before 2025-01-01 -y

# Delete specific checkpoints
tinker checkpoint delete \
  tinker://<RUN_ID>/weights/0001 \
  tinker://<RUN_ID>/weights/0002 -y
```

### 4. Scripting with JSON Output

```bash
# Export all runs as JSON
tinker --format json run list --limit=0 > runs.json

# Parse with jq
jq '.runs[].training_run_id' runs.json

# Batch list checkpoints per run
for rid in $(jq -r '.runs[].training_run_id' runs.json)
do
  tinker --format json checkpoint list --run-id "$rid"
done
```

---

## Cookbook Recipes

The tinker-cookbook provides ready-to-use training recipes.
Repo: https://github.com/thinking-machines-lab/tinker-cookbook

### Recipe Architecture

Every recipe follows the same pattern:

```python
import chz
from tinker_cookbook.rl import train  # or supervised

# 1. Build a typed config via chz.Blueprint
def build_config_blueprint() -> chz.Blueprint[train.Config]:
    return chz.Blueprint(train.Config).apply({
        "model_name": "meta-llama/Llama-3.1-8B",
        "learning_rate": 2e-4,
        ...
    })

# 2. Run the training loop
def main(config):
    asyncio.run(train.main(config))

# 3. CLI entry point with chz overrides
if __name__ == "__main__":
    bp = build_config_blueprint()
    bp.make_from_argv(sys.argv[1:])
    main(bp.make())
```

Override any config field from the command line:
```bash
python -m tinker_cookbook.recipes.sl_basic \
  --model_name "Qwen/Qwen3-8B" \
  --learning_rate 1e-4 \
  --log_path /tmp/my-run
```

### Running SFT (Supervised Fine-Tuning)

```bash
# Minimal SFT on NoRobots dataset
python -m tinker_cookbook.recipes.sl_basic

# With custom dataset (JSONL of conversations)
# Edit sl_basic.py to use FromConversationFileBuilder:
#   file_path="/path/to/conversations.jsonl"
# Format: same as example_data/conversations.jsonl
```

### Running RL Training

```bash
# Math RL on GSM8K
python -m tinker_cookbook.recipes.rl_basic

# Override hyperparameters
python -m tinker_cookbook.recipes.rl_basic \
  --learning_rate 4e-5 \
  --max_tokens 256
```

### Available Recipes

| Recipe | Type | Use Case |
|--------|------|----------|
| `sl_basic` | SFT | Minimal SFT template |
| `rl_basic` | RL | Minimal RL template |
| `chat_sl/` | SFT | Conversations (Tulu3) |
| `math_rl/` | RL | Math reasoning (GSM8K) |
| `code_rl/` | RL | Code (sandboxed exec) |
| `preference/` | RLHF | SFT → reward → RL |
| `search_tool/` | RL | Retrieval tool use |
| `distillation/` | SFT/RL | Teacher→student |
| `prompt_distillation/` | SFT | Internalize prompts |
| `multiplayer_rl/` | RL | Self-play / multi-agent |
| `rubric/` | RL | LLM grader rubrics |
| `verifiers_rl/` | RL | Community envs |
| `vlm_classifier/` | SFT | Vision-language |
| `harbor_rl/` | RL | Terminal/SWE tasks |

### Key Utilities

```python
from tinker_cookbook import model_info

# Get the right renderer for a model
renderer = model_info.get_recommended_renderer_name(
    "meta-llama/Llama-3.1-8B"
)

# Checkpoint save/resume
from tinker_cookbook import checkpoint_utils
resume = checkpoint_utils.get_last_checkpoint(log_path)
```

### Supported Models

Llama 3.x, Qwen 3/3.5, DeepSeek V3, Nemotron 3,
Kimi K2/K2.5, GPT-OSS, and 30+ more. Each model has
a recommended renderer in `model_info.py`.

> For recipe deep-dives, renderer details, dataset
> builder patterns, and RL environment setup, read
> `references/cookbook-recipes.md`.

---

## Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| Auth failure | Check `TINKER_API_KEY` or `~/.tinker/config.json` |
| Checkpoint not found | Verify path format `tinker://RUN/TYPE/STEP`; list available with `tinker checkpoint list --run-id ID` |
| Download fails | Use `--force` to overwrite; check disk space |
| Cookbook import error | `uv pip install -e .` in cookbook dir; needs Python 3.10+ |
| chz override syntax | `--field value` (flat) or `--outer.inner value` (nested) |
| Rate limit | Wait and retry; reduce `--limit` for batch ops |
| HF push fails | Run `hf auth login`; install `huggingface_hub` |

> For the full error catalog, read
> `references/troubleshooting.md`.

---

## Detailed References

When the SKILL.md cheat sheet is not enough:

- **`references/cli-reference.md`** — Every flag, output
  format example (table + JSON), exit codes, date format
  rules, bulk delete filter logic
- **`references/cookbook-recipes.md`** — Per-recipe config
  fields, renderer selection, dataset builder interface,
  RL environment pattern, hyperparameter guidance
- **`references/troubleshooting.md`** — Extended error
  catalog with 15+ error-to-fix mappings, network/proxy
  issues, W&B integration, checkpoint corruption
