# Tinker CLI — Full Command Reference

This reference covers every CLI command, flag, output
format, and edge case. Read this when the SKILL.md cheat
sheet is not enough.

## Table of Contents

1. [Global Options](#global-options)
2. [tinker version](#tinker-version)
3. [tinker run list](#tinker-run-list)
4. [tinker run info](#tinker-run-info)
5. [tinker checkpoint list](#tinker-checkpoint-list)
6. [tinker checkpoint info](#tinker-checkpoint-info)
7. [tinker checkpoint download](#tinker-checkpoint-download)
8. [tinker checkpoint publish / unpublish](#publish--unpublish)
9. [tinker checkpoint set-ttl](#tinker-checkpoint-set-ttl)
10. [tinker checkpoint delete](#tinker-checkpoint-delete)
11. [tinker checkpoint push-hf](#tinker-checkpoint-push-hf)
12. [Tinker Path Anatomy](#tinker-path-anatomy)
13. [Exit Codes](#exit-codes)
14. [Output System](#output-system)

---

## Global Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f, --format` | `table\|json` | `table` | Output format |
| `-h, --help` | flag | — | Show help |

Global `--format` is placed **before** the subcommand:
```bash
tinker --format json run list
```

---

## tinker version

Show SDK version. No arguments.

```bash
tinker version
# Output: tinker 0.8.0
```

---

## tinker run list

List training runs for the authenticated user.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--limit` | int | 20 | Max runs (0 = all) |
| `-c, --columns` | str | `id,model,lora,updated,status` | Comma-separated columns |

**Available columns:**
`id`, `model`, `owner`, `lora`, `updated`, `status`,
`checkpoint`, `checkpoint_time`

**Table output:**
```
Run ID         Base Model     LoRA        Updated       Status
──────────────────────────────────────────────────────────────
run-abc123     llama2-7b      Rank 32     2 hours ago   OK
run-def456     mistral-7b     Rank 64     1 day ago     OK
```

**JSON output:**
```json
{
  "runs": [
    {
      "training_run_id": "run-abc123",
      "base_model": "llama2-7b",
      "model_owner": "user123",
      "is_lora": true,
      "lora_rank": 32,
      "corrupted": false,
      "last_request_time": "2024-03-27T15:30:00Z",
      "last_checkpoint": {
        "checkpoint_id": "weights/00040",
        "checkpoint_type": "training",
        "time": "2024-03-27T14:00:00Z",
        "tinker_path": "tinker://run-abc123/weights/00040",
        "size_bytes": 1073741824,
        "public": false,
        "expires_at": null
      },
      "user_metadata": {"task": "instruction-tuning"}
    }
  ]
}
```

**Pagination:** Fetches in batches of 100. Title shows
count with hint (e.g., "20 runs (5 more not shown, use
--limit to see more)").

---

## tinker run info

Show details for a single training run.

| Argument | Required | Description |
|----------|----------|-------------|
| `RUN_ID` | yes | Training run ID |

```bash
tinker run info run-abc123
```

**Table output** shows key-value pairs:
```
Training Run: run-abc123

Property                    Value
──────────────────────────────────────────
Run ID                      run-abc123
Base Model                  llama2-7b
Owner                       user123
LoRA                        Yes (Rank 32)
Last Update                 2 hours ago
Status                      OK
Last Training Checkpoint    weights/00040
  - Time                    2 hours ago
  - Path                    tinker://run-abc123/weights/00040
```

---

## tinker checkpoint list

List checkpoints, optionally filtered by run.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | str | None | Filter to one run |
| `--limit` | int | 20 | Max results (0 = all) |

**Behavior:**
- With `--run-id`: lists all checkpoints for that run
  (no pagination)
- Without: lists across all runs with pagination

```bash
tinker checkpoint list --run-id run-abc123
```

**Table output:**
```
4 checkpoints

Checkpoint ID        Type       Size      Public  Created       Path
─────────────────────────────────────────────────────────────────────
weights/00040        training   1.5 GB    No      2 hours ago   tinker://run-abc123/weights/00040
sampler_weights/05   sampler    512 MB    No      3 days ago    tinker://run-abc123/sampler_weights/05
```

---

## tinker checkpoint info

Show details for a single checkpoint.

| Argument | Required | Description |
|----------|----------|-------------|
| `CHECKPOINT_PATH` | yes | tinker:// path |

```bash
tinker checkpoint info tinker://run-abc123/weights/00040
```

Shows: checkpoint ID, type, tinker path, size, public
status, creation time, expiration, run ID, LoRA info.

---

## tinker checkpoint download

Download and extract a checkpoint archive.

| Argument/Option | Type | Default | Description |
|-----------------|------|---------|-------------|
| `CHECKPOINT_PATH` | str | — | tinker:// path (required) |
| `-o, --output` | path | cwd | Parent directory |
| `--force` | flag | false | Overwrite existing dir |

**Directory naming:**
`tinker://run-abc123/weights/00040`
→ `run-abc123_weights_00040/`

```bash
tinker checkpoint download \
  tinker://run-abc123/weights/00040 \
  -o ./models/ --force
```

**Extracted contents (typical LoRA):**
```
run-abc123_weights_00040/
├── adapter_config.json
├── adapter_model.safetensors
└── checkpoint_complete
```

**Safety:** Rejects symlinks, hardlinks, and path
traversal in tar archives.

---

## Publish / Unpublish

Toggle public accessibility of a checkpoint.

```bash
tinker checkpoint publish <TINKER_PATH>
tinker checkpoint unpublish <TINKER_PATH>
```

Silent on success. Only the run owner can change this.

---

## tinker checkpoint set-ttl

Set or remove checkpoint expiration.

| Option | Type | Description |
|--------|------|-------------|
| `--ttl` | int | TTL in seconds from now |
| `--remove` | flag | Clear expiration |

Must specify exactly one of `--ttl` or `--remove`.

```bash
# Expire in 7 days
tinker checkpoint set-ttl \
  tinker://run-abc123/weights/00040 --ttl 604800

# Remove expiration
tinker checkpoint set-ttl \
  tinker://run-abc123/weights/00040 --remove
```

---

## tinker checkpoint delete

Delete checkpoints permanently. Two modes:

**Mode 1: By explicit paths**
```bash
tinker checkpoint delete \
  tinker://run-id/weights/0001 \
  tinker://run-id/weights/0002 [-y]
```

**Mode 2: By run ID with filters**

| Option | Type | Description |
|--------|------|-------------|
| `--run-id` | str | Target run |
| `--type` | str | `weights` or `sampler_weights` |
| `--before` | str | ISO 8601 date (UTC) |
| `--after` | str | ISO 8601 date (UTC) |
| `-y, --yes` | flag | Skip confirmation |

```bash
tinker checkpoint delete --run-id run-abc123 \
  --type weights --before 2025-01-01 -y
```

**Constraints:**
- Cannot mix explicit paths with `--run-id`
- Filters (`--type`, `--before`, `--after`) require
  `--run-id`
- Date format: `2024-01-01`, `2024-01-01T12:00:00Z`
- Without `-y`, shows confirmation prompt

**Concurrency:** Deletes up to 32 checkpoints in
parallel using ThreadPoolExecutor.

---

## tinker checkpoint push-hf

Upload checkpoint to HuggingFace Hub as a PEFT adapter.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CHECKPOINT_PATH` | str | — | tinker:// path (required) |
| `-r, --repo` | str | auto | HF repo ID |
| `--public` | flag | false | Public repo |
| `--revision` | str | auto | Branch/revision |
| `--commit-message` | str | auto | Commit message |
| `--create-pr` | flag | false | Create PR instead |
| `--allow-pattern` | str | — | File include pattern (repeatable) |
| `--ignore-pattern` | str | — | File exclude pattern (repeatable) |
| `--no-model-card` | flag | false | Skip README.md |

**Prerequisites:**
```bash
pip install huggingface_hub
hf auth login
```

**Auto-derived values:**
- Repo ID: `tinker-<model-short-name>-<run-id>`
- Revision: sanitized checkpoint ID
  (e.g., `sampler_weights-0005`)

```bash
tinker checkpoint push-hf \
  tinker://run-abc123/sampler_weights/0005 \
  -r myorg/my-lora --public \
  --commit-message "Checkpoint after epoch 5"
```

**Generates model card** with PEFT metadata, base model
info, usage snippet, and tinker source path.

---

## Tinker Path Anatomy

```
tinker://<RUN_ID>/<CHECKPOINT_TYPE>/<CHECKPOINT_ID>
```

| Component | Values | Example |
|-----------|--------|---------|
| `RUN_ID` | run identifier | `run-abc123` |
| `CHECKPOINT_TYPE` | `weights` (training), `sampler_weights` (sampler) | `weights` |
| `CHECKPOINT_ID` | step number or name | `00040`, `final` |

Full example: `tinker://run-abc123/weights/00040`

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (TinkerCliError) |
| 2 | Click validation error (bad args) |
| 130 | User interrupt (Ctrl+C) |

---

## Output System

All commands support `--format table` (default) and
`--format json`.

- **Table**: Uses Rich library, colored headers, cyan
  first column, emoji disabled to prevent ID mangling
- **JSON**: 2-space indent, trailing newline, suitable
  for piping to `jq`
- Progress bars only shown in table mode
