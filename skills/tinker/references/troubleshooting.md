# Tinker Troubleshooting Guide

Extended error catalog and common fixes for the Tinker
CLI and cookbook.

## Table of Contents

1. [Authentication Errors](#authentication-errors)
2. [Checkpoint Errors](#checkpoint-errors)
3. [Network / Connection Errors](#network--connection-errors)
4. [Cookbook / Import Errors](#cookbook--import-errors)
5. [HuggingFace Push Errors](#huggingface-push-errors)
6. [W&B Integration](#wb-integration)
7. [CLI Error Reference](#cli-error-reference)

---

## Authentication Errors

### "Authentication failed"

**Message:**
```
Error: Authentication failed
Please check your API key or authentication credentials.
```

**Fixes:**
1. Check env var: `echo $TINKER_API_KEY`
2. Check config: `cat ~/.tinker/config.json`
3. API keys are workspace-scoped — verify you're using
   the correct workspace key
4. Regenerate key from the Tinker dashboard

### "Permission denied"

**Message:**
```
Error: Permission denied
You do not have access to this resource.
```

**Causes:**
- Trying to modify another user's checkpoint (publish,
  unpublish, delete, set-ttl)
- API key lacks required permissions
- Wrong workspace

---

## Checkpoint Errors

### "Resource not found"

**Message:**
```
Error: Resource not found
```

**Fixes:**
1. Verify path format: `tinker://RUN_ID/TYPE/STEP`
   - TYPE must be `weights` or `sampler_weights`
   - STEP must match exactly (e.g., `00040` not `40`)
2. List available checkpoints:
   ```bash
   tinker checkpoint list --run-id <RUN_ID>
   ```
3. Check if checkpoint was deleted or TTL-expired
4. Verify the run still exists:
   ```bash
   tinker run info <RUN_ID>
   ```

### "Invalid checkpoint path"

**Message:**
```
Error: Invalid checkpoint path: <path>
Checkpoint path must be in the format:
tinker://run-id/weights/0001
```

**Fix:** Ensure path starts with `tinker://` and follows
the format `tinker://RUN_ID/TYPE/STEP`.

### "Target directory already exists"

**Message:**
```
Error: Target directory already exists: <path>
Use --force to overwrite or choose a different output
directory.
```

**Fix:** Add `--force` flag to overwrite, or use `-o` to
specify a different output directory.

### "Failed to extract archive"

**Message:**
```
Error: Failed to extract archive: <details>
The downloaded file may be corrupted.
```

**Fixes:**
1. Re-download the checkpoint
2. Check disk space: `df -h`
3. Verify the checkpoint is not corrupted by checking
   its info: `tinker checkpoint info <PATH>`

### Checkpoint Corruption Detection

Signs of a corrupted checkpoint:
- Missing `checkpoint_complete` marker file
- Missing `adapter_config.json`
- Missing weight files (`.safetensors` or `.bin`)
- Run shows `corrupted: true` in `tinker run info`

**Fix:** Download a different checkpoint step, or contact
Tinker support if the entire run is corrupted.

---

## Network / Connection Errors

### "Connection failed"

**Message:**
```
Error: Connection failed
Please check your network connection and try again.
```

**Fixes:**
1. Check internet: `ping api.thinkingmachines.ai`
2. Check proxy: `echo $HTTP_PROXY $HTTPS_PROXY`
3. Firewall may block Tinker API endpoints
4. Try again — may be transient

### "Request timeout"

**Message:**
```
Error: Request timeout
The request took too long. Please try again.
```

**Fixes:**
1. Retry the command
2. For large downloads, check network bandwidth
3. For bulk operations, reduce batch size

### "Rate limit exceeded"

**Message:**
```
Error: Rate limit exceeded
Please wait and try again.
```

**Fixes:**
1. Wait 30-60 seconds and retry
2. Reduce `--limit` for batch listing operations
3. Avoid running multiple CLI sessions in parallel

---

## Cookbook / Import Errors

### ModuleNotFoundError: tinker_cookbook

**Fix:**
```bash
cd /path/to/tinker-cookbook
uv pip install -e .
```

### ModuleNotFoundError: tinker

**Fix:**
```bash
uv pip install tinker
# Or with CLI extras:
uv pip install "tinker[cli]"
```

### Python Version

The cookbook requires Python 3.10+. Check:
```bash
python --version
```

### Missing Optional Dependencies

Some recipes need extras:

| Recipe | Extra | Install |
|--------|-------|---------|
| math_rl | math | `uv pip install -e ".[math-rl]"` |
| code_rl | modal | `uv pip install -e ".[modal]"` |
| search_tool | vector | `uv pip install -e ".[vector-search]"` |
| Any + W&B | wandb | `uv pip install -e ".[wandb]"` |
| verifiers_rl | verifiers | `uv pip install -e ".[verifiers]"` |
| eval | inspect | `uv pip install -e ".[inspect]"` |

### chz Configuration Errors

**"Unknown field"** when using CLI overrides:
- Check available fields: `python -m <recipe> --help`
- Use exact field names (case-sensitive)
- Nested fields: `--dataset_builder.batch_size 64`

**"Cannot convert"** type errors:
- Ensure values match expected types
- Strings need quotes: `--model_name "Qwen/Qwen3-8B"`
- Booleans: `--flag True` or `--flag False`

### Log Directory Conflicts

When re-running a recipe with the same `--log_path`:
- Choose "resume" to continue from last checkpoint
- Choose "overwrite" to start fresh
- Use `cli_utils.check_log_dir()` behavior parameter:
  - `"ask"` — interactive prompt
  - `"resume"` — auto-resume
  - `"overwrite"` — auto-overwrite

---

## HuggingFace Push Errors

### "huggingface_hub is not installed"

**Fix:**
```bash
pip install huggingface_hub
```

### "Not logged in to Hugging Face"

**Fix:**
```bash
hf auth login
# Paste your HF token when prompted
# Verify: hf whoami
```

### "Repo contains different Tinker checkpoint"

The target HF repo already has a checkpoint from a
different Tinker run. This prevents accidental overwrites.

**Fixes:**
1. Use a different `--repo` name
2. Use `--revision` to push to a different branch
3. Delete the existing repo on HF and retry

### "Invalid adapter format"

The checkpoint doesn't contain required PEFT files
(`adapter_config.json` + weight files).

**Fixes:**
1. Verify it's a LoRA checkpoint (not a full model)
2. Check the run: `tinker run info <RUN_ID>` — look
   for `is_lora: true`
3. Try a different checkpoint step

---

## W&B Integration

### "wandb not installed"

```bash
pip install wandb
# Or: uv pip install -e ".[wandb]"
```

### "wandb not logged in"

```bash
wandb login
# Paste your API key
```

### Metrics Not Appearing

1. Check `--wandb_project` is set correctly
2. Verify W&B API key: `wandb verify`
3. Check network access to `api.wandb.ai`
4. Look for errors in recipe stdout/stderr

---

## CLI Error Reference

Full mapping of API errors to CLI messages:

| API Error | CLI Message | Hint |
|-----------|------------|------|
| `NotFoundError` | Resource not found | Check path/ID |
| `AuthenticationError` | Authentication failed | Check API key |
| `PermissionDeniedError` | Permission denied | Wrong owner |
| `BadRequestError` | Invalid request | Check args |
| `UnprocessableEntityError` | Invalid data | Check format |
| `RateLimitError` | Rate limit exceeded | Wait & retry |
| `InternalServerError` | Internal server error | Retry later |
| `APITimeoutError` | Request timeout | Retry |
| `APIConnectionError` | Connection failed | Check network |
| `APIStatusError` | API error (status N) | Check status |

**Exit codes:**
- `0` — Success
- `1` — General error
- `2` — Bad arguments (Click validation)
- `130` — Ctrl+C interrupt
