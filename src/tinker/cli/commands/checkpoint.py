"""Commands for managing checkpoints.

This module implements the 'tinker checkpoint' commands, including:
- list: List all checkpoints or checkpoints for a specific run
- info: Show details of a specific checkpoint
- download: Download and extract checkpoint archives
"""

from typing import TYPE_CHECKING, Any, Dict, List

import click

if TYPE_CHECKING:
    from tinker.lib.public_interfaces.rest_client import RestClient
    from tinker.types import Checkpoint, TrainingRun

from ..client import create_rest_client, handle_api_errors
from ..context import CLIContext
from ..exceptions import TinkerCliError
from ..output import OutputBase, format_bool, format_size, format_timestamp


class CheckpointListOutput(OutputBase):
    """Output for 'tinker checkpoint list' command."""

    def __init__(
        self,
        checkpoints: List["Checkpoint"],
        run_id: str | None = None,
        total_count: int | None = None,
        shown_count: int | None = None,
    ):
        """Initialize with list of checkpoints.

        Args:
            checkpoints: List of Checkpoint objects
            run_id: Optional training run ID if filtering by run
            total_count: Total number of checkpoints available
            shown_count: Number of checkpoints shown in this response
        """
        self.checkpoints = checkpoints
        self.run_id = run_id
        self.total_count = total_count
        self.shown_count = shown_count if shown_count is not None else len(checkpoints)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result = {}
        if self.run_id:
            result["run_id"] = self.run_id

        # Check if these are Pydantic models
        if self.checkpoints and hasattr(self.checkpoints[0], "model_dump"):
            result["checkpoints"] = [c.model_dump() for c in self.checkpoints]
        else:
            result["checkpoints"] = [dict(c) for c in self.checkpoints]

        return result

    def get_title(self) -> str | None:
        """Return title for table output."""
        count = len(self.checkpoints)

        if self.run_id:
            if count == 0:
                return f"No checkpoints found for run {self.run_id}"
            elif count == 1:
                title = f"1 checkpoint for run {self.run_id}"
            else:
                title = f"{count} checkpoints for run {self.run_id}"
        else:
            if count == 0:
                return "No checkpoints found"
            elif count == 1:
                title = "1 checkpoint"
            else:
                title = f"{count} checkpoints"

        # Add information about remaining checkpoints if available
        if self.total_count is not None and self.total_count > self.shown_count:
            remaining = self.total_count - self.shown_count
            if remaining == 1:
                title += " (1 more not shown, use --limit to see more)"
            else:
                title += f" ({remaining} more not shown, use --limit to see more)"

        return title

    def get_table_columns(self) -> List[str]:
        """Return column headers for table output."""
        return ["Checkpoint ID", "Type", "Size", "Public", "Created", "Expires", "Path"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = []
        for ckpt in self.checkpoints:
            rows.append(
                [
                    ckpt.checkpoint_id,
                    ckpt.checkpoint_type,
                    format_size(ckpt.size_bytes)
                    if hasattr(ckpt, "size_bytes") and ckpt.size_bytes is not None
                    else "N/A",
                    format_bool(ckpt.public),
                    format_timestamp(ckpt.time),
                    format_timestamp(ckpt.expires_at) if ckpt.expires_at else "Never",
                    ckpt.tinker_path,
                ]
            )
        return rows


class CheckpointInfoOutput(OutputBase):
    """Output for 'tinker checkpoint info' command."""

    def __init__(self, checkpoint: "Checkpoint", training_run: "TrainingRun"):
        """Initialize with a single checkpoint.

        Args:
            checkpoint: Checkpoint object
            training_run: TrainingRun object for additional info like LoRA rank
        """
        self.checkpoint = checkpoint
        self.training_run = training_run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        if hasattr(self.checkpoint, "model_dump"):
            result = self.checkpoint.model_dump()
        else:
            result = dict(self.checkpoint)

        # Add training run info
        result["is_lora"] = self.training_run.is_lora
        result["lora_rank"] = self.training_run.lora_rank

        return result

    def get_title(self) -> str | None:
        """Return title for table output."""
        return f"Checkpoint: {self.checkpoint.checkpoint_id}"

    def get_table_columns(self) -> List[str]:
        """Return column headers for table output."""
        return ["Property", "Value"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = [
            ["Checkpoint ID", self.checkpoint.checkpoint_id],
            ["Type", self.checkpoint.checkpoint_type],
            ["Tinker Path", self.checkpoint.tinker_path],
        ]

        # Size if available
        if hasattr(self.checkpoint, "size_bytes") and self.checkpoint.size_bytes is not None:
            rows.append(["Size", format_size(self.checkpoint.size_bytes)])

        # Public status
        rows.append(["Public", format_bool(self.checkpoint.public)])

        # Creation time
        rows.append(["Created", format_timestamp(self.checkpoint.time)])

        # Expiration time
        if self.checkpoint.expires_at:
            rows.append(["Expires", format_timestamp(self.checkpoint.expires_at)])
        else:
            rows.append(["Expires", "Never"])

        # Parse training run ID from path
        if self.checkpoint.tinker_path.startswith("tinker://"):
            parts = self.checkpoint.tinker_path.replace("tinker://", "").split("/")
            if parts:
                rows.append(["Training Run ID", parts[0]])

        # LoRA information from training run
        if self.training_run.is_lora:
            if self.training_run.lora_rank:
                rows.append(["LoRA", f"Yes (Rank {self.training_run.lora_rank})"])
            else:
                rows.append(["LoRA", "Yes"])
        else:
            rows.append(["LoRA", "No"])

        return rows


class CheckpointDownloadOutput(OutputBase):
    """Output for 'tinker checkpoint download' command."""

    def __init__(
        self,
        checkpoint_path: str,
        file_size_bytes: int | None = None,
        destination: str | None = None,
    ):
        """Initialize with download information.

        Args:
            checkpoint_path: The tinker path to the checkpoint
            file_size_bytes: Size of the archive in bytes
            destination: Where the checkpoint was extracted
        """
        self.checkpoint_path = checkpoint_path
        self.file_size_bytes = file_size_bytes
        self.destination = destination

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result = {
            "checkpoint_path": self.checkpoint_path,
            "destination": self.destination,
        }
        if self.file_size_bytes is not None:
            result["file_size_bytes"] = self.file_size_bytes
        return result

    def get_title(self) -> str | None:
        """Return title for table output."""
        return f"Checkpoint Download: {self.checkpoint_path}"

    def get_table_columns(self) -> List[str]:
        """Return column headers for table output."""
        return ["Property", "Value"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = [
            ["Checkpoint Path", self.checkpoint_path],
        ]

        if self.file_size_bytes is not None:
            rows.append(["Archive Size", format_size(self.file_size_bytes)])

        if self.destination:
            rows.append(["Extracted to", self.destination])

        return rows


class CheckpointHubUploadOutput(OutputBase):
    """Output for 'tinker checkpoint push-hf' command."""

    def __init__(
        self,
        checkpoint_path: str,
        repo_id: str,
        revision: str | None = None,
        public: bool | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.repo_id = repo_id
        self.revision = revision
        self.public = public

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "checkpoint_path": self.checkpoint_path,
            "repo_id": self.repo_id,
        }
        if self.revision is not None:
            result["revision"] = self.revision
        if self.public is not None:
            result["public"] = self.public
        return result

    def get_title(self) -> str | None:
        return f"Checkpoint Hub Upload: {self.checkpoint_path}"

    def get_table_columns(self) -> List[str]:
        return ["Property", "Value"]

    def get_table_rows(self) -> List[List[str]]:
        rows = [
            ["Checkpoint Path", self.checkpoint_path],
            ["Repo ID", self.repo_id],
        ]
        if self.revision is not None:
            rows.append(["Revision", self.revision])
        if self.public is not None:
            rows.append(["Public", format_bool(self.public)])
        return rows


def get_checkpoint_from_path(client: "RestClient", checkpoint_path: str) -> "Checkpoint":
    """Get checkpoint details from a tinker path.

    Args:
        checkpoint_path: A tinker path like "tinker://run-id/weights/0001"
        client: RestClient instance to use for API calls

    Returns:
        Checkpoint object

    Raises:
        TinkerCliError: If the checkpoint cannot be retrieved
    """
    # Lazy import
    from tinker import ParsedCheckpointTinkerPath

    try:
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(checkpoint_path)

        # Get the checkpoint info
        checkpoints_response = client.list_checkpoints(parsed.training_run_id).result()

        # Find the matching checkpoint
        for ckpt in checkpoints_response.checkpoints:
            if ckpt.tinker_path == checkpoint_path:
                return ckpt

        raise TinkerCliError(f"Checkpoint not found: {checkpoint_path}")

    except ValueError as e:
        raise TinkerCliError(
            f"Invalid checkpoint path: {e}",
            "Checkpoint paths should be in the format: tinker://run-id/weights/0001",
        )
    except TinkerCliError:
        # Re-raise our own errors
        raise
    except Exception as e:
        raise TinkerCliError(f"Failed to retrieve checkpoint: {e}")


def _export_checkpoint_to_hub(
    client: "RestClient",
    tinker_path: str,
    repo_id: str | None,
    *,
    private: bool,
    revision: str | None,
    commit_message: str | None,
    create_pr: bool,
    exist_ok: bool,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    add_model_card: bool,
) -> str:
    # Lazy imports to keep CLI startup fast
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise TinkerCliError(
            "huggingface_hub is required for this command.",
            "Install it with: pip install huggingface_hub, then run: hf auth login",
        ) from exc

    import json
    import os
    import re
    import tempfile
    from pathlib import Path

    from tinker import ParsedCheckpointTinkerPath

    # Validate tinker path
    parsed_tinker_path = ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)

    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:
        raise TinkerCliError("Not logged in", "Run: hf auth login") from exc

    def _sanitize_repo_name(value: str) -> str:
        safe_chars = []
        for ch in value:
            if ch.isalnum() or ch in {"-", "_", "."}:
                safe_chars.append(ch)
            else:
                safe_chars.append("-")
        name = "".join(safe_chars)
        while "--" in name:
            name = name.replace("--", "-")
        return name.strip("-_ .")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / "checkpoint.tar"
        extract_dir = temp_root / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)

        url_response = client.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
        _download_checkpoint_archive(
            url_response.url,
            archive_path=archive_path,
            show_progress=False,
            format="json",
        )
        _safe_extract_tar(archive_path, extract_dir, show_progress=False, format="json")

        adapter_config = extract_dir / "adapter_config.json"
        adapter_safetensors = extract_dir / "adapter_model.safetensors"
        adapter_bin = extract_dir / "adapter_model.bin"
        checkpoint_complete = extract_dir / "checkpoint_complete"
        if not adapter_config.exists() or not (
            adapter_safetensors.exists() or adapter_bin.exists()
        ):
            raise TinkerCliError(
                "Checkpoint archive does not contain a PEFT adapter.",
                "Expected adapter_config.json and adapter_model.safetensors (or adapter_model.bin).",
            )
        if not checkpoint_complete.exists():
            raise TinkerCliError(
                "Checkpoint archive is missing 'checkpoint_complete'.",
                "The adapter files may be incomplete.",
            )

        base_model = "unknown"
        lora_rank = None
        train_mlp = None
        train_attn = None
        train_unembed = None
        try:
            weights_info = client.get_weights_info_by_tinker_path(tinker_path).result()
            base_model = weights_info.base_model
            lora_rank = weights_info.lora_rank
            train_mlp = weights_info.train_mlp
            train_attn = weights_info.train_attn
            train_unembed = weights_info.train_unembed
        except Exception:
            pass

        try:
            config_data = json.loads(adapter_config.read_text(encoding="utf-8"))
            if not isinstance(config_data.get("base_model_name_or_path"), str):
                config_data["base_model_name_or_path"] = base_model
                adapter_config.write_text(
                    json.dumps(config_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
        except Exception:
            pass

        if repo_id is None:
            base_short = base_model.split("/")[-1] if base_model != "unknown" else "adapter"
            derived = f"tinker-{base_short}-{parsed_tinker_path.training_run_id}"
            repo_id = _sanitize_repo_name(derived)
            if revision is None:
                revision = _sanitize_repo_name(parsed_tinker_path.checkpoint_id.replace("/", "-"))

        readme_path = extract_dir / "README.md"
        if add_model_card and not readme_path.exists():
            tags: List[str] = ["tinker", "peft", "lora"]
            if base_model != "unknown":
                tags.append(f"base_model:adapter:{base_model}")
            model_card = [
                "---",
                f"base_model: {base_model}",
                "library_name: peft",
                "tags:",
            ]
            for tag in tags:
                model_card.append(f"- {tag}")
            model_card.append(f"tinker_path: {tinker_path}")
            model_card.extend(
                [
                    "---",
                    "",
                    "# Tinker LoRA Adapter",
                    "",
                    "This repository contains a LoRA adapter exported from Tinker.",
                    "",
                    "## Usage",
                    "",
                    "```python",
                    "from transformers import AutoModelForCausalLM",
                    "",
                    f'adapter_id = "{repo_id}"',
                    f'base_model = "{base_model}"',
                    "",
                    'model = AutoModelForCausalLM.from_pretrained(adapter_id, device_map="auto")',
                    "```",
                    "",
                    "## Source",
                    "",
                    "```",
                    f"{tinker_path}",
                    "```",
                    "",
                    "## Details",
                    "",
                    f"- Base model: {base_model}",
                ]
            )
            if lora_rank is not None:
                model_card.append(f"- LoRA rank: {lora_rank}")
            if train_mlp is not None or train_attn is not None or train_unembed is not None:
                model_card.append(
                    f"- Trained modules: attn={train_attn}, mlp={train_mlp}, unembed={train_unembed}"
                )
            model_card.append("")
            readme_path.write_text("\n".join(model_card), encoding="utf-8")

        api.create_repo(repo_id=repo_id, private=private, exist_ok=exist_ok)

        def _readme_tinker_path() -> str | None:
            try:
                readme_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    revision=revision,
                    token=None,
                )
            except Exception:
                return None
            try:
                text = Path(readme_file).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None
            match = re.search(r"tinker://[^\s`]+", text)
            return match.group(0) if match else None

        existing_tinker_path = _readme_tinker_path()
        if existing_tinker_path and existing_tinker_path != tinker_path:
            raise TinkerCliError(
                "Repo ID appears to contain a different Tinker checkpoint.",
                f"Found {existing_tinker_path}, expected {tinker_path}.",
            )

        if allow_patterns is None:
            ignore_patterns = list(ignore_patterns) if ignore_patterns else []
            if "checkpoint_complete" not in ignore_patterns:
                ignore_patterns.append("checkpoint_complete")

        api.upload_folder(
            folder_path=os.fspath(extract_dir),
            repo_id=repo_id,
            path_in_repo="",
            revision=revision,
            commit_message=commit_message,
            create_pr=create_pr,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        )

    return repo_id


def _safe_extract_tar(
    archive_path,
    extract_dir,
    *,
    show_progress: bool,
    format: str,
) -> None:
    import tarfile

    base = extract_dir.resolve()
    with tarfile.open(archive_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise TinkerCliError(
                    "Unsafe symlink or hardlink in tar archive",
                    "Archive may be corrupted or malicious.",
                )
            member_path = (extract_dir / member.name).resolve()
            if not str(member_path).startswith(str(base)):
                raise TinkerCliError(
                    "Unsafe path in tar archive",
                    "Archive may be corrupted or malicious.",
                )
        if show_progress and format != "json":
            with click.progressbar(
                members,
                label="Extracting archive ",
                show_percent=True,
                show_pos=True,
            ) as bar:
                for member in bar:
                    tar.extract(member, path=extract_dir)
        else:
            tar.extractall(path=extract_dir)


def _download_checkpoint_archive(
    url: str,
    *,
    archive_path,
    show_progress: bool,
    format: str,
) -> int:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            if show_progress and format != "json":
                with click.progressbar(
                    length=total_size,
                    label="Downloading archive",
                    show_percent=True,
                    show_pos=True,
                    show_eta=True,
                ) as bar:
                    with open(archive_path, "wb") as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))
            else:
                with open(archive_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
    except urllib.error.URLError as e:
        raise TinkerCliError(
            f"Failed to download checkpoint: {e}",
            "Please check your network connection and try again.",
        ) from e
    except IOError as e:
        raise TinkerCliError(
            f"Failed to save checkpoint: {e}",
            f"Please check that you have write permissions to {archive_path.parent}",
        ) from e

    return total_size


# Click command group for checkpoint commands
@click.group()
def cli():
    """Manage checkpoints."""
    pass


@cli.command()
@click.option("--run-id", help="Training run ID")
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of checkpoints to display when listing from all runs (default: 20, use --limit=0 to show all)",
)
@click.pass_obj
@handle_api_errors
def list(cli_context: CLIContext, run_id: str | None, limit: int) -> None:
    """List checkpoints.

    If --run-id is provided, list checkpoints for that specific training run.
    Otherwise, list checkpoints from all recent runs.
    """
    # Get format from context object
    format = cli_context.format

    # Create client
    client = create_rest_client()

    if run_id:
        # List checkpoints for specific run.
        # Note that there's no pagination for listing checkpoints on a single training run.
        response = client.list_checkpoints(run_id).result()

        # Create output object
        output = CheckpointListOutput(checkpoints=response.checkpoints, run_id=run_id)
    else:
        # List checkpoints from all user's training runs using list_user_checkpoints()
        all_checkpoints = []
        offset = 0
        # Fetch in batches of 1000 since the queries are so slow
        BATCH_SIZE = 1000

        # First fetch to get initial data and total count
        first_response = client.list_user_checkpoints(
            limit=min(BATCH_SIZE, limit) if limit > 0 else BATCH_SIZE, offset=0
        ).result()
        all_checkpoints.extend(first_response.checkpoints)
        total_count = (
            first_response.cursor.total_count
            if first_response.cursor
            else len(first_response.checkpoints)
        )

        # Determine target count: either user-specified limit or total available
        target_count = limit if limit > 0 else total_count
        target_count = min(target_count, total_count)  # Can't fetch more than exists

        # If we need to fetch more checkpoints, paginate with a progress bar
        if len(all_checkpoints) < target_count:
            with click.progressbar(
                length=target_count,
                label=f"Fetching {'all' if limit == 0 else str(target_count)} checkpoints",
                show_percent=True,
                show_pos=True,
                show_eta=True,
                hidden=cli_context.format != "table",
            ) as bar:
                bar.update(len(all_checkpoints))

                # Fetch remaining checkpoints in batches
                while len(all_checkpoints) < target_count:
                    offset = len(all_checkpoints)
                    remaining = target_count - len(all_checkpoints)
                    next_batch_size = min(BATCH_SIZE, remaining)

                    response = client.list_user_checkpoints(
                        limit=next_batch_size, offset=offset
                    ).result()
                    all_checkpoints.extend(response.checkpoints)
                    bar.update(len(response.checkpoints))

                    # Break if we got fewer than requested (reached the end)
                    if len(response.checkpoints) < next_batch_size:
                        break

        # Create output object with pagination information
        output = CheckpointListOutput(
            checkpoints=all_checkpoints, total_count=total_count, shown_count=len(all_checkpoints)
        )

    # Print in requested format
    output.print(format=format)


@cli.command()
@click.argument("checkpoint_path")
@click.pass_obj
@handle_api_errors
def info(cli_context: CLIContext, checkpoint_path: str) -> None:
    """Show details of a specific checkpoint.

    CHECKPOINT_PATH must be a tinker path (e.g., tinker://run-id/weights/0001).
    """
    # Get format from context object
    format = cli_context.format

    # Validate it's a tinker path
    if not checkpoint_path.startswith("tinker://"):
        raise TinkerCliError(
            f"Invalid checkpoint path: {checkpoint_path}",
            "Checkpoint path must be in the format: tinker://run-id/weights/0001",
        )

    # Lazy import
    from tinker import ParsedCheckpointTinkerPath

    client = create_rest_client()
    checkpoint = get_checkpoint_from_path(client, checkpoint_path)

    # Fetch training run for additional info (like LoRA rank)
    parsed = ParsedCheckpointTinkerPath.from_tinker_path(checkpoint_path)
    training_run = client.get_training_run(parsed.training_run_id).result()

    # Create output object
    output = CheckpointInfoOutput(checkpoint=checkpoint, training_run=training_run)

    # Print in requested format
    output.print(format=format)


@cli.command()
@click.argument("checkpoint_path")
@click.pass_obj
@handle_api_errors
def publish(cli_context: CLIContext, checkpoint_path: str) -> None:
    """Publish a checkpoint to make it publicly accessible.

    CHECKPOINT_PATH must be a tinker path (e.g., tinker://run-id/weights/0001).
    Only the owner of the training run can publish checkpoints.
    """
    # Validate it's a tinker path
    if not checkpoint_path.startswith("tinker://"):
        raise TinkerCliError(
            f"Invalid checkpoint path: {checkpoint_path}",
            "Checkpoint path must be in the format: tinker://run-id/weights/0001",
        )

    # Create client and publish
    client = create_rest_client()
    client.publish_checkpoint_from_tinker_path(checkpoint_path).result()


@cli.command()
@click.argument("checkpoint_path")
@click.pass_obj
@handle_api_errors
def unpublish(cli_context: CLIContext, checkpoint_path: str) -> None:
    """Unpublish a checkpoint to make it private again.

    CHECKPOINT_PATH must be a tinker path (e.g., tinker://run-id/weights/0001).
    Only the owner of the training run can unpublish checkpoints.
    """
    # Validate it's a tinker path
    if not checkpoint_path.startswith("tinker://"):
        raise TinkerCliError(
            f"Invalid checkpoint path: {checkpoint_path}",
            "Checkpoint path must be in the format: tinker://run-id/weights/0001",
        )

    # Create client and unpublish
    client = create_rest_client()
    client.unpublish_checkpoint_from_tinker_path(checkpoint_path).result()


@cli.command()
@click.argument("checkpoint_paths", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
@handle_api_errors
def delete(cli_context: CLIContext, checkpoint_paths: tuple[str, ...], yes: bool) -> None:
    """Delete one or more checkpoints permanently.

    CHECKPOINT_PATHS must be tinker paths (e.g., tinker://run-id/weights/0001).
    Only the owner of the training run can delete checkpoints.

    WARNING: This action is permanent and cannot be undone.
    """
    # Validate all paths upfront
    for path in checkpoint_paths:
        if not path.startswith("tinker://"):
            raise TinkerCliError(
                f"Invalid checkpoint path: {path}",
                "Checkpoint path must be in the format: tinker://run-id/weights/0001",
            )

    # If not using --yes, show checkpoint list and prompt for confirmation
    if not yes:
        count = len(checkpoint_paths)
        click.echo(f"Will delete {count} checkpoint(s):")
        for path in checkpoint_paths:
            click.echo(f"  - {path}")
        click.echo()

        # Confirmation prompt
        click.echo("WARNING: This action is permanent and cannot be undone.")
        if not click.confirm(f"Are you sure you want to delete {count} checkpoint(s)?"):
            click.echo("Deletion cancelled.")
            return

    # Create client and delete with progress bar
    client = create_rest_client()

    with click.progressbar(
        checkpoint_paths,
        label="Deleting checkpoints",
        show_percent=True,
        show_pos=True,
        hidden=cli_context.format != "table",
    ) as bar:
        for path in bar:
            client.delete_checkpoint_from_tinker_path(path).result()


@cli.command()
@click.argument("checkpoint_path")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Parent directory for extracted checkpoint (default: current directory)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing directory if it exists",
)
@click.pass_obj
@handle_api_errors
def download(
    cli_context: CLIContext,
    checkpoint_path: str,
    output: str | None,
    force: bool,
) -> None:
    """Download and extract a checkpoint archive.

    CHECKPOINT_PATH must be a tinker path (e.g., tinker://run-id/weights/0001).

    Downloads and extracts the checkpoint into a dedicated directory named after
    the checkpoint ID. The tar archive is automatically deleted after successful
    extraction. If the target directory already exists, the command will fail
    unless --force is specified.

    Examples:

        # Creates ./run-123_weights_final/ with checkpoint files
        tinker checkpoint download tinker://run-123/weights/final

        # Creates ./models/run-123_weights_final/ with checkpoint files
        tinker checkpoint download tinker://run-123/weights/final --output ./models/

        # Overwrites existing ./run-123_weights_final/ directory
        tinker checkpoint download tinker://run-123/weights/final --force
    """
    # Lazy imports to maintain fast CLI startup
    import shutil
    import tempfile
    from pathlib import Path

    # Validate it's a tinker path
    if not checkpoint_path.startswith("tinker://"):
        raise TinkerCliError(
            f"Invalid checkpoint path: {checkpoint_path}",
            "Checkpoint path must be in the format: tinker://run-id/weights/0001",
        )

    # Get format from context object
    format = cli_context.format

    # Determine output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path.cwd()

    # Generate checkpoint ID from checkpoint path
    checkpoint_id = checkpoint_path.replace("tinker://", "").replace("/", "_")

    # Target directory for extracted checkpoint
    target_path = output_dir / checkpoint_id

    # Check if target directory already exists
    if target_path.exists():
        if force:
            shutil.rmtree(target_path)
        else:
            raise TinkerCliError(
                f"Target directory already exists: {target_path}",
                "Use --force to overwrite or choose a different output directory.",
            )

    # Use a temporary directory for the archive
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / f"{checkpoint_id}.tar"
        extract_dir = target_path

        # Create client and get download URL
        client = create_rest_client()
        url_response = client.get_checkpoint_archive_url_from_tinker_path(checkpoint_path).result()

        total_size = _download_checkpoint_archive(
            url_response.url,
            archive_path=archive_path,
            show_progress=True,
            format=format,
        )

        # Extract the checkpoint
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            _safe_extract_tar(archive_path, extract_dir, show_progress=True, format=format)
            destination = str(extract_dir)
            if archive_path.exists():
                archive_path.unlink()
        except Exception as e:
            raise TinkerCliError(
                f"Failed to extract archive: {e}",
                "The downloaded file may be corrupted. Try downloading again.",
            )

        output_obj = CheckpointDownloadOutput(
            checkpoint_path=checkpoint_path,
            file_size_bytes=total_size if total_size > 0 else None,
            destination=destination,
        )
        output_obj.print(format=format)


@cli.command(name="push-hf")
@click.argument("checkpoint_path")
@click.option(
    "--repo",
    "-r",
    "repo_id",
    type=str,
    default=None,
    help="Hugging Face repo ID (e.g., username/my-lora-adapter). If omitted, derive from run.",
)
@click.option(
    "--public",
    is_flag=True,
    help="Create or upload to a public repo (default: private).",
)
@click.option(
    "--revision",
    type=str,
    default=None,
    help="Target branch/revision to upload to (optional).",
)
@click.option(
    "--commit-message",
    type=str,
    default=None,
    help="Commit message for the upload (optional).",
)
@click.option(
    "--create-pr",
    is_flag=True,
    help="Create a pull request instead of pushing to the main branch.",
)
@click.option(
    "--allow-pattern",
    "allow_patterns",
    multiple=True,
    help="Only upload files matching this pattern (can be repeated).",
)
@click.option(
    "--ignore-pattern",
    "ignore_patterns",
    multiple=True,
    help="Skip files matching this pattern (can be repeated).",
)
@click.option(
    "--no-model-card",
    is_flag=True,
    help="Do not create a README.md model card if one is missing.",
)
@click.pass_obj
@handle_api_errors
def push_hf(
    cli_context: CLIContext,
    checkpoint_path: str,
    repo_id: str | None,
    public: bool,
    revision: str | None,
    commit_message: str | None,
    create_pr: bool,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
    no_model_card: bool,
) -> None:
    """Upload a checkpoint to the Hugging Face Hub as a PEFT adapter.

    CHECKPOINT_PATH must be a tinker path (e.g., tinker://run-id/sampler_weights/0001).
    """
    # Validate it's a tinker path
    if not checkpoint_path.startswith("tinker://"):
        raise TinkerCliError(
            f"Invalid checkpoint path: {checkpoint_path}",
            "Checkpoint path must be in the format: tinker://run-id/sampler_weights/0001",
        )

    client = create_rest_client()
    repo_id_out = _export_checkpoint_to_hub(
        client,
        checkpoint_path,
        repo_id,
        private=not public,
        revision=revision,
        commit_message=commit_message,
        create_pr=create_pr,
        exist_ok=True,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        add_model_card=not no_model_card,
    )

    output_obj = CheckpointHubUploadOutput(
        checkpoint_path=checkpoint_path,
        repo_id=repo_id_out,
        revision=revision,
        public=public,
    )
    output_obj.print(format=cli_context.format)
