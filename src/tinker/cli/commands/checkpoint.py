"""Commands for managing checkpoints.

This module implements the 'tinker checkpoint' commands, including:
- list: List all checkpoints or checkpoints for a specific run
- info: Show details of a specific checkpoint
"""

from typing import TYPE_CHECKING, Any, Dict, List

import click

if TYPE_CHECKING:
    from tinker.types import Checkpoint

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
        return ["Checkpoint ID", "Type", "Size", "Public", "Created", "Path"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = []
        for ckpt in self.checkpoints:
            rows.append(
                [
                    ckpt.checkpoint_id,
                    ckpt.checkpoint_type,
                    format_size(ckpt.size_bytes) if hasattr(ckpt, "size_bytes") else "N/A",
                    format_bool(ckpt.public),
                    format_timestamp(ckpt.time),
                    ckpt.tinker_path,
                ]
            )
        return rows


class CheckpointInfoOutput(OutputBase):
    """Output for 'tinker checkpoint info' command."""

    def __init__(self, checkpoint: "Checkpoint"):
        """Initialize with a single checkpoint.

        Args:
            checkpoint: Checkpoint object
        """
        self.checkpoint = checkpoint

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        if hasattr(self.checkpoint, "model_dump"):
            return self.checkpoint.model_dump()
        return dict(self.checkpoint)

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
        if hasattr(self.checkpoint, "size_bytes"):
            rows.append(["Size", format_size(self.checkpoint.size_bytes)])

        # Public status
        rows.append(["Public", format_bool(self.checkpoint.public)])

        # Creation time
        rows.append(["Created", format_timestamp(self.checkpoint.time)])

        # Parse training run ID from path
        if self.checkpoint.tinker_path.startswith("tinker://"):
            parts = self.checkpoint.tinker_path.replace("tinker://", "").split("/")
            if parts:
                rows.append(["Training Run ID", parts[0]])

        return rows


def get_checkpoint_from_path(checkpoint_path: str) -> "Checkpoint":
    """Get checkpoint details from a tinker path.

    Args:
        checkpoint_path: A tinker path like "tinker://run-id/weights/0001"

    Returns:
        Checkpoint object

    Raises:
        TinkerCliError: If the checkpoint cannot be retrieved
    """
    # Lazy import
    from tinker import ParsedCheckpointTinkerPath

    try:
        parsed = ParsedCheckpointTinkerPath.from_tinker_path(checkpoint_path)
        client = create_rest_client()

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

    checkpoint = get_checkpoint_from_path(checkpoint_path)

    # Create output object
    output = CheckpointInfoOutput(checkpoint=checkpoint)

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
@click.argument("checkpoint_path")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_obj
@handle_api_errors
def delete(cli_context: CLIContext, checkpoint_path: str, yes: bool) -> None:
    """Delete a checkpoint permanently.

    CHECKPOINT_PATH must be a tinker path (e.g., tinker://run-id/weights/0001).
    Only the owner of the training run can delete checkpoints.

    WARNING: This action is permanent and cannot be undone.
    """
    # Validate it's a tinker path
    if not checkpoint_path.startswith("tinker://"):
        raise TinkerCliError(
            f"Invalid checkpoint path: {checkpoint_path}",
            "Checkpoint path must be in the format: tinker://run-id/weights/0001",
        )

    # Get format from context object
    format = cli_context.format

    # If not using --yes, show checkpoint info and prompt for confirmation
    if not yes:
        try:
            checkpoint = get_checkpoint_from_path(checkpoint_path)

            # Display checkpoint info using the same format as 'info' command
            output = CheckpointInfoOutput(checkpoint)
            output.print(format=format)
            click.echo()

        except TinkerCliError:
            # If we can't get checkpoint info, still allow deletion attempt
            # The API will return appropriate error if checkpoint doesn't exist
            click.echo(f"Checkpoint path: {checkpoint_path}")
            click.echo()

        # Confirmation prompt
        click.echo("WARNING: This action is permanent and cannot be undone.")
        if not click.confirm("Are you sure you want to delete this checkpoint?"):
            click.echo("Deletion cancelled.")
            return

    # Create client and delete
    client = create_rest_client()
    client.delete_checkpoint_from_tinker_path(checkpoint_path).result()
