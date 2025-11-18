"""Commands for managing training runs.

This module implements the 'tinker run' commands, including:
- list: List all training runs
- info: Show details of a specific run
"""

from typing import Any, Dict, List, TYPE_CHECKING

import click

if TYPE_CHECKING:
    from tinker.types import TrainingRun

from ..client import create_rest_client, handle_api_errors
from ..context import CLIContext
from ..output import OutputBase, format_timestamp


class RunListOutput(OutputBase):
    """Output for 'tinker run list' command."""

    def __init__(
        self,
        runs: List["TrainingRun"],
        total_count: int | None = None,
        shown_count: int | None = None,
    ):
        """Initialize with list of training runs.

        Args:
            runs: List of TrainingRun objects
            total_count: Total number of runs available (from cursor)
            shown_count: Number of runs shown in this response
        """
        self.runs = runs
        self.total_count = total_count
        self.shown_count = shown_count if shown_count is not None else len(runs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        # Check if these are Pydantic models
        if self.runs and hasattr(self.runs[0], "model_dump"):
            return {"runs": [run.model_dump() for run in self.runs]}
        return {"runs": [dict(run) for run in self.runs]}

    def get_title(self) -> str | None:
        """Return title for table output."""
        count = len(self.runs)
        if count == 0:
            return "No training runs found"

        # Build the base title
        if count == 1:
            title = "1 training run"
        else:
            title = f"{count} training runs"

        # Add information about remaining runs if available
        if self.total_count is not None and self.total_count > self.shown_count:
            remaining = self.total_count - self.shown_count
            if remaining == 1:
                title += f" (1 more not shown, use --limit to see more)"
            else:
                title += f" ({remaining} more not shown, use --limit to see more)"

        return title

    def get_table_columns(self) -> List[str]:
        """Return column headers for table output."""
        return ["Run ID", "Base Model", "Owner", "LoRA", "Last Update", "Corrupted"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = []
        for run in self.runs:
            # Format LoRA information
            if run.is_lora and run.lora_rank:
                lora_info = f"Rank {run.lora_rank}"
            elif run.is_lora:
                lora_info = "Yes"
            else:
                lora_info = "No"

            rows.append(
                [
                    run.training_run_id,
                    run.base_model,
                    run.model_owner,
                    lora_info,
                    format_timestamp(run.last_request_time),
                    str(run.corrupted),
                ]
            )

        return rows


class RunInfoOutput(OutputBase):
    """Output for 'tinker run info' command."""

    def __init__(self, run: "TrainingRun"):
        """Initialize with a single training run.

        Args:
            run: TrainingRun object
        """
        self.run = run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        if hasattr(self.run, "model_dump"):
            return self.run.model_dump()
        return dict(self.run)

    def get_title(self) -> str | None:
        """Return title for table output."""
        return f"Training Run: {self.run.training_run_id}"

    def get_table_columns(self) -> List[str]:
        """Return column headers for table output."""
        return ["Property", "Value"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = [
            ["Run ID", self.run.training_run_id],
            ["Base Model", self.run.base_model],
            ["Owner", self.run.model_owner],
        ]

        # LoRA information
        if self.run.is_lora:
            if self.run.lora_rank:
                rows.append(["LoRA", f"Yes (Rank {self.run.lora_rank})"])
            else:
                rows.append(["LoRA", "Yes"])
        else:
            rows.append(["LoRA", "No"])

        # Last update time
        rows.append(["Last Update", format_timestamp(self.run.last_request_time)])

        # Corruption status
        rows.append(["Status", "Corrupted" if self.run.corrupted else "Active"])

        # Last checkpoints
        if self.run.last_checkpoint:
            rows.append(["Last Training Checkpoint", self.run.last_checkpoint.checkpoint_id])
            rows.append(["  - Time", format_timestamp(self.run.last_checkpoint.time)])
            rows.append(["  - Path", self.run.last_checkpoint.tinker_path])

        if self.run.last_sampler_checkpoint:
            rows.append(["Last Sampler Checkpoint", self.run.last_sampler_checkpoint.checkpoint_id])
            rows.append(["  - Time", format_timestamp(self.run.last_sampler_checkpoint.time)])
            rows.append(["  - Path", self.run.last_sampler_checkpoint.tinker_path])

        # User metadata if present
        if self.run.user_metadata:
            rows.append(["Metadata", ""])
            for key, value in self.run.user_metadata.items():
                rows.append([f"  - {key}", value])

        return rows


# Click command group for run commands
@click.group()
def cli():
    """Manage training runs."""
    pass


@cli.command()
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of runs to fetch (default: 20, use --limit=0 to fetch all)",
)
@click.pass_obj
@handle_api_errors
def list(cli_context: CLIContext, limit: int) -> None:
    """List all training runs."""
    # Get format from context object
    format = cli_context.format

    # Create client
    client = create_rest_client()

    all_runs = []
    offset = 0
    batch_size = 100  # Fetch in batches of 100 for efficiency

    # First fetch to get initial data and total count
    first_response = client.list_training_runs(
        limit=min(batch_size, limit) if limit > 0 else batch_size, offset=0
    ).result()
    all_runs.extend(first_response.training_runs)
    total_count = first_response.cursor.total_count

    # Determine target count: either user-specified limit or total available
    target_count = limit if limit > 0 else total_count
    target_count = min(target_count, total_count)  # Can't fetch more than exists

    # If we need to fetch more runs, paginate with a progress bar
    if len(all_runs) < target_count:
        with click.progressbar(
            length=target_count,
            label=f"Fetching {'all' if limit == 0 else str(target_count)} training runs",
            show_percent=True,
            show_pos=True,
            show_eta=True,
        ) as bar:
            bar.update(len(all_runs))

            # Fetch remaining runs in batches
            while len(all_runs) < target_count:
                offset = len(all_runs)
                remaining = target_count - len(all_runs)
                next_batch_size = min(batch_size, remaining)

                response = client.list_training_runs(limit=next_batch_size, offset=offset).result()
                all_runs.extend(response.training_runs)
                bar.update(len(response.training_runs))

                # Break if we got fewer than requested (reached the end)
                if len(response.training_runs) < next_batch_size:
                    break

    # Create output object with pagination information
    output = RunListOutput(runs=all_runs, total_count=total_count, shown_count=len(all_runs))

    # Print in requested format
    output.print(format=format)


@cli.command()
@click.argument("run_id")
@click.pass_obj
@handle_api_errors
def info(cli_context: CLIContext, run_id: str) -> None:
    """Show details of a specific run."""
    # Get format from context object
    format = cli_context.format

    # Create client
    client = create_rest_client()

    # Fetch training run details
    response = client.get_training_run(run_id).result()

    # Create output object
    output = RunInfoOutput(run=response)

    # Print in requested format
    output.print(format=format)
