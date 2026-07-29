"""Commands for managing sessions.

This module implements the 'tinker session' commands, including:
- export-trace: Export a session's timeline as a Perfetto trace (.pftrace)
"""

from typing import Any, Dict, List

import click

from ..client import create_rest_client, handle_api_errors
from ..context import CLIContext
from ..exceptions import TinkerCliError
from ..output import OutputBase, format_size


class SessionTraceExportOutput(OutputBase):
    """Output for 'tinker session export-trace' command."""

    def __init__(
        self,
        session_id: str,
        url: str | None = None,
        destination: str | None = None,
        file_size_bytes: int | None = None,
    ):
        """Initialize with trace export information.

        Args:
            session_id: The session ID the trace was exported for
            url: Signed download URL (only for --url-only)
            destination: Where the trace file was written
            file_size_bytes: Size of the downloaded trace in bytes
        """
        self.session_id = session_id
        self.url = url
        self.destination = destination
        self.file_size_bytes = file_size_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        result: Dict[str, Any] = {"session_id": self.session_id}
        if self.url is not None:
            result["url"] = self.url
        if self.destination is not None:
            result["destination"] = self.destination
        if self.file_size_bytes is not None:
            result["file_size_bytes"] = self.file_size_bytes
        return result

    def get_title(self) -> str | None:
        """Return title for table output."""
        return f"Trace Export: {self.session_id}"

    def get_table_columns(self) -> List[str]:
        """Return column headers for table output."""
        return ["Property", "Value"]

    def get_table_rows(self) -> List[List[str]]:
        """Return rows for table output."""
        rows = [["Session ID", self.session_id]]
        if self.url is not None:
            rows.append(["Download URL", self.url])
        if self.destination is not None:
            rows.append(["Saved to", self.destination])
        if self.file_size_bytes is not None:
            rows.append(["Size", format_size(self.file_size_bytes)])
        return rows


def _download_trace(
    url: str,
    *,
    trace_path,
    show_progress: bool,
    format: str,
) -> int:
    """Download the trace from the signed URL to trace_path. Returns bytes written."""
    import urllib.error
    import urllib.request

    total_written = 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            if show_progress and format != "json":
                with click.progressbar(
                    length=total_size,
                    label="Downloading trace",
                    show_percent=True,
                    show_pos=True,
                    show_eta=True,
                ) as bar:
                    with open(trace_path, "wb") as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            total_written += len(chunk)
                            bar.update(len(chunk))
            else:
                with open(trace_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_written += len(chunk)
    except urllib.error.URLError as e:
        raise TinkerCliError(
            f"Failed to download trace: {e}",
            "The signed URL may have expired (it is valid for about an hour). "
            "Re-run the command to get a fresh one.",
        ) from e
    except IOError as e:
        raise TinkerCliError(
            f"Failed to save trace: {e}",
            f"Please check that you have write permissions to {trace_path.parent}",
        ) from e

    return total_written


# Click command group for session commands
@click.group()
def cli():
    """Manage sessions."""
    pass


@cli.command(name="export-trace")
@click.argument("session_id")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: ./<session-id>.pftrace)",
)
@click.option(
    "--url-only",
    is_flag=True,
    help="Print the signed download URL instead of downloading the trace",
)
@click.pass_obj
@handle_api_errors
def export_trace(
    cli_context: CLIContext,
    session_id: str,
    output: str | None,
    url_only: bool,
) -> None:
    """Export a session's timeline as a Perfetto trace (.pftrace).

    Builds a trace of the session's training and sampling requests on the
    server and downloads it. Open the resulting file in https://ui.perfetto.dev
    to view the timeline.

    Examples:

    \b
        # Creates ./<session-id>.pftrace
        tinker session export-trace <session-id>
    \b
        # Custom output path
        tinker session export-trace <session-id> --output my-session.pftrace
    \b
        # Just print the signed download URL (expires after about an hour)
        tinker session export-trace <session-id> --url-only
    """
    # Lazy import to maintain fast CLI startup
    from pathlib import Path

    format = cli_context.format

    client = create_rest_client()

    if format != "json":
        click.echo(f"Exporting trace for session {session_id} (this may take a while)...", err=True)
    url = client.export_session_trace(session_id).result()

    if url_only:
        if format == "json":
            output_obj = SessionTraceExportOutput(session_id=session_id, url=url)
            output_obj.print(format=format)
        else:
            # Bare URL on stdout so it can be used in command substitution,
            # e.g. curl -o trace.pftrace "$(tinker session export-trace <id> --url-only)"
            click.echo(url)
        return

    trace_path = Path(output) if output else Path.cwd() / f"{session_id}.pftrace"
    if trace_path.parent != Path("."):
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = _download_trace(
        url,
        trace_path=trace_path,
        show_progress=True,
        format=format,
    )

    output_obj = SessionTraceExportOutput(
        session_id=session_id,
        destination=str(trace_path),
        file_size_bytes=total_written if total_written > 0 else None,
    )
    output_obj.print(format=format)
