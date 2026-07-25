"""Commands for viewing billing usage.

This module implements the 'tinker billing' commands:
- usage: hourly-bucketed billing usage rows for your organization

The CLI is deliberately schema-agnostic: it renders whatever fields the
usage-events response carries and never interprets individual fields, so
response schema changes only require updating the response models, not
this module. The one structural step: each event's nested event_info
payload (a tagged union) is flattened into its row for table/CSV output —
columns become the union of the envelope and payload fields, blank where a
field does not apply — while JSON output preserves the true nested shape.
"""

import csv
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import click

if TYPE_CHECKING:
    from tinker.types import BillingUsageResponse

from ..client import create_rest_client, handle_api_errors
from ..context import CLIContext
from ..output import OutputBase


def _row_dicts(rows: Sequence[Any]) -> List[Dict[str, Any]]:
    """Rows as JSON-safe dicts, exactly as the API returned them (plain
    dicts pass through)."""
    return [row if isinstance(row, dict) else row.model_dump(mode="json") for row in rows]


def _session_rows(sessions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The sessions mapping (session_id -> BillingUsageSession) as flat rows
    for CSV output, user_metadata JSON-encoded per row."""
    import json

    return [
        {
            "session_id": sid,
            "user_metadata": (
                None if session.user_metadata is None else json.dumps(session.user_metadata)
            ),
        }
        for sid, session in sessions.items()
    ]


def _flat_dicts(row_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows with any nested event_info payload flattened into the row, for
    rectangular (table/CSV) output. Envelope and payload field names do not
    collide by construction."""
    out = []
    for row in row_dicts:
        info = row.get("event_info")
        if isinstance(info, dict):
            row = {**{k: v for k, v in row.items() if k != "event_info"}, **info}
        out.append(row)
    return out


def _columns(row_dicts: List[Dict[str, Any]]) -> List[str]:
    """Column order: the first row's key order (the response model's field
    order), plus any keys only later rows carry."""
    columns: List[str] = []
    for row in row_dicts:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


class BillingUsageOutput(OutputBase):
    """Output for 'tinker billing usage'. The table view shows the usage
    rows; the sessions side-table is included in JSON output and available
    via --sessions-csv."""

    def __init__(self, response: "BillingUsageResponse"):
        self.row_dicts = _row_dicts(response.data)
        self.flat_dicts = _flat_dicts(self.row_dicts)
        self.session_dicts = {
            sid: session.model_dump(mode="json") for sid, session in response.sessions.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        # JSON output keeps the true nested response shape, including the
        # session_id -> session attributes mapping.
        return {"data": self.row_dicts, "sessions": self.session_dicts}

    def get_title(self) -> str | None:
        count = len(self.row_dicts)
        if count == 0:
            return "No billing usage in this window"
        return f"{count} hourly usage row(s)"

    def get_table_columns(self) -> List[str]:
        return _columns(self.flat_dicts)

    def get_table_rows(self) -> List[List[str]]:
        columns = self.get_table_columns()
        return [[_cell(row.get(column)) for column in columns] for row in self.flat_dicts]


def _write_csv(rows: Sequence[Any], path: str) -> None:
    row_dicts = _flat_dicts(_row_dicts(rows))

    def write_to(out: Any) -> None:
        writer = csv.DictWriter(out, fieldnames=_columns(row_dicts), restval="")
        writer.writeheader()
        writer.writerows(row_dicts)

    if path == "-":
        write_to(sys.stdout)
    else:
        with open(path, "w", newline="") as f:
            write_to(f)
        click.echo(f"Wrote {len(rows)} row(s) to {path}", err=True)


# Click command group for billing commands
@click.group()
def cli():
    """View billing usage."""
    pass


@cli.command(name="usage")
@click.argument("starting_on")
@click.argument("ending_before")
@click.option(
    "--csv",
    "csv_path",
    default=None,
    metavar="PATH",
    help="Write usage rows as CSV to PATH instead of table/JSON output ('-' for stdout)",
)
@click.option(
    "--sessions-csv",
    "sessions_csv_path",
    default=None,
    metavar="PATH",
    help="Also write the per-session side-table (session_id + user_metadata) "
    "as CSV to PATH ('-' for stdout); join it against the usage rows on session_id",
)
@click.pass_obj
@handle_api_errors
def usage(
    cli_context: CLIContext,
    starting_on: str,
    ending_before: str,
    csv_path: str | None,
    sessions_csv_path: str | None,
) -> None:
    """Show hourly billing usage for your organization.

    STARTING_ON and ENDING_BEFORE are RFC 3339 timestamps aligned to UTC hour
    boundaries (e.g. 2026-07-13T00:00:00Z), at most 14 days apart. Returns one
    row per (hour x usage type x base model x session x user), annotated with
    the project the usage belongs to. Session user metadata comes as a
    separate per-session table (JSON output / --sessions-csv) to join on
    session_id. Quantities are raw tokens / gigabyte-hours; dollar amounts
    are not included. Data lags real time by up to a few hours.

    There are no filter flags: export the window once and filter the
    CSV/JSON client-side.

    Requires billing view access in your organization.
    """
    client = create_rest_client()
    response = client.get_billing_usage(starting_on, ending_before).result()

    if sessions_csv_path is not None:
        _write_csv(_session_rows(response.sessions), sessions_csv_path)
    if csv_path is not None:
        _write_csv(response.data, csv_path)
    if csv_path is None and sessions_csv_path is None:
        BillingUsageOutput(response).print(format=cli_context.format)
