"""Commands for viewing billing usage.

This module implements the 'tinker billing' commands:
- usage: hourly-bucketed billing usage rows for your organization

The CLI renders the usage-events response generically. For table/CSV output,
each event's nested event_info payload is flattened into its row. The table
renders unavailable applicable cost fields as `null`; CSV uses empty cells
and JSON preserves null values.
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


_TOKEN_USAGE_TYPES = frozenset({"training", "sampling_prefill", "sampling_sample"})


def _display_dicts(row_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten rows and render unavailable applicable costs as null.

    The human-readable table distinguishes an unavailable applicable value
    from a field that does not apply to the event type. JSON retains actual
    null values, while CSV uses empty cells for them.
    """
    out = []
    for row in _flat_dicts(row_dicts):
        event_type = row.get("type")
        if event_type in _TOKEN_USAGE_TYPES:
            row = dict(row)
            if row.get("estimated_cost_usd") is None:
                row["estimated_cost_usd"] = "null"
            if row.get("effective_rate_usd_per_million_tokens") is None:
                row["effective_rate_usd_per_million_tokens"] = "null"
        elif event_type == "storage":
            row = dict(row)
            if row.get("estimated_cost_usd") is None:
                row["estimated_cost_usd"] = "null"
            if row.get("effective_rate_usd_per_gigabyte_month") is None:
                row["effective_rate_usd_per_gigabyte_month"] = "null"
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
        self.flat_dicts = _display_dicts(self.row_dicts)
        self.cost_data_through = (
            None
            if response.cost_data_through is None
            else response.cost_data_through.isoformat().replace("+00:00", "Z")
        )
        self.session_dicts = {
            sid: session.model_dump(mode="json") for sid, session in response.sessions.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        # JSON output keeps the true nested response shape, including the
        # session_id -> session attributes mapping.
        return {
            "data": self.row_dicts,
            "sessions": self.session_dicts,
            "cost_data_through": self.cost_data_through,
        }

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
    session_id. Quantities are raw tokens / gigabyte-hours. Token and storage
    rows include estimated gross USD usage cost before credits and commits,
    plus the applicable effective rate per million tokens or per GB-month.
    Token cost is the token rate times token_count / 1,000,000; storage cost is
    the storage rate times gigabyte_hours / 720. These are not invoice amounts
    due.
    A completed UTC day is priced only after its full-day usage quantities
    reconcile with finalized invoice usage line items, or the latest draft
    when no finalized invoice is available. The current incomplete UTC day
    instead uses the published Tinker rate-card snapshot, so its estimate is
    not invoice-reconciled and can change after the day completes.
    Table output renders an unavailable applicable cost or rate as `null`;
    CSV uses blank cells, while the typed SDK and JSON use null. The SDK
    response and JSON output also include `cost_data_through`, a conservative
    invoice-reconciliation watermark that stops before the first unreconciled
    completed day. Table and CSV output do not include this response-level
    field, and current-day rate-card estimates do not advance it. Data lags
    real time by up to a few hours.

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
