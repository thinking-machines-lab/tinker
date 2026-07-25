"""Tests for the 'tinker billing' CLI output formatting.

The CLI renders the response generically — columns come from the response
models, not from the CLI. Each event is an envelope with a nested
event_info payload union; the payload is flattened into the row for
table/CSV output, while JSON output keeps the true nested shape.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

from tinker.cli.commands.billing import BillingUsageOutput, _session_rows, _write_csv
from tinker.types import (
    BillingUsageEvent,
    BillingUsageResponse,
    BillingUsageSession,
    StorageBillingEvent,
    TrainingBillingEvent,
)


def _training(**overrides: object) -> BillingUsageEvent:
    base: dict = {
        "bucket_start": datetime(2026, 7, 13, 5, tzinfo=timezone.utc),
        "bucket_end": datetime(2026, 7, 13, 6, tzinfo=timezone.utc),
        "base_model": "Qwen/Qwen3.5-9B-Base",
        "user_id": "tml:organization_user:u1",
        "user_name": "Ada Lovelace",
        "session_id": "abc",
        "project_id": "proj-1",
    }
    base.update(overrides)
    return BillingUsageEvent.model_validate(
        {**base, "event_info": TrainingBillingEvent(token_count=12345)}
    )


def _storage(**overrides: object) -> BillingUsageEvent:
    base: dict = {
        "bucket_start": datetime(2026, 7, 13, 5, tzinfo=timezone.utc),
        "bucket_end": datetime(2026, 7, 13, 6, tzinfo=timezone.utc),
    }
    base.update(overrides)
    return BillingUsageEvent.model_validate(
        {**base, "event_info": StorageBillingEvent(gigabyte_hours=1.5)}
    )


def _response(events: list | None = None, sessions: dict | None = None) -> BillingUsageResponse:
    if sessions is None:
        sessions = {"abc": BillingUsageSession(user_metadata={"domino_project": "x"})}
    return BillingUsageResponse(
        data=[_training()] if events is None else events,
        sessions=sessions,
    )


class TestBillingUsageOutput:
    def test_table_flattens_event_info(self) -> None:
        """The nested payload union is flattened into the row for tabular
        output: columns are the envelope fields plus the union of the
        payload fields, blank where a field does not apply."""
        output = BillingUsageOutput(_response(events=[_training(), _storage()]))
        columns = output.get_table_columns()
        assert "event_info" not in columns
        assert "type" in columns and "token_count" in columns and "gigabyte_hours" in columns
        training_row, storage_row = output.get_table_rows()
        assert training_row[columns.index("type")] == "training"
        assert training_row[columns.index("token_count")] == "12345"
        assert training_row[columns.index("gigabyte_hours")] == ""  # not on this variant
        assert storage_row[columns.index("type")] == "storage"
        assert storage_row[columns.index("gigabyte_hours")] == "1.5"
        assert storage_row[columns.index("token_count")] == ""
        assert storage_row[columns.index("session_id")] == ""  # None renders blank

    def test_to_dict_keeps_nested_shape(self) -> None:
        """JSON output mirrors the wire response: event_info stays nested
        and sessions is the session_id -> user_metadata mapping."""
        data = BillingUsageOutput(_response()).to_dict()
        assert data["data"][0]["bucket_start"] == "2026-07-13T05:00:00Z"
        assert data["data"][0]["event_info"] == {"type": "training", "token_count": 12345}
        assert data["sessions"] == {"abc": {"user_metadata": {"domino_project": "x"}}}

    def test_empty_rows(self) -> None:
        output = BillingUsageOutput(_response(events=[], sessions={}))
        assert output.get_title() == "No billing usage in this window"
        assert output.get_table_columns() == []
        assert output.get_table_rows() == []


def test_write_csv(tmp_path: Path) -> None:
    path = tmp_path / "usage.csv"
    _write_csv([_training(), _storage()], str(path))
    with open(path, newline="") as f:
        parsed = list(csv.DictReader(f))
    assert len(parsed) == 2
    # header is the envelope fields plus the union of the payload fields;
    # blanks where a field does not apply to a row's variant
    assert "event_info" not in parsed[0]
    assert parsed[0]["type"] == "training"
    assert parsed[0]["token_count"] == "12345"
    assert parsed[0]["project_id"] == "proj-1"
    assert parsed[1]["type"] == "storage"
    assert parsed[1]["gigabyte_hours"] == "1.5"
    assert parsed[1]["token_count"] == ""


def test_write_sessions_csv(tmp_path: Path) -> None:
    """The sessions mapping flattens to (session_id, user_metadata) CSV rows
    that join against the usage CSV on session_id; metadata-less sessions
    get a blank cell."""
    path = tmp_path / "sessions.csv"
    _write_csv(
        _session_rows(
            {
                "abc": BillingUsageSession(user_metadata={"domino_project": "x"}),
                "empty": BillingUsageSession(user_metadata=None),
            }
        ),
        str(path),
    )
    with open(path, newline="") as f:
        parsed = list(csv.DictReader(f))
    assert parsed == [
        {"session_id": "abc", "user_metadata": '{"domino_project": "x"}'},
        {"session_id": "empty", "user_metadata": ""},
    ]
