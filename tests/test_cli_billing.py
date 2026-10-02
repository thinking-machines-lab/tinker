"""Tests for the 'tinker billing' CLI output formatting.

The CLI renders the response generically — columns come from the response
models, not from the CLI. Each event is an envelope with a nested
event_info payload union; the payload is flattened into the row for
table/CSV output, while JSON output keeps the true nested shape.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from tinker.cli.commands import billing as billing_command
from tinker.cli.commands.billing import (
    BillingUsageOutput,
    CurrentCheckpointStorageUsageOutput,
    _session_rows,
    _write_csv,
)
from tinker.cli.context import CLIContext
from tinker.types import (
    BillingUsageEvent,
    BillingUsageResponse,
    BillingUsageSession,
    CurrentCheckpointStorageUsageItem,
    CurrentCheckpointStorageUsageResponse,
    GetCurrentCheckpointStorageUsageRequest,
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
        "estimated_cost_usd": 0.02469,
        "effective_rate_usd_per_million_tokens": 2.0,
    }
    base.update(overrides)
    return BillingUsageEvent.model_validate(
        {**base, "event_info": TrainingBillingEvent(token_count=12345)}
    )


def _storage(**overrides: object) -> BillingUsageEvent:
    base: dict = {
        "bucket_start": datetime(2026, 7, 13, 5, tzinfo=timezone.utc),
        "bucket_end": datetime(2026, 7, 13, 6, tzinfo=timezone.utc),
        "estimated_cost_usd": 0.00020833333333333335,
        "effective_rate_usd_per_gigabyte_month": 0.1,
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
        cost_data_through=datetime(2026, 7, 14, tzinfo=timezone.utc),
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
        assert training_row[columns.index("estimated_cost_usd")] == "0.02469"
        assert training_row[columns.index("effective_rate_usd_per_million_tokens")] == "2.0"
        assert training_row[columns.index("gigabyte_hours")] == ""  # not on this variant
        assert storage_row[columns.index("type")] == "storage"
        assert storage_row[columns.index("gigabyte_hours")] == "1.5"
        assert storage_row[columns.index("estimated_cost_usd")] == "0.00020833333333333335"
        assert storage_row[columns.index("effective_rate_usd_per_gigabyte_month")] == "0.1"
        assert storage_row[columns.index("token_count")] == ""
        assert storage_row[columns.index("session_id")] == ""  # None renders blank

    def test_to_dict_keeps_nested_shape(self) -> None:
        """JSON output mirrors the wire response: event_info stays nested
        and sessions is the session_id -> user_metadata mapping."""
        data = BillingUsageOutput(_response()).to_dict()
        assert data["data"][0]["bucket_start"] == "2026-07-13T05:00:00Z"
        assert data["data"][0]["estimated_cost_usd"] == 0.02469
        assert data["data"][0]["effective_rate_usd_per_million_tokens"] == 2.0
        assert data["data"][0]["event_info"] == {"type": "training", "token_count": 12345}
        assert data["sessions"] == {"abc": {"user_metadata": {"domino_project": "x"}}}
        assert data["cost_data_through"] == "2026-07-14T00:00:00Z"

    def test_table_labels_applicable_missing_costs_as_null(self) -> None:
        output = BillingUsageOutput(
            _response(
                events=[
                    _training(
                        estimated_cost_usd=None,
                        effective_rate_usd_per_million_tokens=None,
                    ),
                    _storage(
                        estimated_cost_usd=None,
                        effective_rate_usd_per_gigabyte_month=None,
                    ),
                ]
            )
        )
        columns = output.get_table_columns()
        training_row, storage_row = output.get_table_rows()

        assert training_row[columns.index("estimated_cost_usd")] == "null"
        assert training_row[columns.index("effective_rate_usd_per_million_tokens")] == "null"
        assert training_row[columns.index("effective_rate_usd_per_gigabyte_month")] == ""
        assert storage_row[columns.index("estimated_cost_usd")] == "null"
        assert storage_row[columns.index("effective_rate_usd_per_million_tokens")] == ""
        assert storage_row[columns.index("effective_rate_usd_per_gigabyte_month")] == "null"

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
    assert parsed[0]["effective_rate_usd_per_million_tokens"] == "2.0"
    assert parsed[0]["project_id"] == "proj-1"
    assert parsed[1]["type"] == "storage"
    assert parsed[1]["gigabyte_hours"] == "1.5"
    assert parsed[1]["estimated_cost_usd"] == "0.00020833333333333335"
    assert parsed[1]["effective_rate_usd_per_gigabyte_month"] == "0.1"
    assert parsed[1]["token_count"] == ""


def test_write_csv_keeps_applicable_missing_costs_blank(tmp_path: Path) -> None:
    path = tmp_path / "null-usage.csv"
    _write_csv(
        [
            _training(
                estimated_cost_usd=None,
                effective_rate_usd_per_million_tokens=None,
            ),
            _storage(
                estimated_cost_usd=None,
                effective_rate_usd_per_gigabyte_month=None,
            ),
        ],
        str(path),
    )
    with open(path, newline="") as f:
        training, storage = list(csv.DictReader(f))

    assert training["estimated_cost_usd"] == ""
    assert training["effective_rate_usd_per_million_tokens"] == ""
    assert training["effective_rate_usd_per_gigabyte_month"] == ""
    assert storage["estimated_cost_usd"] == ""
    assert storage["effective_rate_usd_per_million_tokens"] == ""
    assert storage["effective_rate_usd_per_gigabyte_month"] == ""


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


def _checkpoint_storage_response() -> CurrentCheckpointStorageUsageResponse:
    return CurrentCheckpointStorageUsageResponse(
        effective_rate_usd_per_gigabyte_month=0.1,
        data=[
            CurrentCheckpointStorageUsageItem(
                project_id="project-a",
                org_user_urn="tml:organization_user:user-a",
                user_email="ada@example.com",
                user_name="Ada Lovelace",
                checkpoint_count=2,
                size_bytes=2**30,
                size_gigabytes=1.0,
                estimated_monthly_cost_usd=0.1,
            )
        ],
    )


def test_current_checkpoint_storage_output_keeps_identity_and_rate() -> None:
    output = CurrentCheckpointStorageUsageOutput(_checkpoint_storage_response())

    assert output.to_dict() == {
        "effective_rate_usd_per_gigabyte_month": 0.1,
        "data": [
            {
                "project_id": "project-a",
                "org_user_urn": "tml:organization_user:user-a",
                "user_email": "ada@example.com",
                "user_name": "Ada Lovelace",
                "checkpoint_count": 2,
                "size_bytes": 2**30,
                "size_gigabytes": 1.0,
                "estimated_monthly_cost_usd": 0.1,
            }
        ],
    }
    columns = output.get_table_columns()
    assert "snapshot_at" not in columns
    assert output.get_title() == "Current checkpoint storage"
    (row,) = output.get_table_rows()
    assert row[columns.index("user_email")] == "ada@example.com"
    assert row[columns.index("org_user_urn")] == "tml:organization_user:user-a"
    assert "user_id" not in columns
    assert row[columns.index("estimated_monthly_cost_usd")] == "0.1"


@pytest.mark.parametrize("project_id", [None, "project-a"])
def test_checkpoint_storage_command_org_default_and_project_filter(
    monkeypatch: pytest.MonkeyPatch, project_id: str | None
) -> None:
    future = MagicMock()
    future.result.return_value = _checkpoint_storage_response()
    client = MagicMock()
    client.get_current_checkpoint_storage_usage.return_value = future
    monkeypatch.setattr(billing_command, "create_rest_client", lambda: client)

    result = CliRunner().invoke(
        billing_command.cli,
        ["checkpoint-storage"] + ([] if project_id is None else ["--project-id", project_id]),
        obj=CLIContext(format="json"),
    )

    assert result.exit_code == 0, result.output
    client.get_current_checkpoint_storage_usage.assert_called_once_with(
        project_id=project_id,
    )
    assert '"user_email": "ada@example.com"' in result.output
    assert '"org_user_urn": "tml:organization_user:user-a"' in result.output
    assert '"user_id"' not in result.output
    assert "snapshot" not in result.output.lower()


def test_checkpoint_storage_csv_omits_internal_timestamp(tmp_path: Path) -> None:
    output = CurrentCheckpointStorageUsageOutput(_checkpoint_storage_response())
    path = tmp_path / "checkpoints.csv"
    _write_csv(output.table_dicts, str(path))

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert "snapshot_at" not in (reader.fieldnames or [])
        assert "user_id" not in (reader.fieldnames or [])
        (row,) = list(reader)
    assert row["checkpoint_count"] == "2"
    assert row["user_email"] == "ada@example.com"
    assert row["org_user_urn"] == "tml:organization_user:user-a"


def test_checkpoint_storage_empty_output_omits_internal_timestamp() -> None:
    response = CurrentCheckpointStorageUsageResponse(
        effective_rate_usd_per_gigabyte_month=0.1,
        data=[],
    )
    output = CurrentCheckpointStorageUsageOutput(response)
    assert output.get_title() == "No active checkpoint storage"
    assert output.to_dict() == {"effective_rate_usd_per_gigabyte_month": 0.1, "data": []}
    assert "snapshot_at" not in response.model_dump()
    assert "snapshot_at" not in response.model_json_schema()["properties"]


def test_checkpoint_storage_help_explains_lag_without_internal_details() -> None:
    result = CliRunner().invoke(billing_command.cli, ["checkpoint-storage", "--help"])

    assert result.exit_code == 0, result.output
    assert "1-2 hours" in result.output
    assert "snapshot" not in result.output.lower()
    assert "entire authenticated organization" in " ".join(result.output.split())
    assert "current publicly available rates" in " ".join(result.output.split())
    assert "rate card" not in result.output.lower()
    assert "--user-email" not in result.output


def test_checkpoint_storage_request_only_accepts_project_filter() -> None:
    assert GetCurrentCheckpointStorageUsageRequest().model_dump(exclude_none=True) == {}
    assert GetCurrentCheckpointStorageUsageRequest(project_id="project-a").model_dump(
        exclude_none=True
    ) == {"project_id": "project-a"}
    with pytest.raises(ValidationError):
        GetCurrentCheckpointStorageUsageRequest.model_validate({"user_email": "ada@example.com"})
    result = CliRunner().invoke(
        billing_command.cli, ["checkpoint-storage", "--user-email", "ada@example.com"]
    )
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "--user-email" in result.output


def test_checkpoint_storage_ignores_timestamp_from_older_server() -> None:
    payload = _checkpoint_storage_response().model_dump(mode="json")
    response = CurrentCheckpointStorageUsageResponse.model_validate(
        {**payload, "snapshot_at": "2026-09-23T18:20:00Z"}
    )

    assert response.model_dump(mode="json") == payload
    assert not hasattr(response, "snapshot_at")
    assert CurrentCheckpointStorageUsageOutput(response).to_dict() == payload
