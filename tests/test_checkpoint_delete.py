"""Tests for bulk checkpoint deletion: CLI flags and date parsing."""

from datetime import UTC, datetime, timedelta

import pytest
from click.testing import CliRunner

from tinker.cli.commands.checkpoint import _filter_checkpoints, _parse_date


class TestParseDate:
    """Tests for the _parse_date ISO 8601 parser."""

    def test_date_only(self) -> None:
        dt = _parse_date("2024-01-15")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.tzinfo is not None

    def test_datetime_with_z(self) -> None:
        dt = _parse_date("2024-06-01T12:00:00Z")
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.hour == 12
        assert dt.tzinfo is not None

    def test_datetime_with_offset(self) -> None:
        dt = _parse_date("2024-06-01T12:00:00+00:00")
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_datetime_naive_gets_utc(self) -> None:
        dt = _parse_date("2024-06-01T12:00:00")
        assert dt.tzinfo is not None

    def test_whitespace_stripped(self) -> None:
        dt = _parse_date("  2024-01-01  ")
        assert dt.year == 2024

    def test_invalid_raises(self) -> None:
        from tinker.cli.exceptions import TinkerCliError

        with pytest.raises(TinkerCliError):
            _parse_date("not-a-date")

    def test_invalid_format_raises(self) -> None:
        from tinker.cli.exceptions import TinkerCliError

        with pytest.raises(TinkerCliError):
            _parse_date("01/15/2024")


class TestFilterCheckpoints:
    """Tests for the _filter_checkpoints function."""

    @pytest.fixture()
    def sample_checkpoints(self):
        from tinker.types.checkpoint import Checkpoint

        now = datetime.now(UTC)
        return [
            Checkpoint(
                checkpoint_id="weights/0001",
                checkpoint_type="training",
                time=now - timedelta(days=10),
                tinker_path="tinker://run-1/weights/0001",
                size_bytes=1000,
            ),
            Checkpoint(
                checkpoint_id="weights/0002",
                checkpoint_type="training",
                time=now - timedelta(days=3),
                tinker_path="tinker://run-1/weights/0002",
                size_bytes=2000,
            ),
            Checkpoint(
                checkpoint_id="sampler_weights/0001",
                checkpoint_type="sampler",
                time=now - timedelta(days=10),
                tinker_path="tinker://run-1/sampler_weights/0001",
                size_bytes=500,
            ),
            Checkpoint(
                checkpoint_id="weights/0003",
                checkpoint_type="training",
                time=now - timedelta(hours=1),
                tinker_path="tinker://run-1/weights/0003",
                size_bytes=3000,
            ),
        ]

    def test_no_filters(self, sample_checkpoints) -> None:
        result = _filter_checkpoints(sample_checkpoints, None, None, None)
        assert len(result) == 4

    def test_filter_by_weights_type(self, sample_checkpoints) -> None:
        result = _filter_checkpoints(sample_checkpoints, "weights", None, None)
        assert len(result) == 3
        assert all(c.checkpoint_type == "training" for c in result)

    def test_filter_by_sampler_weights_type(self, sample_checkpoints) -> None:
        result = _filter_checkpoints(sample_checkpoints, "sampler_weights", None, None)
        assert len(result) == 1
        assert result[0].checkpoint_type == "sampler"

    def test_filter_before(self, sample_checkpoints) -> None:
        # Before 7 days ago → only the 10-day-old checkpoints
        cutoff = datetime.now(UTC) - timedelta(days=7)
        result = _filter_checkpoints(sample_checkpoints, None, cutoff, None)
        assert len(result) == 2
        assert all("0001" in c.checkpoint_id for c in result)

    def test_filter_after(self, sample_checkpoints) -> None:
        # After 7 days ago → the 3-day-old and 1-hour-old checkpoints
        cutoff = datetime.now(UTC) - timedelta(days=7)
        result = _filter_checkpoints(sample_checkpoints, None, None, cutoff)
        assert len(result) == 2
        paths = {c.tinker_path for c in result}
        assert "tinker://run-1/weights/0002" in paths
        assert "tinker://run-1/weights/0003" in paths

    def test_filter_date_range(self, sample_checkpoints) -> None:
        # Between 5 and 2 days ago → only the 3-day-old checkpoint
        after_dt = datetime.now(UTC) - timedelta(days=5)
        before_dt = datetime.now(UTC) - timedelta(days=2)
        result = _filter_checkpoints(sample_checkpoints, None, before_dt, after_dt)
        assert len(result) == 1
        assert result[0].tinker_path == "tinker://run-1/weights/0002"

    def test_filter_by_type_and_before(self, sample_checkpoints) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=7)
        result = _filter_checkpoints(sample_checkpoints, "weights", cutoff, None)
        assert len(result) == 1
        assert result[0].tinker_path == "tinker://run-1/weights/0001"

    def test_invalid_type_raises(self, sample_checkpoints) -> None:
        from tinker.cli.exceptions import TinkerCliError

        with pytest.raises(TinkerCliError):
            _filter_checkpoints(sample_checkpoints, "invalid_type", None, None)


class TestDeleteCLIValidation:
    """Tests for CLI delete command argument validation."""

    def _get_error_message(self, result) -> str:
        """Get error message from either output or exception."""
        if result.output:
            return result.output
        if result.exception:
            return str(result.exception)
        return ""

    def test_no_args_shows_error(self) -> None:
        from tinker.cli.commands.checkpoint import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["delete"])
        assert result.exit_code != 0

    def test_paths_and_run_id_conflict(self) -> None:
        from tinker.cli.commands.checkpoint import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "tinker://run-1/weights/0001", "--run-id", "run-1"])
        assert result.exit_code != 0
        assert "Cannot specify both" in self._get_error_message(result)

    def test_type_without_run_id_error(self) -> None:
        from tinker.cli.commands.checkpoint import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["delete", "tinker://run-1/weights/0001", "--type", "weights"])
        assert result.exit_code != 0
        assert "--run-id" in self._get_error_message(result)

    def test_before_without_run_id_error(self) -> None:
        from tinker.cli.commands.checkpoint import cli

        runner = CliRunner()
        result = runner.invoke(
            cli, ["delete", "tinker://run-1/weights/0001", "--before", "2024-01-01"]
        )
        assert result.exit_code != 0
        assert "--run-id" in self._get_error_message(result)

    def test_after_without_run_id_error(self) -> None:
        from tinker.cli.commands.checkpoint import cli

        runner = CliRunner()
        result = runner.invoke(
            cli, ["delete", "tinker://run-1/weights/0001", "--after", "2024-01-01"]
        )
        assert result.exit_code != 0
        assert "--run-id" in self._get_error_message(result)
