"""Tests for CLI output formatting utilities."""

import re
from datetime import datetime, timedelta, timezone

from tinker.cli.output import format_bool, format_size, format_timestamp


class TestFormatTimestamp:
    """Tests for the format_timestamp function."""

    def test_none_returns_na(self) -> None:
        assert format_timestamp(None) == "N/A"

    def test_empty_string_returns_na(self) -> None:
        assert format_timestamp("") == "N/A"

    def test_just_now_past(self) -> None:
        """Times within the last minute should show 'just now'."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(seconds=30)
        assert format_timestamp(dt) == "just now"

    def test_just_now_future(self) -> None:
        """Times within the next minute should show 'in less than a minute'."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(seconds=30)
        assert format_timestamp(dt) == "in less than a minute"

    def test_minutes_ago(self) -> None:
        """Times a few minutes in the past."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(minutes=5, seconds=30)
        result = format_timestamp(dt)
        # Allow for slight timing variations (4-5 minutes)
        assert re.match(r"[45] minutes ago", result), (
            f"Expected '4 minutes ago' or '5 minutes ago', got '{result}'"
        )

    def test_minutes_future(self) -> None:
        """Times a few minutes in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(minutes=5, seconds=30)
        result = format_timestamp(dt)
        # Allow for slight timing variations (4-5 minutes)
        assert re.match(r"in [45] minutes", result), (
            f"Expected 'in 4 minutes' or 'in 5 minutes', got '{result}'"
        )

    def test_one_minute_ago(self) -> None:
        """Singular 'minute' for exactly 1 minute."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(minutes=1, seconds=30)
        assert format_timestamp(dt) == "1 minute ago"

    def test_one_minute_future(self) -> None:
        """Singular 'minute' for exactly 1 minute in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(minutes=1, seconds=30)
        assert format_timestamp(dt) == "in 1 minute"

    def test_hours_ago(self) -> None:
        """Times a few hours in the past."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(hours=3, minutes=30)
        result = format_timestamp(dt)
        # Allow for slight timing variations (2-3 hours)
        assert re.match(r"[23] hours ago", result), (
            f"Expected '2 hours ago' or '3 hours ago', got '{result}'"
        )

    def test_hours_future(self) -> None:
        """Times a few hours in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(hours=3, minutes=30)
        result = format_timestamp(dt)
        # Allow for slight timing variations (2-3 hours)
        assert re.match(r"in [23] hours", result), (
            f"Expected 'in 2 hours' or 'in 3 hours', got '{result}'"
        )

    def test_one_hour_ago(self) -> None:
        """Singular 'hour' for exactly 1 hour."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(hours=1, minutes=30)
        assert format_timestamp(dt) == "1 hour ago"

    def test_one_hour_future(self) -> None:
        """Singular 'hour' for exactly 1 hour in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(hours=1, minutes=30)
        assert format_timestamp(dt) == "in 1 hour"

    def test_days_ago(self) -> None:
        """Times a few days in the past."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(days=3, hours=12)
        result = format_timestamp(dt)
        # Allow for slight timing variations (2-3 days)
        assert re.match(r"[23] days ago", result), (
            f"Expected '2 days ago' or '3 days ago', got '{result}'"
        )

    def test_days_future(self) -> None:
        """Times a few days in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(days=3, hours=12)
        result = format_timestamp(dt)
        # Allow for slight timing variations (2-3 days)
        assert re.match(r"in [23] days", result), (
            f"Expected 'in 2 days' or 'in 3 days', got '{result}'"
        )

    def test_one_day_ago(self) -> None:
        """Singular 'day' for exactly 1 day."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(days=1, hours=12)
        assert format_timestamp(dt) == "1 day ago"

    def test_one_day_future(self) -> None:
        """Singular 'day' for exactly 1 day in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(days=1, hours=12)
        assert format_timestamp(dt) == "in 1 day"

    def test_weeks_ago(self) -> None:
        """Times a few weeks in the past."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(weeks=2, days=3)
        result = format_timestamp(dt)
        # Allow for slight timing variations (1-2 weeks)
        assert re.match(r"[12] weeks? ago", result), (
            f"Expected '1 week ago' or '2 weeks ago', got '{result}'"
        )

    def test_weeks_future(self) -> None:
        """Times a few weeks in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(weeks=2, days=3)
        result = format_timestamp(dt)
        # Allow for slight timing variations (1-2 weeks)
        assert re.match(r"in [12] weeks?", result), (
            f"Expected 'in 1 week' or 'in 2 weeks', got '{result}'"
        )

    def test_one_week_ago(self) -> None:
        """Singular 'week' for exactly 1 week."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(weeks=1, days=3)
        assert format_timestamp(dt) == "1 week ago"

    def test_one_week_future(self) -> None:
        """Singular 'week' for exactly 1 week in the future."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(weeks=1, days=3)
        assert format_timestamp(dt) == "in 1 week"

    def test_old_date_shows_absolute(self) -> None:
        """Dates more than 30 days ago show absolute date."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(days=45)
        result = format_timestamp(dt)
        # Should be in YYYY-MM-DD format
        assert result == dt.strftime("%Y-%m-%d")

    def test_far_future_date_shows_absolute(self) -> None:
        """Dates more than 30 days in future show absolute date."""
        now = datetime.now(timezone.utc)
        dt = now + timedelta(days=45)
        result = format_timestamp(dt)
        # Should be in YYYY-MM-DD format
        assert result == dt.strftime("%Y-%m-%d")

    def test_iso_string_input(self) -> None:
        """ISO format strings are parsed correctly."""
        # Create a time 5 minutes ago
        now = datetime.now(timezone.utc)
        dt = now - timedelta(minutes=5)
        iso_str = dt.isoformat()
        result = format_timestamp(iso_str)
        assert "minute" in result

    def test_iso_string_with_z_suffix(self) -> None:
        """ISO strings with Z suffix are parsed correctly."""
        now = datetime.now(timezone.utc)
        dt = now - timedelta(hours=2)
        # Replace +00:00 with Z
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        result = format_timestamp(iso_str)
        assert "hour" in result

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetimes (no timezone) are treated as UTC."""
        now = datetime.now(timezone.utc)
        # Create naive datetime
        naive_dt = (now - timedelta(minutes=10)).replace(tzinfo=None)
        result = format_timestamp(naive_dt)
        assert "minute" in result

    def test_non_utc_timezone(self) -> None:
        """Datetimes with non-UTC timezone are converted properly."""
        # Create a timezone +5 hours from UTC
        tz_plus5 = timezone(timedelta(hours=5))
        now_utc = datetime.now(timezone.utc)
        # Create time 2 hours ago in UTC, but expressed in +5 timezone
        dt = (now_utc - timedelta(hours=2)).astimezone(tz_plus5)
        result = format_timestamp(dt)
        assert "hour" in result

    def test_invalid_string_returns_string(self) -> None:
        """Invalid datetime strings are returned as-is."""
        result = format_timestamp("not a date")
        assert result == "not a date"

    def test_non_datetime_object_returns_string(self) -> None:
        """Non-datetime objects are converted to string."""
        result = format_timestamp(12345)  # type: ignore
        assert result == "12345"


class TestFormatSize:
    """Tests for the format_size function."""

    def test_bytes(self) -> None:
        assert format_size(500) == "500 B"

    def test_kilobytes(self) -> None:
        assert format_size(1536) == "1.5 KB"

    def test_megabytes(self) -> None:
        assert format_size(1572864) == "1.5 MB"

    def test_gigabytes(self) -> None:
        assert format_size(1610612736) == "1.5 GB"

    def test_terabytes(self) -> None:
        assert format_size(1649267441664) == "1.5 TB"

    def test_zero_bytes(self) -> None:
        assert format_size(0) == "0 B"

    def test_negative_returns_na(self) -> None:
        assert format_size(-1) == "N/A"


class TestFormatBool:
    """Tests for the format_bool function."""

    def test_true(self) -> None:
        assert format_bool(True) == "Yes"

    def test_false(self) -> None:
        assert format_bool(False) == "No"
