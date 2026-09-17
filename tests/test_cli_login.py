"""Tests for the helpers used by `tinker auth login`."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from tinker.cli import login as login_module


class TestApiKeyName:
    # Catches keys showing up in the console under an unrecognizable name.
    def test_names_the_key_after_the_machine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.socket, "gethostname", lambda: "laptop.local")
        assert login_module.api_key_name() == "tinker-cli-laptop"

    # Catches a machine with no hostname yielding a trailing-dash name.
    def test_falls_back_when_the_hostname_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.socket, "gethostname", lambda: "")
        assert login_module.api_key_name() == "tinker-cli-unknown"


def test_masked_input_replaces_pasted_characters_with_asterisks() -> None:
    entered = iter(("tml-", "secret", "\n"))

    with CliRunner().isolation() as streams:
        key = login_module._read_masked_input(lambda: next(entered))

    assert key == "tml-secret"
    assert streams[0].getvalue().decode() == "Paste your API key: **********"
