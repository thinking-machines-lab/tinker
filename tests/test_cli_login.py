"""Tests for the key name and browser opener `tinker auth login` uses."""

from __future__ import annotations

import threading

import pytest

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


class TestOpenUrl:
    """Opening the browser is best-effort and must never block the login."""

    @staticmethod
    def _record_open(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], threading.Event]:
        opened: list[str] = []
        done = threading.Event()

        def fake_open(url: str) -> bool:
            opened.append(url)
            done.set()
            return True

        monkeypatch.setattr(login_module.webbrowser, "open", fake_open)
        return opened, done

    # Catches webbrowser's headless-Linux fallback: without a graphical
    # session it launches a console browser (lynx) that takes over the very
    # terminal the login is running in.
    def test_no_browser_without_a_graphical_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "linux")
        for name in ("BROWSER", "DISPLAY", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        opened, _ = self._record_open(monkeypatch)

        assert login_module.open_url("https://tinker.test/keys") is False
        assert opened == []

    # Catches the headless guard being over-tightened so that Linux desktops
    # (which do have a display) stop getting a browser opened for them.
    def test_opens_the_url_with_a_display(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        opened, done = self._record_open(monkeypatch)

        assert login_module.open_url("https://tinker.test/keys") is True
        assert done.wait(timeout=5)
        assert opened == ["https://tinker.test/keys"]

    # Catches open_url calling webbrowser synchronously: a $BROWSER command
    # runs in the foreground, which would stall the login (it still has to
    # prompt for the key) for as long as that browser stays open.
    def test_returns_without_waiting_for_a_blocking_browser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "darwin")
        release = threading.Event()
        monkeypatch.setattr(
            login_module.webbrowser, "open", lambda url: bool(release.wait(timeout=10))
        )
        try:
            assert login_module.open_url("https://tinker.test/keys") is True
        finally:
            release.set()

    # Catches a browser that fails to start (webbrowser.Error/OSError on the
    # background thread) spewing a thread-crash traceback into the terminal
    # in the middle of a login that can proceed fine with the printed URL.
    def test_browser_errors_are_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(login_module.sys, "platform", "darwin")
        raised = threading.Event()

        def boom(url: str) -> bool:
            raised.set()
            raise login_module.webbrowser.Error("no browser")

        monkeypatch.setattr(login_module.webbrowser, "open", boom)
        assert login_module.open_url("https://tinker.test/keys") is True
        assert raised.wait(timeout=5)
