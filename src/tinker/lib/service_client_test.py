"""Tests for ServiceClient."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _capture_holder_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from tinker.lib.public_interfaces import service_client as service_client_module

    captured_kwargs: dict[str, object] = {}

    def fake_holder(**kwargs: object) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(_session_id="test-session-id")

    monkeypatch.setattr(service_client_module, "InternalClientHolder", fake_holder)
    return captured_kwargs


def test_service_client_ignores_unsupported_kwargs_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tinker.lib.public_interfaces.service_client import ServiceClient

    captured_kwargs = _capture_holder_kwargs(monkeypatch)

    with pytest.warns(FutureWarning, match="ignores unsupported arguments: max_retries, timeout"):
        sc = ServiceClient(base_url="http://127.0.0.1:4010", timeout=5.0, max_retries=0)
    _ = sc.holder  # holders are lazy; force creation

    assert captured_kwargs == {
        "user_metadata": None,
        "project_id": None,
        "display_name": None,
        "base_url": "http://127.0.0.1:4010",
        "api_key": None,
        "default_headers_overrides": {},
    }


def test_holders_created_lazily_and_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """No holder at construction; REST gets a session-less holder, training a sessionful one."""
    from tinker.lib.public_interfaces import service_client as service_client_module
    from tinker.lib.public_interfaces.service_client import ServiceClient

    calls: list[dict[str, object]] = []

    def fake_holder(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(_session_id="test-session-id")

    monkeypatch.setattr(service_client_module, "InternalClientHolder", fake_holder)

    sc = ServiceClient(base_url="http://127.0.0.1:4010")
    assert calls == []

    rest_holder = sc._get_rest_holder()
    assert len(calls) == 1
    assert calls[0].get("_skip_session") is True

    session_holder = sc.holder
    assert len(calls) == 2
    assert "_skip_session" not in calls[1]

    # Both holders are cached, and the deprecated factory still wraps the REST one.
    assert sc._get_rest_holder() is rest_holder
    assert sc.holder is session_holder
    with pytest.warns(FutureWarning, match="Construct tinker.RestClient\\(\\) directly"):
        assert sc.create_rest_client().holder is rest_holder
    assert len(calls) == 2
