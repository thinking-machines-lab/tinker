"""Tests for ServiceClient."""

from __future__ import annotations

import pytest


def test_service_client_accepts_existing_strict_response_validation_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ServiceClient may be reconstructed from InternalClientHolder kwargs."""
    from tinker.lib.public_interfaces import service_client as service_client_module
    from tinker.lib.public_interfaces.service_client import ServiceClient

    captured_kwargs: dict[str, object] = {}

    class Holder:
        _session_id = "test-session-id"

    def fake_holder(**kwargs: object) -> Holder:
        captured_kwargs.update(kwargs)
        return Holder()

    monkeypatch.setattr(service_client_module, "InternalClientHolder", fake_holder)

    ServiceClient(base_url="http://127.0.0.1:4010", _strict_response_validation=True)

    assert captured_kwargs["_strict_response_validation"] is True
