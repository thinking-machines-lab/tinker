"""Regression: default pyqwest transport must trust system CA certs (#51).

pyqwest 0.7.0 introduced tls_include_system_certs=False by default, which made
ServiceClient() fail with UnknownIssuer against public endpoints. The default
transport constructor must pass tls_include_system_certs=True when supported.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def test_default_pyqwest_transport_passes_system_certs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeHTTPTransport:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

    class FakeAsyncPyqwestTransport:
        def __init__(self, transport: Any = None) -> None:
            self.transport = transport

    fake_pyqwest = MagicMock()
    fake_pyqwest.HTTPTransport = FakeHTTPTransport
    fake_httpx = MagicMock()
    fake_httpx.AsyncPyqwestTransport = FakeAsyncPyqwestTransport

    import sys

    monkeypatch.setitem(sys.modules, "pyqwest", fake_pyqwest)
    monkeypatch.setitem(sys.modules, "pyqwest.httpx", fake_httpx)

    # Force re-import path: call the helper after stubbing modules.
    from tinker._base_client import _default_pyqwest_transport

    transport = _default_pyqwest_transport()
    assert isinstance(transport, FakeAsyncPyqwestTransport)
    assert captured["kwargs"].get("tls_include_system_certs") is True


def test_default_pyqwest_transport_falls_back_without_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older pyqwest rejects the kwarg with TypeError; still construct a transport."""
    calls: list[dict[str, Any]] = []

    class FakeHTTPTransport:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            if "tls_include_system_certs" in kwargs:
                raise TypeError("unexpected keyword argument")

    class FakeAsyncPyqwestTransport:
        def __init__(self, transport: Any = None) -> None:
            self.transport = transport

    fake_pyqwest = MagicMock()
    fake_pyqwest.HTTPTransport = FakeHTTPTransport
    fake_httpx = MagicMock()
    fake_httpx.AsyncPyqwestTransport = FakeAsyncPyqwestTransport

    import sys

    monkeypatch.setitem(sys.modules, "pyqwest", fake_pyqwest)
    monkeypatch.setitem(sys.modules, "pyqwest.httpx", fake_httpx)

    from tinker._base_client import _default_pyqwest_transport

    transport = _default_pyqwest_transport()
    assert isinstance(transport, FakeAsyncPyqwestTransport)
    assert calls[0] == {"tls_include_system_certs": True}
    assert calls[1] == {}  # fallback no-arg construction
