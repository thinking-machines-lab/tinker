"""Tests for ServiceClient."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Coroutine

import pytest

if TYPE_CHECKING:
    from tinker.lib.public_interfaces.service_client import ServiceClient


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

    sc = ServiceClient(base_url="http://127.0.0.1:4010", _strict_response_validation=True)
    _ = sc.holder  # holders are lazy; force creation

    assert captured_kwargs["_strict_response_validation"] is True


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

    rest_client = sc.create_rest_client()
    assert len(calls) == 1
    assert calls[0].get("_skip_session") is True

    session_holder = sc.holder
    assert len(calls) == 2
    assert "_skip_session" not in calls[1]

    # Both holders are cached.
    assert sc.create_rest_client().holder is rest_client.holder
    assert sc.holder is session_holder
    assert len(calls) == 2


class _ResultFuture:
    """Stand-in for the SDK futures; only .result() is exercised in these tests."""

    def __init__(self, value: object) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> object:
        return self._value


def _returns(value: str) -> Callable[[], Coroutine[object, object, str]]:
    """Build a zero-arg coroutine function that resolves to `value`."""

    async def _get() -> str:
        return value

    return _get


def _run_to_result(coro: Coroutine[object, object, object]) -> _ResultFuture:
    """Fake holder.run_coroutine_threadsafe: drain the coroutine synchronously."""
    return _ResultFuture(asyncio.run(coro))


def _record(sink: list[str | None], token: str | None) -> _ResultFuture:
    """Fake training_client.load_state: capture the token it was handed."""
    sink.append(token)
    return _ResultFuture(None)


_WEIGHTS_INFO = SimpleNamespace(
    base_model="Qwen/Qwen3-8B",
    lora_rank=32,
    train_unembed=True,
    train_mlp=True,
    train_attn=True,
)


def _build_service_client(
    monkeypatch: pytest.MonkeyPatch, source_rest: object, training_client: object
) -> ServiceClient:
    from tinker.lib.public_interfaces.service_client import ServiceClient

    sc = ServiceClient.__new__(ServiceClient)
    monkeypatch.setattr(sc, "_get_rest_client_for_weights", lambda token: source_rest)
    monkeypatch.setattr(sc, "create_lora_training_client", lambda **kwargs: training_client)
    return sc


def test_create_training_client_from_state_exchanges_source_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source access token must be exchanged for a JWT before load_state, not forwarded raw."""
    seen: list[str | None] = []
    source_rest = SimpleNamespace(
        get_weights_info_by_tinker_path=lambda path: _ResultFuture(_WEIGHTS_INFO),
        holder=SimpleNamespace(
            run_coroutine_threadsafe=_run_to_result,
            _default_auth=SimpleNamespace(get_token=_returns("exchanged.jwt")),
        ),
    )
    training_client = SimpleNamespace(
        load_state=lambda path, weights_access_token: _record(seen, weights_access_token)
    )
    sc = _build_service_client(monkeypatch, source_rest, training_client)

    result = sc.create_training_client_from_state(
        "tinker://run-id/weights/ckpt", weights_access_token="raw-source-credential"
    )

    assert result is training_client
    assert seen == ["exchanged.jwt"]


def test_create_training_client_from_state_without_token_skips_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a source token, load_state gets None and no token exchange is attempted."""
    seen: list[str | None] = []
    source_rest = SimpleNamespace(
        get_weights_info_by_tinker_path=lambda path: _ResultFuture(_WEIGHTS_INFO),
        # get_token is absent: if the code tried to exchange, this would AttributeError.
        holder=SimpleNamespace(_default_auth=SimpleNamespace()),
    )
    training_client = SimpleNamespace(
        load_state=lambda path, weights_access_token: _record(seen, weights_access_token)
    )
    sc = _build_service_client(monkeypatch, source_rest, training_client)

    sc.create_training_client_from_state("tinker://run-id/weights/ckpt")

    assert seen == [None]
