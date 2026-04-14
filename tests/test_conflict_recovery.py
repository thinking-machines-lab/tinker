"""Tests for 409 ConflictError recovery in checkpoint save operations."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest

from tinker._exceptions import ConflictError
from tinker.lib.public_interfaces.training_client import TrainingClient


def _make_conflict_error() -> ConflictError:
    """Create a ConflictError for testing."""
    request = httpx.Request("POST", "http://test/api/v1/save_weights")
    response = httpx.Response(409, request=request)
    return ConflictError("conflict", response=response, body=None)


def _make_mock_holder() -> Mock:
    """Create a mock InternalClientHolder whose weights.save() raises ConflictError."""
    mock_client = MagicMock()
    mock_client.weights.save = AsyncMock(side_effect=_make_conflict_error())
    mock_client.weights.save_for_sampler = AsyncMock(side_effect=_make_conflict_error())

    @contextmanager
    def fake_aclient(*args: Any, **kwargs: Any):
        yield mock_client

    async def fake_execute_with_retries(fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    holder = Mock()
    holder.aclient = fake_aclient
    holder.get_telemetry = Mock(return_value=None)
    holder.execute_with_retries = fake_execute_with_retries
    holder.get_loop = Mock(side_effect=lambda: asyncio.get_event_loop())

    def fake_run_coroutine_threadsafe(coro: Any) -> Any:
        return asyncio.ensure_future(coro)

    holder.run_coroutine_threadsafe = fake_run_coroutine_threadsafe
    return holder


@pytest.mark.asyncio
async def test_save_state_returns_synthetic_path_on_conflict() -> None:
    """save_state catches 409 and returns SaveWeightsResponse with synthetic path."""
    holder = _make_mock_holder()
    client = TrainingClient(holder, model_seq_id=0, model_id="model-123")

    result = await client.save_state("ckpt-001")
    assert result.path == "tinker://model-123/weights/ckpt-001"


@pytest.mark.asyncio
async def test_save_weights_for_sampler_returns_synthetic_path_on_conflict() -> None:
    """save_weights_for_sampler catches 409 and returns response with synthetic path."""
    holder = _make_mock_holder()
    holder._sampling_client_counter = 0
    client = TrainingClient(holder, model_seq_id=0, model_id="model-789")

    result = await client.save_weights_for_sampler("ckpt-001")
    assert result.path == "tinker://model-789/sampler_weights/ckpt-001"


@pytest.mark.asyncio
async def test_save_weights_for_sampler_unnamed_reraises_conflict() -> None:
    """409 on unnamed sampler save (name=None) should re-raise, not swallow."""
    holder = _make_mock_holder()
    holder._sampling_client_counter = 0
    client = TrainingClient(holder, model_seq_id=0, model_id="model-000")

    with pytest.raises(ConflictError):
        await client.save_weights_for_sampler(None)
