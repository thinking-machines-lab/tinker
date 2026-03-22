"""Tests for ServiceClient create_training_client_from_state method."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import httpx
import pytest
from respx import MockRouter

import tinker
from tinker import types
from tinker.lib.internal_client_holder import InternalClientHolder
from tinker.lib.public_interfaces.training_client import (
    TrainingClient,
    _find_matching_checkpoint_path,
    _matching_checkpoint_ids,
)

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


@pytest.mark.respx(base_url=base_url)
def test_service_client_passes_project_id_on_session_create(respx_mock: MockRouter) -> None:
    create_session_route = respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "test-session-id"})
    )

    service_client = tinker.ServiceClient(base_url=base_url, project_id="project-123")
    service_client.holder.close()

    assert create_session_route.called
    sent_payload = json.loads(create_session_route.calls[0].request.content.decode())
    assert sent_payload["project_id"] == "project-123"


@pytest.mark.respx(base_url=base_url)
async def test_create_training_client_from_state_async(respx_mock: MockRouter) -> None:
    """Test create_training_client_from_state_async uses public endpoint."""
    tinker_path = "tinker://test-model-123/weights/checkpoint-001"
    weights_info_response = types.WeightsInfoResponse(
        base_model="meta-llama/Llama-3.2-1B", is_lora=True, lora_rank=32
    )

    # Mock the get_weights_info endpoint call
    respx_mock.post("/api/v1/weights_info").mock(
        return_value=httpx.Response(200, json=weights_info_response.model_dump())
    )

    # Mock the create model call
    respx_mock.post("/api/v1/models").mock(
        return_value=httpx.Response(200, json={"model_id": "new-model-id"})
    )

    # Mock the load state call
    respx_mock.post("/api/v1/load_weights").mock(return_value=httpx.Response(200, json={}))

    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_training_client_from_state_async(tinker_path)

    assert training_client is not None
    assert training_client.model_id == "new-model-id"


@pytest.mark.respx(base_url=base_url)
async def test_create_training_client_from_state_async_with_user_metadata(
    respx_mock: MockRouter,
) -> None:
    """Test create_training_client_from_state_async preserves user metadata."""
    tinker_path = "tinker://test-model-123/weights/checkpoint-001"
    user_metadata = {"key1": "value1", "key2": "value2"}
    weights_info_response = types.WeightsInfoResponse(
        base_model="meta-llama/Llama-3.2-1B", is_lora=True, lora_rank=32
    )

    # Mock the get_weights_info endpoint call
    respx_mock.post("/api/v1/weights_info").mock(
        return_value=httpx.Response(200, json=weights_info_response.model_dump())
    )

    # Mock the create model call
    respx_mock.post("/api/v1/models").mock(
        return_value=httpx.Response(200, json={"model_id": "new-model-id"})
    )

    # Mock the load state call
    respx_mock.post("/api/v1/load_weights").mock(return_value=httpx.Response(200, json={}))

    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_training_client_from_state_async(
        tinker_path, user_metadata=user_metadata
    )

    assert training_client is not None
    # Verify user_metadata was passed through (we can't directly check it, but the call succeeded)


@pytest.mark.respx(base_url=base_url)
async def test_create_training_client_from_state_async_not_lora(respx_mock: MockRouter) -> None:
    """Test create_training_client_from_state_async raises assertion for non-LoRA model."""
    tinker_path = "tinker://test-model-123/weights/checkpoint-001"

    # Mock WeightsInfo response with is_lora=False
    weights_info_response = types.WeightsInfoResponse(
        base_model="meta-llama/Llama-3.2-1B", is_lora=False, lora_rank=None
    )

    # Mock the get_weights_info endpoint call
    respx_mock.post("/api/v1/weights_info").mock(
        return_value=httpx.Response(200, json=weights_info_response.model_dump())
    )

    service_client = tinker.ServiceClient(base_url=base_url)

    # Should raise AssertionError because is_lora=False or lora_rank=None
    with pytest.raises(AssertionError):
        await service_client.create_training_client_from_state_async(tinker_path)


@pytest.mark.respx(base_url=base_url)
async def test_create_training_client_from_state_async_uses_public_endpoint(
    respx_mock: MockRouter,
) -> None:
    """Test that create_training_client_from_state_async uses get_weights_info_by_tinker_path."""
    tinker_path = "tinker://test-model-123/weights/checkpoint-001"

    # Mock WeightsInfo response
    weights_info_response = types.WeightsInfoResponse(
        base_model="meta-llama/Llama-3.2-1B", is_lora=True, lora_rank=32
    )

    # Mock the get_weights_info endpoint call (public endpoint)
    info_lite_route = respx_mock.post("/api/v1/weights_info").mock(
        return_value=httpx.Response(200, json=weights_info_response.model_dump())
    )

    # Mock the create model call
    respx_mock.post("/api/v1/models").mock(
        return_value=httpx.Response(200, json={"model_id": "new-model-id"})
    )

    # Mock the load state call
    respx_mock.post("/api/v1/load_weights").mock(return_value=httpx.Response(200, json={}))

    service_client = tinker.ServiceClient(base_url=base_url)
    await service_client.create_training_client_from_state_async(tinker_path)

    # Verify it uses the public endpoint (info_lite), not the full training run endpoint
    assert info_lite_route.called


@pytest.mark.respx(base_url=base_url)
def test_create_training_client_from_state_sync(respx_mock: MockRouter) -> None:
    """Test create_training_client_from_state (sync) uses public endpoint."""
    tinker_path = "tinker://test-model-123/weights/checkpoint-001"
    weights_info_response = types.WeightsInfoResponse(
        base_model="meta-llama/Llama-3.2-1B", is_lora=True, lora_rank=32
    )

    # Mock the get_weights_info endpoint call
    respx_mock.post("/api/v1/weights_info").mock(
        return_value=httpx.Response(200, json=weights_info_response.model_dump())
    )

    # Mock the create model call
    respx_mock.post("/api/v1/models").mock(
        return_value=httpx.Response(200, json={"model_id": "new-model-id"})
    )

    # Mock the load state call
    respx_mock.post("/api/v1/load_weights").mock(return_value=httpx.Response(200, json={}))

    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = service_client.create_training_client_from_state(tinker_path)

    assert training_client is not None
    assert training_client.model_id == "new-model-id"


@pytest.mark.respx(base_url=base_url)
def test_create_training_client_from_state_sync_uses_public_endpoint(
    respx_mock: MockRouter,
) -> None:
    """Test that create_training_client_from_state (sync) uses get_weights_info_by_tinker_path."""
    tinker_path = "tinker://test-model-123/weights/checkpoint-001"

    # Mock WeightsInfo response
    weights_info_response = types.WeightsInfoResponse(
        base_model="meta-llama/Llama-3.2-1B", is_lora=True, lora_rank=32
    )

    # Mock the get_weights_info endpoint call (public endpoint)
    info_lite_route = respx_mock.post("/api/v1/weights_info").mock(
        return_value=httpx.Response(200, json=weights_info_response.model_dump())
    )

    # Mock the create model call
    respx_mock.post("/api/v1/models").mock(
        return_value=httpx.Response(200, json={"model_id": "new-model-id"})
    )

    # Mock the load state call
    respx_mock.post("/api/v1/load_weights").mock(return_value=httpx.Response(200, json={}))

    service_client = tinker.ServiceClient(base_url=base_url)
    service_client.create_training_client_from_state(tinker_path)

    # Verify it uses the public endpoint (info_lite), not the full training run endpoint
    assert info_lite_route.called


def test_matching_checkpoint_ids_supports_bare_and_prefixed_names() -> None:
    assert _matching_checkpoint_ids("000010", "training") == {"000010", "weights/000010"}
    assert _matching_checkpoint_ids("000010", "sampler") == {
        "000010",
        "sampler_weights/000010",
    }


def test_find_matching_checkpoint_path_filters_by_type_and_name() -> None:
    checkpoints = [
        types.Checkpoint(
            checkpoint_id="weights/000010",
            checkpoint_type="training",
            time=datetime.now(UTC),
            tinker_path="tinker://run-1/weights/000010",
        ),
        types.Checkpoint(
            checkpoint_id="sampler_weights/000010",
            checkpoint_type="sampler",
            time=datetime.now(UTC),
            tinker_path="tinker://run-1/sampler_weights/000010",
        ),
    ]

    assert (
        _find_matching_checkpoint_path(checkpoints, "000010", "training")
        == "tinker://run-1/weights/000010"
    )
    assert (
        _find_matching_checkpoint_path(checkpoints, "000010", "sampler")
        == "tinker://run-1/sampler_weights/000010"
    )


def test_retryable_status_codes_do_not_include_409() -> None:
    assert InternalClientHolder._is_retryable_status_code(408)
    assert not InternalClientHolder._is_retryable_status_code(409)
    assert InternalClientHolder._is_retryable_status_code(429)


class _DummyHolder:
    def __init__(self, checkpoints: list[types.Checkpoint]):
        self._response = types.CheckpointsListResponse(checkpoints=checkpoints)

    async def execute_with_retries(self, _func):
        return self._response


@pytest.mark.asyncio
async def test_recover_checkpoint_path_from_conflict_returns_matching_path() -> None:
    client = TrainingClient.__new__(TrainingClient)
    client.holder = _DummyHolder(
        checkpoints=[
            types.Checkpoint(
                checkpoint_id="weights/000010",
                checkpoint_type="training",
                time=datetime.now(UTC),
                tinker_path="tinker://run-1/weights/000010",
            )
        ]
    )
    client._guaranteed_model_id = lambda: "run-1"  # type: ignore[method-assign]

    recovered = await client._recover_checkpoint_path_from_conflict(
        checkpoint_name="000010",
        checkpoint_type="training",
    )
    assert recovered == "tinker://run-1/weights/000010"


@pytest.mark.asyncio
async def test_recover_checkpoint_path_from_conflict_returns_none_when_missing() -> None:
    client = TrainingClient.__new__(TrainingClient)
    client.holder = _DummyHolder(
        checkpoints=[
            types.Checkpoint(
                checkpoint_id="weights/000011",
                checkpoint_type="training",
                time=datetime.now(UTC),
                tinker_path="tinker://run-1/weights/000011",
            )
        ]
    )
    client._guaranteed_model_id = lambda: "run-1"  # type: ignore[method-assign]

    recovered = await client._recover_checkpoint_path_from_conflict(
        checkpoint_name="000010",
        checkpoint_type="training",
    )
    assert recovered is None
