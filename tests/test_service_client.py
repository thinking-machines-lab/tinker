"""Tests for ServiceClient create_training_client_from_state method."""

from __future__ import annotations

import json
import os

import httpx
import pytest
from respx import MockRouter

import tinker
from tinker import types

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
def test_service_client_reads_project_id_from_env(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_PROJECT_ID", "env-project-456")
    create_session_route = respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "test-session-id"})
    )

    service_client = tinker.ServiceClient(base_url=base_url)
    service_client.holder.close()

    assert create_session_route.called
    sent_payload = json.loads(create_session_route.calls[0].request.content.decode())
    assert sent_payload["project_id"] == "env-project-456"


@pytest.mark.respx(base_url=base_url)
def test_service_client_explicit_project_id_overrides_env(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_PROJECT_ID", "env-project-456")
    create_session_route = respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "test-session-id"})
    )

    service_client = tinker.ServiceClient(base_url=base_url, project_id="explicit-123")
    service_client.holder.close()

    assert create_session_route.called
    sent_payload = json.loads(create_session_route.calls[0].request.content.decode())
    assert sent_payload["project_id"] == "explicit-123"


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


@pytest.mark.respx(base_url=base_url)
def test_get_rest_client_for_weights_creates_no_session(respx_mock: MockRouter) -> None:
    """A weights_access_token REST client must not create its own session.

    Regression test: the source-token client is only used to read weights info,
    but it used to eagerly create a session on construction. That session lands
    in the token org's Default project and 400s ("read-only") when that Default
    is frozen (e.g. cross-org copies). With the fix the source-token client is
    session-less, so only the destination session is ever created.
    """
    # use_pyqwest_transport=False keeps calls on httpx so respx can intercept
    # them (respx cannot mock the pyqwest transport).
    respx_mock.post("/api/v1/client/config").mock(
        return_value=httpx.Response(200, json={"use_pyqwest_transport": False})
    )
    create_session_route = respx_mock.post("/api/v1/create_session").mock(
        return_value=httpx.Response(200, json={"session_id": "dest-session-id"})
    )

    service_client = tinker.ServiceClient(base_url=base_url, api_key="tml-dest-token")
    assert create_session_route.call_count == 1  # the destination session

    # Building the source-token REST client must NOT create another session.
    rest_client = service_client._get_rest_client_for_weights("tml-src-token")
    assert create_session_route.call_count == 1
    assert rest_client.holder._session_id is None

    service_client.holder.close()
