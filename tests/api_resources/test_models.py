# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tinker import Tinker, AsyncTinker
from tests.utils import assert_matches_type
from tinker.types import (
    GetInfoResponse,
)
from tinker.types.shared import UntypedAPIFuture

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestModels:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_create(self, client: Tinker) -> None:
        model = client.models.create(
            base_model="meta-llama/Llama-3.2-1B",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_create_with_all_params(self, client: Tinker) -> None:
        model = client.models.create(
            base_model="meta-llama/Llama-3.2-1B",
            lora_config={
                "rank": 16,
                "seed": 42,
            },
            type="create_model",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_create(self, client: Tinker) -> None:
        response = client.models.with_raw_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_create(self, client: Tinker) -> None:
        with client.models.with_streaming_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(UntypedAPIFuture, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_get_info(self, client: Tinker) -> None:
        model = client.models.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(GetInfoResponse, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_get_info_with_all_params(self, client: Tinker) -> None:
        model = client.models.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="get_info",
        )
        assert_matches_type(GetInfoResponse, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_get_info(self, client: Tinker) -> None:
        response = client.models.with_raw_response.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(GetInfoResponse, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_get_info(self, client: Tinker) -> None:
        with client.models.with_streaming_response.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(GetInfoResponse, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_unload(self, client: Tinker) -> None:
        model = client.models.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_unload_with_all_params(self, client: Tinker) -> None:
        model = client.models.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="unload_model",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_unload(self, client: Tinker) -> None:
        response = client.models.with_raw_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = response.parse()
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_unload(self, client: Tinker) -> None:
        with client.models.with_streaming_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = response.parse()
            assert_matches_type(UntypedAPIFuture, model, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncModels:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_create(self, async_client: AsyncTinker) -> None:
        model = await async_client.models.create(
            base_model="meta-llama/Llama-3.2-1B",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncTinker) -> None:
        model = await async_client.models.create(
            base_model="meta-llama/Llama-3.2-1B",
            lora_config={
                "rank": 16,
                "seed": 42,
            },
            type="create_model",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncTinker) -> None:
        response = await async_client.models.with_raw_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncTinker) -> None:
        async with async_client.models.with_streaming_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(UntypedAPIFuture, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_get_info(self, async_client: AsyncTinker) -> None:
        model = await async_client.models.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(GetInfoResponse, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_get_info_with_all_params(self, async_client: AsyncTinker) -> None:
        model = await async_client.models.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="get_info",
        )
        assert_matches_type(GetInfoResponse, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_get_info(self, async_client: AsyncTinker) -> None:
        response = await async_client.models.with_raw_response.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(GetInfoResponse, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_get_info(self, async_client: AsyncTinker) -> None:
        async with async_client.models.with_streaming_response.get_info(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(GetInfoResponse, model, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_unload(self, async_client: AsyncTinker) -> None:
        model = await async_client.models.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_unload_with_all_params(self, async_client: AsyncTinker) -> None:
        model = await async_client.models.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="unload_model",
        )
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_unload(self, async_client: AsyncTinker) -> None:
        response = await async_client.models.with_raw_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        model = await response.parse()
        assert_matches_type(UntypedAPIFuture, model, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_unload(self, async_client: AsyncTinker) -> None:
        async with async_client.models.with_streaming_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            model = await response.parse()
            assert_matches_type(UntypedAPIFuture, model, path=["response"])

        assert cast(Any, response.is_closed) is True
