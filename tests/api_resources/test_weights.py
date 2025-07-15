# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tinker import Tinker, AsyncTinker
from tests.utils import assert_matches_type
from tinker.types.shared import UntypedAPIFuture

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestWeights:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_load(self, client: Tinker) -> None:
        weight = client.weights.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_load_with_all_params(self, client: Tinker) -> None:
        weight = client.weights.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
            type="load_weights",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_load(self, client: Tinker) -> None:
        response = client.weights.with_raw_response.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        weight = response.parse()
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_load(self, client: Tinker) -> None:
        with client.weights.with_streaming_response.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            weight = response.parse()
            assert_matches_type(UntypedAPIFuture, weight, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_save(self, client: Tinker) -> None:
        weight = client.weights.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_save_with_all_params(self, client: Tinker) -> None:
        weight = client.weights.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="For the tinker path 'tinker://model-id/weights/step-0123', the WeightsLabel would be 'step-0123'",
            type="save_weights",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_save(self, client: Tinker) -> None:
        response = client.weights.with_raw_response.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        weight = response.parse()
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_save(self, client: Tinker) -> None:
        with client.weights.with_streaming_response.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            weight = response.parse()
            assert_matches_type(UntypedAPIFuture, weight, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_save_for_sampler(self, client: Tinker) -> None:
        weight = client.weights.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_save_for_sampler_with_all_params(self, client: Tinker) -> None:
        weight = client.weights.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="For the tinker path 'tinker://model-id/weights/step-0123', the WeightsLabel would be 'step-0123'",
            type="save_weights_for_sampler",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_save_for_sampler(self, client: Tinker) -> None:
        response = client.weights.with_raw_response.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        weight = response.parse()
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_save_for_sampler(self, client: Tinker) -> None:
        with client.weights.with_streaming_response.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            weight = response.parse()
            assert_matches_type(UntypedAPIFuture, weight, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncWeights:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_load(self, async_client: AsyncTinker) -> None:
        weight = await async_client.weights.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_load_with_all_params(self, async_client: AsyncTinker) -> None:
        weight = await async_client.weights.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
            type="load_weights",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_load(self, async_client: AsyncTinker) -> None:
        response = await async_client.weights.with_raw_response.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        weight = await response.parse()
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_load(self, async_client: AsyncTinker) -> None:
        async with async_client.weights.with_streaming_response.load(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="tinker://model-id/weights/step-0123",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            weight = await response.parse()
            assert_matches_type(UntypedAPIFuture, weight, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_save(self, async_client: AsyncTinker) -> None:
        weight = await async_client.weights.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_save_with_all_params(self, async_client: AsyncTinker) -> None:
        weight = await async_client.weights.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="For the tinker path 'tinker://model-id/weights/step-0123', the WeightsLabel would be 'step-0123'",
            type="save_weights",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_save(self, async_client: AsyncTinker) -> None:
        response = await async_client.weights.with_raw_response.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        weight = await response.parse()
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_save(self, async_client: AsyncTinker) -> None:
        async with async_client.weights.with_streaming_response.save(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            weight = await response.parse()
            assert_matches_type(UntypedAPIFuture, weight, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_save_for_sampler(self, async_client: AsyncTinker) -> None:
        weight = await async_client.weights.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_save_for_sampler_with_all_params(self, async_client: AsyncTinker) -> None:
        weight = await async_client.weights.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            path="For the tinker path 'tinker://model-id/weights/step-0123', the WeightsLabel would be 'step-0123'",
            type="save_weights_for_sampler",
        )
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_save_for_sampler(self, async_client: AsyncTinker) -> None:
        response = await async_client.weights.with_raw_response.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        weight = await response.parse()
        assert_matches_type(UntypedAPIFuture, weight, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_save_for_sampler(self, async_client: AsyncTinker) -> None:
        async with async_client.weights.with_streaming_response.save_for_sampler(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            weight = await response.parse()
            assert_matches_type(UntypedAPIFuture, weight, path=["response"])

        assert cast(Any, response.is_closed) is True
