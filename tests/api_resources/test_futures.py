# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tinker import Tinker, AsyncTinker
from tests.utils import assert_matches_type
from tinker.types import FutureRetrieveResponse

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestFutures:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_retrieve(self, client: Tinker) -> None:
        future = client.futures.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(FutureRetrieveResponse, future, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_retrieve_with_all_params(self, client: Tinker) -> None:
        future = client.futures.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(FutureRetrieveResponse, future, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_retrieve(self, client: Tinker) -> None:
        response = client.futures.with_raw_response.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        future = response.parse()
        assert_matches_type(FutureRetrieveResponse, future, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_retrieve(self, client: Tinker) -> None:
        with client.futures.with_streaming_response.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            future = response.parse()
            assert_matches_type(FutureRetrieveResponse, future, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncFutures:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_retrieve(self, async_client: AsyncTinker) -> None:
        future = await async_client.futures.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(FutureRetrieveResponse, future, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_retrieve_with_all_params(self, async_client: AsyncTinker) -> None:
        future = await async_client.futures.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(FutureRetrieveResponse, future, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_retrieve(self, async_client: AsyncTinker) -> None:
        response = await async_client.futures.with_raw_response.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        future = await response.parse()
        assert_matches_type(FutureRetrieveResponse, future, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_retrieve(self, async_client: AsyncTinker) -> None:
        async with async_client.futures.with_streaming_response.retrieve(
            request_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            future = await response.parse()
            assert_matches_type(FutureRetrieveResponse, future, path=["response"])

        assert cast(Any, response.is_closed) is True
