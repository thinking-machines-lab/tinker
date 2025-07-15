# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tinker import Tinker, AsyncTinker
from tests.utils import assert_matches_type
from tinker.types import SampleResponse
from tinker.types.shared import UntypedAPIFuture

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestSampling:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_asample(self, client: Tinker) -> None:
        sampling = client.sampling.asample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )
        assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_asample_with_all_params(self, client: Tinker) -> None:
        sampling = client.sampling.asample(
            num_samples=1,
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={
                "max_tokens": 0,
                "seed": 0,
                "stop": "string",
                "temperature": 0,
                "top_k": 0,
                "top_p": 0,
            },
            base_model="base_model",
            model_path="model_path",
            prompt_logprobs=True,
            type="sample",
        )
        assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_asample(self, client: Tinker) -> None:
        response = client.sampling.with_raw_response.asample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        sampling = response.parse()
        assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_asample(self, client: Tinker) -> None:
        with client.sampling.with_streaming_response.asample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            sampling = response.parse()
            assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_sample(self, client: Tinker) -> None:
        sampling = client.sampling.sample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )
        assert_matches_type(SampleResponse, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_sample_with_all_params(self, client: Tinker) -> None:
        sampling = client.sampling.sample(
            num_samples=1,
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={
                "max_tokens": 0,
                "seed": 0,
                "stop": "string",
                "temperature": 0,
                "top_k": 0,
                "top_p": 0,
            },
            base_model="base_model",
            model_path="model_path",
            prompt_logprobs=True,
            type="sample",
        )
        assert_matches_type(SampleResponse, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_sample(self, client: Tinker) -> None:
        response = client.sampling.with_raw_response.sample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        sampling = response.parse()
        assert_matches_type(SampleResponse, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_sample(self, client: Tinker) -> None:
        with client.sampling.with_streaming_response.sample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            sampling = response.parse()
            assert_matches_type(SampleResponse, sampling, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncSampling:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_asample(self, async_client: AsyncTinker) -> None:
        sampling = await async_client.sampling.asample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )
        assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_asample_with_all_params(self, async_client: AsyncTinker) -> None:
        sampling = await async_client.sampling.asample(
            num_samples=1,
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={
                "max_tokens": 0,
                "seed": 0,
                "stop": "string",
                "temperature": 0,
                "top_k": 0,
                "top_p": 0,
            },
            base_model="base_model",
            model_path="model_path",
            prompt_logprobs=True,
            type="sample",
        )
        assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_asample(self, async_client: AsyncTinker) -> None:
        response = await async_client.sampling.with_raw_response.asample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        sampling = await response.parse()
        assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_asample(self, async_client: AsyncTinker) -> None:
        async with async_client.sampling.with_streaming_response.asample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            sampling = await response.parse()
            assert_matches_type(UntypedAPIFuture, sampling, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_sample(self, async_client: AsyncTinker) -> None:
        sampling = await async_client.sampling.sample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )
        assert_matches_type(SampleResponse, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_sample_with_all_params(self, async_client: AsyncTinker) -> None:
        sampling = await async_client.sampling.sample(
            num_samples=1,
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={
                "max_tokens": 0,
                "seed": 0,
                "stop": "string",
                "temperature": 0,
                "top_k": 0,
                "top_p": 0,
            },
            base_model="base_model",
            model_path="model_path",
            prompt_logprobs=True,
            type="sample",
        )
        assert_matches_type(SampleResponse, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_sample(self, async_client: AsyncTinker) -> None:
        response = await async_client.sampling.with_raw_response.sample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        sampling = await response.parse()
        assert_matches_type(SampleResponse, sampling, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_sample(self, async_client: AsyncTinker) -> None:
        async with async_client.sampling.with_streaming_response.sample(
            prompt={
                "chunks": [
                    {
                        "tokens": [1234, 5678, 9012],
                        "type": "encoded_text",
                    }
                ]
            },
            sampling_params={},
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            sampling = await response.parse()
            assert_matches_type(SampleResponse, sampling, path=["response"])

        assert cast(Any, response.is_closed) is True
