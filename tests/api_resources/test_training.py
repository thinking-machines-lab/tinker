# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import Any, cast

import pytest

from tinker import Tinker, AsyncTinker
from tests.utils import assert_matches_type
from tinker.types.shared import UntypedAPIFuture

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")


class TestTraining:
    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_forward(self, client: Tinker) -> None:
        training = client.training.forward(
            forward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    def test_method_create(self, client: Tinker) -> None:
        training = client.training.create(
            base_model="meta-llama/Llama-3.2-1B",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_method_create_with_all_params(self, client: Tinker) -> None:
        training = client.training.create(
            base_model="meta-llama/Llama-3.2-1B",
            lora_config={"rank": 16},
            type="create_model",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_raw_response_create(self, client: Tinker) -> None:
        response = client.training.with_raw_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_streaming_response_create(self, client: Tinker) -> None:
        with client.training.with_streaming_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip()
    @parametrize
    def test_method_forward(self, client: Tinker) -> None:
        training = client.training.forward(
            fwdbwd_input={

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_forward(self, client: Tinker) -> None:
        response = client.training.with_raw_response.forward(
            forward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_forward(self, client: Tinker) -> None:
        with client.training.with_streaming_response.forward(
            forward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_forward_backward(self, client: Tinker) -> None:
        training = client.training.forward_backward(
            forward_backward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_forward_backward(self, client: Tinker) -> None:
        response = client.training.with_raw_response.forward_backward(
            forward_backward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_forward_backward(self, client: Tinker) -> None:
        with client.training.with_streaming_response.forward_backward(
            forward_backward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_optim_step(self, client: Tinker) -> None:
        training = client.training.optim_step(
            adam_params={"learning_rate": 0},
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_method_optim_step_with_all_params(self, client: Tinker) -> None:
        training = client.training.optim_step(
            adam_params={
                "learning_rate": 0,
                "beta1": 0,
                "beta2": 0,
                "eps": 0,
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="optim_step",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_raw_response_optim_step(self, client: Tinker) -> None:
        response = client.training.with_raw_response.optim_step(
            adam_params={"learning_rate": 0},
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    def test_streaming_response_optim_step(self, client: Tinker) -> None:
        with client.training.with_streaming_response.optim_step(
            adam_params={"learning_rate": 0},
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip()
    @parametrize
    def test_method_unload(self, client: Tinker) -> None:
        training = client.training.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_method_unload_with_all_params(self, client: Tinker) -> None:
        training = client.training.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="unload_model",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_raw_response_unload(self, client: Tinker) -> None:
        response = client.training.with_raw_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    def test_streaming_response_unload(self, client: Tinker) -> None:
        with client.training.with_streaming_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True


class TestAsyncTraining:
    parametrize = pytest.mark.parametrize(
        "async_client", [False, True, {"http_client": "aiohttp"}], indirect=True, ids=["loose", "strict", "aiohttp"]
    )

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_forward(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.forward(
            forward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    async def test_method_create(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.create(
            base_model="meta-llama/Llama-3.2-1B",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_method_create_with_all_params(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.create(
            base_model="meta-llama/Llama-3.2-1B",
            lora_config={"rank": 16},
            type="create_model",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_raw_response_create(self, async_client: AsyncTinker) -> None:
        response = await async_client.training.with_raw_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = await response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_streaming_response_create(self, async_client: AsyncTinker) -> None:
        async with async_client.training.with_streaming_response.create(
            base_model="meta-llama/Llama-3.2-1B",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = await response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip()
    @parametrize
    async def test_method_forward(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.forward(
            fwdbwd_input={

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_forward(self, async_client: AsyncTinker) -> None:
        response = await async_client.training.with_raw_response.forward(
            forward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = await response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_forward(self, async_client: AsyncTinker) -> None:
        async with async_client.training.with_streaming_response.forward(
            forward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = await response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_forward_backward(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.forward_backward(
            forward_backward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_forward_backward(self, async_client: AsyncTinker) -> None:
        response = await async_client.training.with_raw_response.forward_backward(
            forward_backward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = await response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_forward_backward(self, async_client: AsyncTinker) -> None:
        async with async_client.training.with_streaming_response.forward_backward(
            forward_backward_input={
                "data": [
                    {
                        "loss_fn_inputs": {
                            "weights": {
                                "data": [1, 1, 1, 0.5, 0],
                                "dtype": "float32",
                            },
                            "target_tokens": {
                                "data": [123, 456, 789, 101, 202],
                                "dtype": "int64",
                            },
                        },
                        "model_input": {
                            "chunks": [
                                {
                                    "tokens": [1234, 5678, 9012],
                                    "type": "encoded_text",
                                }
                            ]
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = await response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_optim_step(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.optim_step(
            adam_params={"learning_rate": 0},
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_method_optim_step_with_all_params(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.optim_step(
            adam_params={
                "learning_rate": 0,
                "beta1": 0,
                "beta2": 0,
                "eps": 0,
            },
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="optim_step",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_raw_response_optim_step(self, async_client: AsyncTinker) -> None:
        response = await async_client.training.with_raw_response.optim_step(
            adam_params={"learning_rate": 0},
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = await response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip(reason="Prism tests are disabled")
    @parametrize
    async def test_streaming_response_optim_step(self, async_client: AsyncTinker) -> None:
        async with async_client.training.with_streaming_response.optim_step(
            adam_params={"learning_rate": 0},
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = await response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True

    @pytest.mark.skip()
    @parametrize
    async def test_method_unload(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_method_unload_with_all_params(self, async_client: AsyncTinker) -> None:
        training = await async_client.training.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
            type="unload_model",
        )
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_raw_response_unload(self, async_client: AsyncTinker) -> None:
        response = await async_client.training.with_raw_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        )

        assert response.is_closed is True
        assert response.http_request.headers.get("X-Stainless-Lang") == "python"
        training = await response.parse()
        assert_matches_type(UntypedAPIFuture, training, path=["response"])

    @pytest.mark.skip()
    @parametrize
    async def test_streaming_response_unload(self, async_client: AsyncTinker) -> None:
        async with async_client.training.with_streaming_response.unload(
            model_id="123e4567-e89b-12d3-a456-426614174000",
        ) as response:
            assert not response.is_closed
            assert response.http_request.headers.get("X-Stainless-Lang") == "python"

            training = await response.parse()
            assert_matches_type(UntypedAPIFuture, training, path=["response"])

        assert cast(Any, response.is_closed) is True
