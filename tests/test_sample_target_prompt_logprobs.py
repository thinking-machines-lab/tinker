"""`SamplingClient.sample(target_prompt_logprobs=...)` carries the caller's
target tensor as is, dense or sparse CSR, and hands back the response's target
logprobs in the same layout. These tests pin that contract at `sample`'s seam
with `_sample_async_impl`, plus the JSON shape the tensor takes on the wire; the
transport is covered by the proto tests and the end-to-end debug-server test."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tinker import types
from tinker._compat import model_dump
from tinker.lib.public_interfaces.sampling_client import (
    SamplingClient,
    _check_target_prompt_logprobs,
    _tensor_data_to_model,
)
from tinker.types._pydantic_types.tensor_data import TensorData as TensorDataModel


def _make_client() -> SamplingClient:
    holder = Mock()
    holder.get_client_config.return_value = Mock(
        sample_use_retrieve_futures=False,
        sample_max_concurrent_requests=4,
        sample_enable_stuck_detection=False,
        sample_no_retries=False,
    )
    holder._client_config = holder.get_client_config.return_value
    holder.get_telemetry.return_value = None
    holder.run_coroutine_threadsafe = lambda coro: _Future(asyncio.ensure_future(coro))
    return SamplingClient(holder, sampling_session_id="sampling-session-test")


class _Future:
    def __init__(self, task: asyncio.Task[Any]) -> None:
        self._task = task

    def future(self) -> asyncio.Task[Any]:
        return self._task


def _record_sample_async_impl(
    client: SamplingClient,
    monkeypatch: pytest.MonkeyPatch,
    response: types.SampleResponse,
) -> list[dict[str, Any]]:
    """Replace `_sample_async_impl` with one that records its arguments by name."""
    calls: list[dict[str, Any]] = []

    async def fake_sample_async_impl(
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int = 0,
        topk_sample_logprobs: int = 0,
        target_prompt_logprobs: TensorDataModel | None = None,
    ) -> types.SampleResponse:
        calls.append(
            {
                "prompt": prompt,
                "num_samples": num_samples,
                "sampling_params": sampling_params,
                "include_prompt_logprobs": include_prompt_logprobs,
                "topk_prompt_logprobs": topk_prompt_logprobs,
                "topk_sample_logprobs": topk_sample_logprobs,
                "target_prompt_logprobs": target_prompt_logprobs,
            }
        )
        return response

    monkeypatch.setattr(client, "_sample_async_impl", fake_sample_async_impl)
    return calls


# A 3-token prompt: 2 rows, 3 requested ids among 16 cells. -1 is "don't care";
# the id 0 in row 1 is a real token id.
IDS = torch.full((2, 8), -1, dtype=torch.int64)
IDS[0, 0], IDS[0, 7], IDS[1, 0] = 5, 6, 0
# The server answers in the request's layout: 0.0 in every placeholder cell.
ANSWER = np.zeros((2, 8), dtype=np.float32)
ANSWER[0, 0], ANSWER[0, 7], ANSWER[1, 0] = -0.5, -1.5, -2.5

_ignore_torch_sparse_invariant_warning = pytest.mark.filterwarnings(
    "ignore:Sparse invariant checks are implicitly disabled:UserWarning"
)


def _target_tokens(sparse: bool) -> types.TensorData:
    return (
        types.TensorData.from_torch_sparse(IDS, pad_value=-1)
        if sparse
        else types.TensorData.from_torch(IDS)
    )


def _answer(sparse: bool) -> types.TensorData:
    if not sparse:
        return types.TensorData.from_numpy(ANSWER)
    return types.TensorData(
        data=[-0.5, -1.5, -2.5],
        dtype="float32",
        shape=[2, 8],
        sparse_crow_indices=[0, 2, 3],
        sparse_col_indices=[0, 7, 0],
    )


@_ignore_torch_sparse_invariant_warning
def test_sparse_fixture_lists_exactly_the_requested_cells() -> None:
    sparse_ids = _target_tokens(sparse=True)
    assert sparse_ids.sparse_crow_indices == [0, 2, 3]
    assert sparse_ids.sparse_col_indices == [0, 7, 0]
    assert sparse_ids.data == [5, 6, 0]


@pytest.mark.asyncio
@_ignore_torch_sparse_invariant_warning
@pytest.mark.parametrize("sparse", [False, True])
async def test_sample_passes_the_target_tensor_through_both_ways(
    monkeypatch: pytest.MonkeyPatch, sparse: bool
) -> None:
    target_tokens, answer = _target_tokens(sparse), _answer(sparse)
    client = _make_client()
    calls = _record_sample_async_impl(
        client, monkeypatch, types.SampleResponse(sequences=[], target_prompt_logprobs=answer)
    )
    prompt = types.ModelInput.from_ints([1, 2, 3])
    sampling_params = types.SamplingParams(max_tokens=1)

    result = await client.sample(
        prompt,
        num_samples=1,
        sampling_params=sampling_params,
        target_prompt_logprobs=target_tokens,
    )

    assert result.target_prompt_logprobs is answer
    np.testing.assert_array_equal(result.target_prompt_logprobs.to_numpy(), ANSWER)
    [call] = calls
    assert call["prompt"] is prompt
    assert call["num_samples"] == 1
    assert call["sampling_params"] is sampling_params
    assert call["include_prompt_logprobs"] is False
    assert call["target_prompt_logprobs"] == TensorDataModel(
        data=target_tokens.data,
        dtype="int64",
        shape=[2, 8],
        sparse_crow_indices=target_tokens.sparse_crow_indices,
        sparse_col_indices=target_tokens.sparse_col_indices,
    )


@pytest.mark.asyncio
async def test_sample_sends_no_target_tensor_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    calls = _record_sample_async_impl(client, monkeypatch, types.SampleResponse(sequences=[]))

    result = await client.sample(
        types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
    )

    assert result.target_prompt_logprobs is None
    [call] = calls
    assert call["target_prompt_logprobs"] is None


@pytest.mark.parametrize(
    ("target_tokens", "message"),
    [
        (types.TensorData.from_numpy(np.array([[0.5]], dtype=np.float32)), "must be an int64"),
        (types.TensorData.from_numpy(np.array([5, 6], dtype=np.int64)), "must be 2-D"),
    ],
)
def test_sample_rejects_target_tensors_the_server_would_refuse(
    target_tokens: types.TensorData, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _check_target_prompt_logprobs(target_tokens)
    # Rejected before anything is scheduled, so the caller sees the error directly.
    with pytest.raises(ValueError, match=message):
        _make_client().sample(
            types.ModelInput.from_ints([1, 2]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            target_prompt_logprobs=target_tokens,
        )


@_ignore_torch_sparse_invariant_warning
def test_sample_request_serializes_the_tensor_in_the_json_tensor_shape() -> None:
    """The wire shape is the one the server reads for `loss_fn_inputs`, CSR
    indices included."""
    request = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        sampling_params=types.SamplingParams(max_tokens=1),
        target_prompt_logprobs=_tensor_data_to_model(_target_tokens(sparse=True)),
    )
    body = model_dump(request, exclude_unset=False, exclude_none=True, mode="json")
    assert body["target_prompt_logprobs"] == {
        "data": [5, 6, 0],
        "dtype": "int64",
        "shape": [2, 8],
        "sparse_crow_indices": [0, 2, 3],
        "sparse_col_indices": [0, 7, 0],
    }
    assert types.SampleRequest.model_validate(body).target_prompt_logprobs is not None

    dense_body = model_dump(
        types.SampleRequest(
            prompt=types.ModelInput.from_ints([1, 2, 3]),
            sampling_params=types.SamplingParams(max_tokens=1),
            target_prompt_logprobs=_tensor_data_to_model(_target_tokens(sparse=False)),
        ),
        exclude_unset=False,
        exclude_none=True,
        mode="json",
    )
    assert set(dense_body["target_prompt_logprobs"]) == {"data", "dtype", "shape"}
