from __future__ import annotations

import asyncio
from concurrent.futures import Future

import pytest

from tinker import types
from tinker.lib.public_interfaces.sampling_client import SamplingClient
from tinker.lib.retryable_exception import RetryableException


class _CoroutineHandle:
    def __init__(self, future: Future):
        self._future = future

    def future(self) -> Future:
        return self._future


class _DummyHolder:
    def run_coroutine_threadsafe(self, coro):
        future: Future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except Exception as exc:
            future.set_exception(exc)
        return _CoroutineHandle(future)

    def get_telemetry(self):
        return None


class _RetryOnceHandler:
    def __init__(self):
        self.calls = 0

    async def execute(self, func, *args, **kwargs):
        last_exc = None
        for _ in range(2):
            self.calls += 1
            try:
                return await func(*args, **kwargs)
            except RetryableException as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc


def _make_client(responses: list[types.SampleResponse]):
    client = object.__new__(SamplingClient)
    client.holder = _DummyHolder()
    client.retry_handler = _RetryOnceHandler()
    client._sampling_client_sidecar_handle = None

    async def _sample_async_impl(*args, **kwargs):
        del args, kwargs
        return responses.pop(0)

    client._sample_async_impl = _sample_async_impl
    return client


def test_compute_logprobs_retries_when_prompt_logprobs_missing():
    prompt = types.ModelInput.from_ints([1, 2, 3])
    client = _make_client(
        [
            types.SampleResponse(sequences=[], prompt_logprobs=None),
            types.SampleResponse(sequences=[], prompt_logprobs=[-0.1, -0.2, None]),
        ]
    )

    result = client.compute_logprobs(prompt).result(timeout=5)

    assert result == [-0.1, -0.2, None]
    assert client.retry_handler.calls == 2


def test_compute_logprobs_raises_if_prompt_logprobs_never_arrive():
    prompt = types.ModelInput.from_ints([1, 2, 3])
    client = _make_client(
        [
            types.SampleResponse(sequences=[], prompt_logprobs=None),
            types.SampleResponse(sequences=[], prompt_logprobs=None),
        ]
    )

    with pytest.raises(
        RetryableException, match="omitted prompt_logprobs for a compute_logprobs request"
    ):
        client.compute_logprobs(prompt).result(timeout=5)
