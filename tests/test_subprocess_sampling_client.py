"""Tests for SamplingClient subprocess mode.

These tests use a picklable fake SamplingClient to verify that
subprocess mode correctly routes sample() and compute_logprobs()
through the sidecar subprocess.

Test organization:
    TestRPCRouting    — sample/compute_logprobs delegation through sidecar
    TestErrorHandling — error propagation, sidecar death
    TestPickle        — roundtrip with/without sidecar, re-enable mode
    TestConcurrency   — multithreaded, async, cancelled futures, mixed ops
"""

from __future__ import annotations

import asyncio
import pickle
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

import pytest

from tinker import types
from tinker._exceptions import SidecarDiedError
from tinker.lib.sidecar import create_sidecar_handle

# ---------------------------------------------------------------------------
# Picklable fake SamplingClient (must be module-level for pickling)
# ---------------------------------------------------------------------------


class _FakeSamplingClient:
    """A picklable fake that simulates SamplingClient for testing.

    This is NOT a real SamplingClient — it provides just enough interface
    to test the sidecar integration. Real SamplingClient requires an
    InternalClientHolder and API connection.
    """

    def __init__(self, delay: float = 0.0, fail: bool = False, subprocess_sampling: bool = False):
        self._delay = delay
        self._fail = fail
        self._sampling_client_sidecar_handle = None  # set by create_sidecar_handle() in tests
        if subprocess_sampling:
            from tinker.lib.sidecar import _inside_sidecar

            if not _inside_sidecar:
                self._sampling_client_sidecar_handle = create_sidecar_handle(self)

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> Any:
        # Delegate through sidecar if enabled (mirrors real SamplingClient behavior)
        if self._sampling_client_sidecar_handle is not None:
            from tinker.lib.public_interfaces.sampling_client import _SampleRPC

            return self._sampling_client_sidecar_handle.submit_rpc(
                _SampleRPC(
                    prompt,
                    num_samples,
                    sampling_params,
                    include_prompt_logprobs,
                    topk_prompt_logprobs,
                )
            )

        f: ConcurrentFuture[types.SampleResponse] = ConcurrentFuture()
        if self._fail:
            f.set_exception(RuntimeError("Simulated sample failure"))
        elif self._delay > 0:

            def _delayed():
                time.sleep(self._delay)
                f.set_result(_make_sample_response())

            threading.Thread(target=_delayed, daemon=True).start()
        else:
            f.set_result(_make_sample_response())
        return f

    def compute_logprobs(self, prompt: types.ModelInput) -> Any:
        # Delegate through sidecar if enabled (mirrors real SamplingClient behavior)
        if self._sampling_client_sidecar_handle is not None:
            from tinker.lib.public_interfaces.sampling_client import _ComputeLogprobsRPC

            return self._sampling_client_sidecar_handle.submit_rpc(_ComputeLogprobsRPC(prompt))

        f: ConcurrentFuture[list[float | None]] = ConcurrentFuture()
        if self._fail:
            f.set_exception(RuntimeError("Simulated logprobs failure"))
        else:
            f.set_result([0.1, 0.2, None])
        return f

    def __reduce__(self) -> tuple[type, tuple[float, bool, bool]]:
        return (
            _FakeSamplingClient,
            (self._delay, self._fail, self._sampling_client_sidecar_handle is not None),
        )


def _make_sample_response() -> types.SampleResponse:
    return types.SampleResponse(
        sequences=[
            types.SampledSequence(
                stop_reason="length",
                tokens=[1, 2, 3],
                logprobs=[0.1, 0.2, 0.3],
            )
        ],
        type="sample",
    )


def _create_proxy(delay: float = 0.0, fail: bool = False) -> _FakeSamplingClient:
    """Create a fake client with sidecar handle for testing."""
    client = _FakeSamplingClient(delay=delay, fail=fail)
    client._sampling_client_sidecar_handle = create_sidecar_handle(client)
    return client


_PROMPT = types.ModelInput.from_ints([1, 2, 3])
_PARAMS = types.SamplingParams(max_tokens=10)


# ===========================================================================
# Tests
# ===========================================================================


class TestRPCRouting:
    """Verify sample() and compute_logprobs() are routed through the sidecar."""

    def test_sample(self):
        """sample() → subprocess → SampleResponse."""
        proxy = _create_proxy()
        result = proxy.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
        assert isinstance(result, types.SampleResponse)
        assert result.sequences[0].tokens == [1, 2, 3]

    def test_constructor_enables_subprocess_mode(self):
        """subprocess_sampling=True in __init__ creates the sidecar handle."""
        client = _FakeSamplingClient(subprocess_sampling=True)
        assert client._sampling_client_sidecar_handle is not None
        result = client.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
        assert isinstance(result, types.SampleResponse)

    def test_compute_logprobs(self):
        """compute_logprobs() → subprocess → list of logprobs."""
        proxy = _create_proxy()
        result = proxy.compute_logprobs(_PROMPT).result(timeout=10)
        assert result == [0.1, 0.2, None]

    def test_mixed_sample_and_logprobs(self):
        """Interleaved sample() and compute_logprobs() all resolve correctly."""
        proxy = _create_proxy(delay=0.01)

        futures_sample = [proxy.sample(_PROMPT, 1, _PARAMS) for _ in range(10)]
        futures_logprobs = [proxy.compute_logprobs(_PROMPT) for _ in range(10)]

        for f in futures_sample:
            result = f.result(timeout=10)
            assert isinstance(result, types.SampleResponse)
            assert result.sequences[0].tokens == [1, 2, 3]

        for f in futures_logprobs:
            assert f.result(timeout=10) == [0.1, 0.2, None]


class TestErrorHandling:
    """Error propagation from subprocess to caller."""

    def test_sample_error(self):
        """Exceptions from sample() in the subprocess are propagated."""
        proxy = _create_proxy(fail=True)
        with pytest.raises(RuntimeError, match="Simulated sample failure"):
            proxy.sample(_PROMPT, 1, _PARAMS).result(timeout=10)

    def test_compute_logprobs_error(self):
        """Exceptions from compute_logprobs() in the subprocess are propagated."""
        proxy = _create_proxy(fail=True)
        with pytest.raises(RuntimeError, match="Simulated logprobs failure"):
            proxy.compute_logprobs(_PROMPT).result(timeout=10)

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_sidecar_death_fails_pending_futures(self):
        """When the subprocess is killed, pending futures get SidecarDiedError."""
        proxy = _create_proxy(delay=0.5)
        future = proxy.sample(_PROMPT, 1, _PARAMS)

        # Kill the underlying subprocess directly
        sidecar = proxy._sampling_client_sidecar_handle._sidecar
        assert sidecar._process is not None
        sidecar._process.kill()
        sidecar._process.join(timeout=5)

        with pytest.raises(SidecarDiedError):
            future.result(timeout=5)


class TestPickle:
    """Pickle roundtrip preserves subprocess mode correctly."""

    def test_roundtrip_preserves_subprocess_mode(self):
        """Pickling a sidecar-enabled client re-enables subprocess mode on unpickle."""
        proxy = _create_proxy()
        assert proxy._sampling_client_sidecar_handle is not None

        restored = pickle.loads(pickle.dumps(proxy))
        assert restored._sampling_client_sidecar_handle is not None

        result = restored.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
        assert isinstance(result, types.SampleResponse)

    def test_roundtrip_without_sidecar(self):
        """Pickling a client without subprocess mode keeps it disabled."""
        client = _FakeSamplingClient()
        assert client._sampling_client_sidecar_handle is None
        restored = pickle.loads(pickle.dumps(client))
        assert restored._sampling_client_sidecar_handle is None

    def test_re_enable_subprocess_mode(self):
        """Replacing the sidecar handle works cleanly."""
        client = _FakeSamplingClient()
        client._sampling_client_sidecar_handle = create_sidecar_handle(client)

        # First handle works
        assert isinstance(
            client.sample(_PROMPT, 1, _PARAMS).result(timeout=10), types.SampleResponse
        )

        # Replace with a new handle (old one is GC'd and unregistered)
        client._sampling_client_sidecar_handle = create_sidecar_handle(client)

        # New handle also works
        assert isinstance(
            client.sample(_PROMPT, 1, _PARAMS).result(timeout=10), types.SampleResponse
        )


class TestConcurrency:
    """Thread safety and concurrent operations through the sidecar."""

    def test_multithreaded_samples(self):
        """sample() from 20 threads all resolve correctly."""
        proxy = _create_proxy(delay=0.01)
        results: list[types.SampleResponse | None] = [None] * 20
        errors: list[Exception] = []

        def _worker(idx: int) -> None:
            try:
                results[idx] = proxy.sample(_PROMPT, 1, _PARAMS).result(timeout=30)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised: {errors}"
        for r in results:
            assert isinstance(r, types.SampleResponse)
            assert r.sequences[0].tokens == [1, 2, 3]

    def test_multithreaded_mixed_operations(self):
        """sample() and compute_logprobs() from different threads simultaneously."""
        proxy = _create_proxy(delay=0.01)
        errors: list[Exception] = []

        def _sample_worker() -> None:
            try:
                for _ in range(10):
                    r = proxy.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
                    assert isinstance(r, types.SampleResponse)
            except Exception as e:
                errors.append(e)

        def _logprobs_worker() -> None:
            try:
                for _ in range(10):
                    r = proxy.compute_logprobs(_PROMPT).result(timeout=10)
                    assert r == [0.1, 0.2, None]
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_sample_worker) for _ in range(3)]
        threads += [threading.Thread(target=_logprobs_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors: {errors}"

    def test_async_concurrent_samples(self):
        """Multiple async sample calls via asyncio.gather all resolve."""
        proxy = _create_proxy(delay=0.01)

        async def _run() -> list[types.SampleResponse]:
            from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture

            coros = [
                AwaitableConcurrentFuture(proxy.sample(_PROMPT, 1, _PARAMS)) for _ in range(20)
            ]
            return await asyncio.gather(*coros)

        results = asyncio.run(_run())
        assert len(results) == 20
        for r in results:
            assert isinstance(r, types.SampleResponse)

    def test_cancelled_future_does_not_crash_collector(self):
        """Cancelling a future doesn't kill the collector thread."""
        proxy = _create_proxy(delay=0.5)

        future1 = proxy.sample(_PROMPT, 1, _PARAMS)
        future1.cancel()

        result = proxy.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
        assert isinstance(result, types.SampleResponse)

    def test_multiple_clients_share_sidecar(self):
        """Two independent clients sharing the sidecar singleton work concurrently."""
        proxy1 = _create_proxy(delay=0.01)
        proxy2 = _create_proxy(delay=0.01)
        errors: list[Exception] = []

        def _worker1() -> None:
            try:
                for _ in range(10):
                    r = proxy1.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
                    assert isinstance(r, types.SampleResponse)
            except Exception as e:
                errors.append(e)

        def _worker2() -> None:
            try:
                for _ in range(10):
                    r = proxy2.compute_logprobs(_PROMPT).result(timeout=10)
                    assert r == [0.1, 0.2, None]
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_worker1)
        t2 = threading.Thread(target=_worker2)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors: {errors}"

    def test_pickle_roundtrip_then_concurrent_use(self):
        """Pickle a client, restore it, then use from multiple threads."""
        proxy = _create_proxy(delay=0.01)
        restored = pickle.loads(pickle.dumps(proxy))
        assert restored._sampling_client_sidecar_handle is not None

        errors: list[Exception] = []

        def _worker() -> None:
            try:
                for _ in range(10):
                    r = restored.sample(_PROMPT, 1, _PARAMS).result(timeout=10)
                    assert isinstance(r, types.SampleResponse)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors: {errors}"
