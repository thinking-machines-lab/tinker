"""Tests for SubprocessSidecar.

Tests use simple picklable objects to verify the subprocess worker,
response collector, and sidecar proxy behavior without any domain-specific
dependencies.

Test organization:
    Unit tests (components in isolation):
        TestSubprocessWorker — raw worker process via queues
        TestPutResponse      — response serialization helper
        TestResponseCollector — collector thread in isolation
        TestSetLoop           — InternalClientHolderThreadSingleton._set_loop

    End-to-end tests (SubprocessSidecar as a whole):
        TestRPCExecution — happy-path RPC flow, return types, async, on-loop unpickling
        TestErrorHandling — exception propagation, serialization failures, missing targets
        TestLifecycle     — shutdown, subprocess death, pickling, singleton, nesting guard
        TestMultiTarget   — multi-target isolation, GC unregistration
        TestConcurrency   — thread safety, concurrent submit/shutdown, stress
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import multiprocessing
import pickle
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

import pytest

from tinker._exceptions import SidecarDiedError, SidecarIPCError
from tinker.lib.internal_client_holder import InternalClientHolderThreadSingleton
from tinker.lib.sidecar import (
    _STARTUP_OK,
    SidecarRPC,
    SubprocessSidecar,
    _get_sidecar,
    _mp_context,
    _put_response,
    _RegisterTargetRPC,
    _ResponseCollector,
    _subprocess_worker,
    create_sidecar_handle,
)

# ---------------------------------------------------------------------------
# Picklable fake targets (must be module-level for pickling)
# ---------------------------------------------------------------------------


class _Calculator:
    """Simple picklable target with sync and Future-returning methods."""

    def __init__(self, delay: float = 0.0, fail: bool = False):
        self._delay = delay
        self._fail = fail

    def add(self, a: int, b: int) -> int:
        if self._fail:
            raise RuntimeError("Simulated failure")
        if self._delay > 0:
            time.sleep(self._delay)
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def add_future(self, a: int, b: int) -> ConcurrentFuture[int]:
        """Returns a Future (simulates async-style APIs like SamplingClient)."""
        f: ConcurrentFuture[int] = ConcurrentFuture()
        if self._fail:
            f.set_exception(RuntimeError("Simulated failure"))
        elif self._delay > 0:

            def _delayed():
                time.sleep(self._delay)
                f.set_result(a + b)

            threading.Thread(target=_delayed, daemon=True).start()
        else:
            f.set_result(a + b)
        return f

    def __reduce__(self) -> tuple[type, tuple[float, bool]]:
        return (_Calculator, (self._delay, self._fail))


class _Multiplier:
    """A second picklable target for multi-target testing."""

    def __init__(self, factor: int = 2):
        self._factor = factor

    def scale(self, x: int) -> int:
        return x * self._factor

    def __reduce__(self) -> tuple[type, tuple[int]]:
        return (_Multiplier, (self._factor,))


class _LoopAwareTarget:
    """Target that creates asyncio tasks during unpickling.

    Simulates the InternalClientHolder shadow-holder pattern: when
    unpickled on the sidecar event loop, it detects it's on the loop
    thread and uses create_task() instead of blocking .result().
    """

    def __init__(self) -> None:
        self._on_loop: bool = False
        self._task_created: bool = False
        self._task_completed: bool = False
        try:
            loop = asyncio.get_running_loop()
            self._on_loop = True

            async def _background():
                await asyncio.sleep(0.01)
                self._task_completed = True

            self._task = loop.create_task(_background())
            self._task_created = True
        except RuntimeError:
            pass

    def get_info(self) -> dict[str, bool]:
        return {
            "on_loop": self._on_loop,
            "task_created": self._task_created,
            "task_completed": self._task_completed,
        }

    def __reduce__(self) -> tuple[type, tuple[()]]:
        return (_LoopAwareTarget, ())


# ---------------------------------------------------------------------------
# Typed RPCs (must be module-level for pickling)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _AddRPC(SidecarRPC):
    a: int
    b: int

    async def execute(self, target: Any) -> int:
        return target.add(self.a, self.b)


@dataclasses.dataclass
class _MultiplyRPC(SidecarRPC):
    a: int
    b: int

    async def execute(self, target: Any) -> int:
        return target.multiply(self.a, self.b)


@dataclasses.dataclass
class _AddFutureRPC(SidecarRPC):
    """RPC that returns a ConcurrentFuture — sidecar awaits it automatically."""

    a: int
    b: int

    async def execute(self, target: Any) -> ConcurrentFuture[int]:
        return target.add_future(self.a, self.b)


@dataclasses.dataclass
class _ScaleRPC(SidecarRPC):
    x: int

    async def execute(self, target: Any) -> int:
        return target.scale(self.x)


@dataclasses.dataclass
class _NoneRPC(SidecarRPC):
    """RPC that returns None."""

    async def execute(self, target: Any) -> None:
        return None


@dataclasses.dataclass
class _UnpicklableResultRPC(SidecarRPC):
    """RPC that returns an object that can't be pickled."""

    async def execute(self, target: Any) -> Any:
        return lambda: "i am unpicklable"


class _DoubleAddRPC(SidecarRPC):
    """Multi-step RPC: calls target.add twice in sequence."""

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    async def execute(self, target: Any) -> int:
        first = target.add(self.a, self.b)
        second = target.add(first, self.b)
        return second


class _UnpicklableRPC:
    """Not a dataclass, has a lambda that prevents pickling."""

    def __init__(self) -> None:
        self.callback = lambda: None

    async def execute(self, target: Any) -> int:
        return 42


@dataclasses.dataclass
class _GetInfoRPC(SidecarRPC):
    async def execute(self, target: Any) -> dict[str, bool]:
        return target.get_info()


@dataclasses.dataclass
class _WaitTaskRPC(SidecarRPC):
    """Wait for the background task created during unpickling, then return info."""

    async def execute(self, target: Any) -> dict[str, bool]:
        if hasattr(target, "_task"):
            await target._task
        return target.get_info()


# ===========================================================================
# Unit tests — components in isolation
# ===========================================================================


def _register_target_in_worker(
    request_queue: multiprocessing.Queue[Any],
    response_queue: multiprocessing.Queue[Any],
    target: Any,
    request_id: int = 0,
) -> int:
    """Helper: register a target via RPC and return the target_id."""
    rpc = _RegisterTargetRPC(pickled_target=pickle.dumps(target))
    # Wire format: (request_id, target_id, rpc) — target_id=None for built-in RPCs
    request_queue.put(pickle.dumps((request_id, None, rpc)))
    rid, target_id, exc = pickle.loads(response_queue.get(timeout=10))
    assert rid == request_id
    assert exc is None
    assert isinstance(target_id, int)
    return target_id


class TestSubprocessWorker:
    """Tests for _subprocess_worker — raw worker process via queues."""

    def setup_method(self) -> None:
        self._parent_conn, self._child_conn = _mp_context.Pipe()

    def teardown_method(self) -> None:
        self._parent_conn.close()

    def test_processes_direct_return(self):
        """Worker processes a method that returns a value directly."""
        request_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        proc = _mp_context.Process(
            target=_subprocess_worker,
            args=(request_queue, response_queue, self._child_conn),
            daemon=True,
        )
        proc.start()

        startup_msg = response_queue.get(timeout=10)
        assert startup_msg[0] == "__startup_ok__"

        target_id = _register_target_in_worker(request_queue, response_queue, _Calculator())

        request_queue.put(pickle.dumps((42, target_id, _AddRPC(a=3, b=4))))

        request_id, result, exception = pickle.loads(response_queue.get(timeout=10))
        assert request_id == 42
        assert exception is None
        assert result == 7

        request_queue.put(None)
        self._parent_conn.close()
        proc.join(timeout=5)
        assert not proc.is_alive()

    def test_processes_future_return(self):
        """Worker handles methods that return a Future."""
        request_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        proc = _mp_context.Process(
            target=_subprocess_worker,
            args=(request_queue, response_queue, self._child_conn),
            daemon=True,
        )
        proc.start()

        startup_msg = response_queue.get(timeout=10)
        assert startup_msg[0] == "__startup_ok__"

        target_id = _register_target_in_worker(request_queue, response_queue, _Calculator())

        request_queue.put(pickle.dumps((7, target_id, _AddFutureRPC(a=10, b=20))))

        request_id, result, exception = pickle.loads(response_queue.get(timeout=10))
        assert request_id == 7
        assert exception is None
        assert result == 30

        request_queue.put(None)
        proc.join(timeout=5)

    def test_handles_method_exception(self):
        """When the called method raises, the exception is sent back."""
        request_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        proc = _mp_context.Process(
            target=_subprocess_worker,
            args=(request_queue, response_queue, self._child_conn),
            daemon=True,
        )
        proc.start()

        startup_msg = response_queue.get(timeout=10)
        assert startup_msg[0] == "__startup_ok__"

        target_id = _register_target_in_worker(
            request_queue, response_queue, _Calculator(fail=True)
        )

        request_queue.put(pickle.dumps((99, target_id, _AddRPC(a=1, b=2))))

        request_id, result, exception = pickle.loads(response_queue.get(timeout=10))
        assert request_id == 99
        assert result is None
        assert isinstance(exception, RuntimeError)
        assert "Simulated failure" in str(exception)

        request_queue.put(None)
        proc.join(timeout=5)

    def test_shutdown_on_sentinel(self):
        """Sending None causes the worker to exit cleanly."""
        request_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        proc = _mp_context.Process(
            target=_subprocess_worker,
            args=(request_queue, response_queue, self._child_conn),
            daemon=True,
        )
        proc.start()

        startup_msg = response_queue.get(timeout=10)
        assert startup_msg[0] == "__startup_ok__"

        request_queue.put(None)
        self._parent_conn.close()
        proc.join(timeout=5)
        assert not proc.is_alive()

    def test_register_target_unpickle_error(self):
        """Bad pickled_target bytes surface as a normal RPC error."""
        request_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        proc = _mp_context.Process(
            target=_subprocess_worker,
            args=(request_queue, response_queue, self._child_conn),
            daemon=True,
        )
        proc.start()

        startup_msg = response_queue.get(timeout=10)
        assert startup_msg[0] == "__startup_ok__"

        # Send corrupt pickle data as a target registration
        rpc = _RegisterTargetRPC(pickled_target=b"not valid pickle data")
        request_queue.put(pickle.dumps((1, None, rpc)))

        request_id, result, exception = pickle.loads(response_queue.get(timeout=10))
        assert request_id == 1
        assert result is None
        assert exception is not None

        request_queue.put(None)
        proc.join(timeout=5)

    def test_unregistered_target_id_gives_clear_error(self):
        """Sending an RPC with a target_id that doesn't exist gives a clear error."""
        request_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        proc = _mp_context.Process(
            target=_subprocess_worker,
            args=(request_queue, response_queue, self._child_conn),
            daemon=True,
        )
        proc.start()

        startup_msg = response_queue.get(timeout=10)
        assert startup_msg[0] == "__startup_ok__"

        request_queue.put(pickle.dumps((1, 999, _AddRPC(a=1, b=2))))

        request_id, result, exception = pickle.loads(response_queue.get(timeout=10))
        assert request_id == 1
        assert result is None
        assert isinstance(exception, RuntimeError)
        assert "target_id=999" in str(exception)
        assert "not registered" in str(exception)
        assert "_AddRPC" in str(exception)

        request_queue.put(None)
        proc.join(timeout=5)


class TestPutResponse:
    """Tests for _put_response serialization helper."""

    def test_wraps_unpicklable_exception(self):
        """Unpicklable exceptions are wrapped in SidecarIPCError."""
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()

        class UnpicklableError(Exception):
            def __reduce__(self):
                raise TypeError("Cannot pickle this")

        _put_response(response_queue, 1, None, UnpicklableError("test"))

        request_id, result, exception = pickle.loads(response_queue.get(timeout=5))
        assert request_id == 1
        assert result is None
        assert isinstance(exception, SidecarIPCError)
        assert "UnpicklableError" in str(exception)


class _FakeProcess:
    """Minimal fake process for testing _ResponseCollector in isolation."""

    exitcode: int | None = None

    def is_alive(self) -> bool:
        return True


class TestResponseCollector:
    """Tests for _ResponseCollector in isolation (no real subprocess)."""

    @staticmethod
    def _start_collector(
        response_queue: multiprocessing.Queue[Any],
        pending: dict[int, ConcurrentFuture[Any]],
        lock: threading.Lock,
    ) -> _ResponseCollector:
        """Create and start a collector, passing it through the startup handshake."""
        response_queue.put((_STARTUP_OK, None, None))
        collector = _ResponseCollector(
            response_queue,
            pending,
            lock,
            _FakeProcess(),  # type: ignore[arg-type]
        )
        collector.start()
        collector.wait_ready()
        return collector

    def test_resolves_futures(self):
        """Responses from queue are matched to pending futures."""
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        pending: dict[int, ConcurrentFuture[Any]] = {}
        lock = threading.Lock()

        f1: ConcurrentFuture[str] = ConcurrentFuture()
        f2: ConcurrentFuture[str] = ConcurrentFuture()
        pending[1] = f1
        pending[2] = f2

        collector = self._start_collector(response_queue, pending, lock)

        response_queue.put(pickle.dumps((1, "result_1", None)))
        response_queue.put(pickle.dumps((2, "result_2", None)))

        assert f1.result(timeout=5) == "result_1"
        assert f2.result(timeout=5) == "result_2"

        response_queue.put(None)
        collector.join(timeout=5)

    def test_resolves_exception(self):
        """Exception responses set the exception on the future."""
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        pending: dict[int, ConcurrentFuture[Any]] = {}
        lock = threading.Lock()

        f: ConcurrentFuture[str] = ConcurrentFuture()
        pending[1] = f

        collector = self._start_collector(response_queue, pending, lock)

        response_queue.put(pickle.dumps((1, None, RuntimeError("test error"))))

        with pytest.raises(RuntimeError, match="test error"):
            f.result(timeout=5)

        response_queue.put(None)
        collector.join(timeout=5)

    def test_fails_all_pending_on_process_death(self):
        """When process dies, all pending futures get SidecarDiedError."""
        response_queue: multiprocessing.Queue[Any] = _mp_context.Queue()
        pending: dict[int, ConcurrentFuture[Any]] = {}
        lock = threading.Lock()

        f: ConcurrentFuture[str] = ConcurrentFuture()
        pending[1] = f

        collector = self._start_collector(response_queue, pending, lock)

        # Send sentinel to stop collector (simulates process exit)
        response_queue.put(None)
        collector.join(timeout=5)

        with pytest.raises(SidecarDiedError, match="exited unexpectedly.*exit code"):
            f.result(timeout=5)


class TestSetLoop:
    """Tests for InternalClientHolderThreadSingleton._set_loop."""

    def test_injects_loop(self):
        """_set_loop injects a loop that get_loop returns, with no background thread."""
        singleton = InternalClientHolderThreadSingleton()
        loop = asyncio.new_event_loop()
        try:
            singleton._set_loop(loop)
            assert singleton.get_loop() is loop
            singleton._ensure_started()  # no-op after _set_loop
            assert singleton._thread is None
        finally:
            loop.close()

    def test_rejects_after_ensure_started(self):
        """_set_loop raises if the singleton already started its own loop."""
        singleton = InternalClientHolderThreadSingleton()
        singleton._ensure_started()
        with pytest.raises(RuntimeError, match="Cannot set_loop after singleton has started"):
            singleton._set_loop(asyncio.new_event_loop())

    def test_rejects_double_call(self):
        """_set_loop cannot be called twice."""
        singleton = InternalClientHolderThreadSingleton()
        loop = asyncio.new_event_loop()
        try:
            singleton._set_loop(loop)
            with pytest.raises(RuntimeError, match="Cannot set_loop after singleton has started"):
                singleton._set_loop(asyncio.new_event_loop())
        finally:
            loop.close()

    def test_concurrent_with_ensure_started(self):
        """Concurrent _set_loop and _ensure_started: exactly one wins, no crash."""
        for _ in range(20):
            singleton = InternalClientHolderThreadSingleton()
            loop = asyncio.new_event_loop()
            errors: list[Exception] = []
            barrier = threading.Barrier(2)

            def _do_set_loop():
                barrier.wait()
                try:
                    singleton._set_loop(loop)
                except RuntimeError:
                    pass  # lost the race — fine
                except Exception as e:
                    errors.append(e)

            def _do_ensure_started():
                barrier.wait()
                try:
                    singleton._ensure_started()
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=_do_set_loop)
            t2 = threading.Thread(target=_do_ensure_started)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

            assert not errors, f"Unexpected errors: {errors}"
            assert singleton._started
            assert singleton.get_loop() is not None
            loop.close()


# ===========================================================================
# End-to-end tests — SubprocessSidecar as a whole
# ===========================================================================


class TestRPCExecution:
    """Happy-path RPC execution: return types, async, on-loop unpickling."""

    def test_direct_return(self):
        """RPC with a direct return value."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        assert handle.submit_rpc(_AddRPC(a=3, b=4)).result(timeout=10) == 7

    def test_future_return(self):
        """RPC returning a ConcurrentFuture is automatically awaited."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        assert handle.submit_rpc(_AddFutureRPC(a=10, b=20)).result(timeout=10) == 30

    def test_multiple_rpc_types(self):
        """Different RPC types on the same handle all route correctly."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        assert handle.submit_rpc(_AddRPC(a=1, b=2)).result(timeout=10) == 3
        assert handle.submit_rpc(_MultiplyRPC(a=3, b=4)).result(timeout=10) == 12

    def test_multi_step_rpc(self):
        """Custom SidecarRPC subclass can call target methods multiple times."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        # (3+4) + 4 = 11
        assert handle.submit_rpc(_DoubleAddRPC(3, 4)).result(timeout=10) == 11

    def test_none_result(self):
        """execute() returning None propagates correctly."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        assert handle.submit_rpc(_NoneRPC()).result(timeout=10) is None

    def test_async_submit(self):
        """submit_rpc() futures can be awaited via asyncio."""
        from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture

        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))

        async def _run() -> list[int]:
            coros = [
                AwaitableConcurrentFuture(handle.submit_rpc(_AddRPC(a=i, b=i))) for i in range(10)
            ]
            return await asyncio.gather(*coros)

        assert asyncio.run(_run()) == [i + i for i in range(10)]

    def test_on_loop_task_during_unpickling(self):
        """Target unpickled on the event loop can create asyncio tasks that complete."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_LoopAwareTarget()))
        info = handle.submit_rpc(_WaitTaskRPC()).result(timeout=10)
        assert info["on_loop"] is True
        assert info["task_created"] is True
        assert info["task_completed"] is True


class TestErrorHandling:
    """Exception propagation, serialization failures, missing targets."""

    def test_target_exception(self):
        """Exceptions from sync target methods are propagated."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(fail=True)))
        with pytest.raises(RuntimeError, match="Simulated failure"):
            handle.submit_rpc(_AddRPC(a=1, b=2)).result(timeout=10)

    def test_future_exception(self):
        """Exceptions from Future results are propagated."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(fail=True)))
        with pytest.raises(RuntimeError, match="Simulated failure"):
            handle.submit_rpc(_AddFutureRPC(a=1, b=2)).result(timeout=10)

    def test_sidecar_continues_after_rpc_error(self):
        """A failed RPC doesn't break the sidecar for subsequent RPCs."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(fail=True)))
        with pytest.raises(RuntimeError, match="Simulated failure"):
            handle.submit_rpc(_AddRPC(a=1, b=2)).result(timeout=10)

        handle2 = sidecar.register_target(pickle.dumps(_Calculator()))
        assert handle2.submit_rpc(_AddRPC(a=10, b=20)).result(timeout=10) == 30

    def test_unpicklable_result(self):
        """execute() returning an unpicklable object gives SidecarIPCError."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        with pytest.raises(SidecarIPCError, match="Failed to serialize response"):
            handle.submit_rpc(_UnpicklableResultRPC()).result(timeout=10)

    def test_unpicklable_rpc(self):
        """Submitting an RPC that can't be pickled raises SidecarIPCError."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        with pytest.raises(SidecarIPCError, match="Failed to serialize"):
            handle.submit_rpc(_UnpicklableRPC()).result(timeout=10)  # type: ignore[arg-type]

    def test_unregistered_target_id(self):
        """RPC with an unknown target_id gives a clear error message."""
        sidecar = SubprocessSidecar()
        with pytest.raises(RuntimeError, match="not registered"):
            sidecar._submit_rpc(_AddRPC(a=1, b=2), target_id=999).result(timeout=10)

    def test_bad_pickle_registration(self):
        """Registering corrupt pickle bytes surfaces as a normal exception."""
        sidecar = SubprocessSidecar()
        with pytest.raises(Exception):
            sidecar.register_target(b"not valid pickle data")

    def test_wrong_rpc_for_target_type(self):
        """Wrong RPC for target type gives a clear error, other targets still work."""
        sidecar = SubprocessSidecar()
        calc_handle = sidecar.register_target(pickle.dumps(_Calculator()))
        mult_handle = sidecar.register_target(pickle.dumps(_Multiplier(factor=3)))

        with pytest.raises(AttributeError):
            mult_handle.submit_rpc(_MultiplyRPC(a=2, b=3)).result(timeout=10)

        # Correct target still works after the error
        assert calc_handle.submit_rpc(_MultiplyRPC(a=2, b=3)).result(timeout=10) == 6


class TestLifecycle:
    """Shutdown, subprocess death, pickling prevention, singleton, nesting guard."""

    def test_shutdown_is_idempotent(self):
        """Calling _shutdown() multiple times doesn't raise."""
        sidecar = SubprocessSidecar()
        sidecar._shutdown()
        sidecar._shutdown()

    def test_submit_after_shutdown_raises(self):
        """_submit_rpc() after _shutdown() raises SidecarDiedError immediately."""
        sidecar = SubprocessSidecar()
        sidecar._shutdown()
        with pytest.raises(SidecarDiedError, match="not running"):
            sidecar._submit_rpc(_AddRPC(a=1, b=2))

    def test_parent_pipe_close_terminates_subprocess(self):
        """Closing the parent pipe causes the subprocess to self-terminate."""
        sidecar = SubprocessSidecar()
        process = sidecar._process
        assert process is not None and process.is_alive()

        assert sidecar._parent_conn is not None
        sidecar._parent_conn.close()

        process.join(timeout=5)
        assert not process.is_alive()

    def test_subprocess_death_fails_pending_futures(self):
        """When the subprocess is killed, pending futures get SidecarDiedError."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(delay=0.5)))
        future = handle.submit_rpc(_AddFutureRPC(a=1, b=2))

        assert sidecar._process is not None
        sidecar._process.kill()
        sidecar._process.join(timeout=5)

        with pytest.raises((SidecarDiedError, SidecarIPCError)):
            future.result(timeout=5)

    def test_submit_after_subprocess_death_raises(self):
        """_submit_rpc() after subprocess death raises immediately instead of hanging."""
        sidecar = SubprocessSidecar()
        assert sidecar._process is not None
        sidecar._process.kill()
        sidecar._process.join(timeout=5)

        with pytest.raises(SidecarDiedError, match="not running.*exit code"):
            sidecar._submit_rpc(_AddRPC(a=1, b=2))

    def test_sidecar_not_picklable(self):
        """SubprocessSidecar cannot be pickled."""
        sidecar = SubprocessSidecar()
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(sidecar)

    def test_handle_not_picklable(self):
        """SidecarHandle cannot be pickled."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        with pytest.raises(TypeError, match="SidecarHandle cannot be pickled"):
            pickle.dumps(handle)

    def test_singleton_returns_same_instance(self):
        """_get_sidecar() returns the same instance on consecutive calls."""
        s1 = _get_sidecar()
        s2 = _get_sidecar()
        assert s1 is s2
        s1._shutdown()

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_singleton_recreates_after_death(self):
        """_get_sidecar() creates a new sidecar if the previous one died."""
        s1 = _get_sidecar()
        s1._shutdown()
        s2 = _get_sidecar()
        assert s2 is not s1
        assert s2._process is not None and s2._process.is_alive()
        s2._shutdown()

    def test_create_sidecar_handle_public_api(self):
        """create_sidecar_handle() registers a target and returns a working handle."""
        handle = create_sidecar_handle(_Calculator())
        assert handle.submit_rpc(_AddRPC(a=5, b=6)).result(timeout=10) == 11


class TestMultiTarget:
    """Multi-target isolation and GC cleanup."""

    def test_two_targets_isolated(self):
        """Two handles on the same sidecar access different target objects."""
        sidecar = SubprocessSidecar()
        h1 = sidecar.register_target(pickle.dumps(_Calculator()))
        h2 = sidecar.register_target(pickle.dumps(_Multiplier(factor=5)))

        assert h1.submit_rpc(_AddRPC(a=3, b=4)).result(timeout=10) == 7
        assert h2.submit_rpc(_ScaleRPC(x=6)).result(timeout=10) == 30

    def test_handle_gc_unregisters_target(self):
        """Deleting a handle sends an unregister RPC (fire-and-forget)."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        target_id = handle._target_id

        assert handle.submit_rpc(_AddRPC(a=1, b=1)).result(timeout=10) == 2

        del handle
        gc.collect()
        time.sleep(0.5)

        with pytest.raises(RuntimeError, match="not registered"):
            sidecar._submit_rpc(_AddRPC(a=1, b=1), target_id=target_id).result(timeout=10)


class TestConcurrency:
    """Thread safety, concurrent submit/shutdown, and stress tests."""

    def test_multithreaded_submits(self):
        """submit_rpc() from 20 threads produces unique request IDs and all resolve."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(delay=0.01)))
        results: list[int | None] = [None] * 20
        errors: list[Exception] = []

        def _worker(idx: int) -> None:
            try:
                results[idx] = handle.submit_rpc(_AddFutureRPC(a=idx, b=idx)).result(timeout=30)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised: {errors}"
        for i, r in enumerate(results):
            assert r == i + i

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    def test_concurrent_submit_and_shutdown(self):
        """submit_rpc() from threads while _shutdown() is called doesn't hang or crash."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(delay=0.01)))
        errors: list[Exception] = []
        barrier = threading.Barrier(6)

        def _submitter() -> None:
            barrier.wait()
            for _ in range(10):
                try:
                    handle.submit_rpc(_AddFutureRPC(a=1, b=2)).result(timeout=5)
                except SidecarDiedError:
                    break
                except Exception as e:
                    errors.append(e)
                    break

        def _shutdowner() -> None:
            barrier.wait()
            time.sleep(0.02)
            sidecar._shutdown()

        threads = [threading.Thread(target=_submitter) for _ in range(5)]
        threads.append(threading.Thread(target=_shutdowner))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised unexpected errors: {errors}"

    def test_cancelled_future_does_not_crash_collector(self):
        """Cancelling a future doesn't kill the collector thread."""
        sidecar = SubprocessSidecar()
        handle = sidecar.register_target(pickle.dumps(_Calculator(delay=0.5)))

        future1 = handle.submit_rpc(_AddFutureRPC(a=1, b=2))
        future1.cancel()

        assert handle.submit_rpc(_AddFutureRPC(a=3, b=4)).result(timeout=10) == 7

    def test_concurrent_registration_with_rpcs(self):
        """RPCs on existing targets work while new targets are being registered."""
        sidecar = SubprocessSidecar()
        calc_handle = sidecar.register_target(pickle.dumps(_Calculator()))
        errors: list[Exception] = []

        def _submit_rpcs() -> None:
            for i in range(20):
                try:
                    assert calc_handle.submit_rpc(_AddRPC(a=i, b=i)).result(timeout=10) == i + i
                except Exception as e:
                    errors.append(e)
                    break

        def _register_targets() -> None:
            for _ in range(5):
                try:
                    h = sidecar.register_target(pickle.dumps(_LoopAwareTarget()))
                    info = h.submit_rpc(_WaitTaskRPC()).result(timeout=10)
                    assert info["task_completed"] is True
                except Exception as e:
                    errors.append(e)
                    break

        t1 = threading.Thread(target=_submit_rpcs)
        t2 = threading.Thread(target=_register_targets)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors: {errors}"

    def test_many_targets_many_threads(self):
        """20 threads each register a target and submit 10 RPCs."""
        sidecar = SubprocessSidecar()
        errors: list[Exception] = []

        def _worker(idx: int) -> None:
            try:
                handle = sidecar.register_target(pickle.dumps(_Calculator()))
                for j in range(10):
                    assert handle.submit_rpc(_AddRPC(a=idx, b=j)).result(timeout=10) == idx + j
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Errors: {errors}"

    def test_rapid_register_unregister_cycles(self):
        """50 rapid register/unregister cycles — sidecar stays healthy."""
        sidecar = SubprocessSidecar()
        for _ in range(50):
            handle = sidecar.register_target(pickle.dumps(_Calculator()))
            assert handle.submit_rpc(_AddRPC(a=1, b=2)).result(timeout=10) == 3
            del handle

        gc.collect()

        handle = sidecar.register_target(pickle.dumps(_Calculator()))
        assert handle.submit_rpc(_AddRPC(a=10, b=20)).result(timeout=10) == 30

    def test_mixed_targets_under_load(self):
        """3 target types x 3 threads x 50 RPCs each, all concurrent."""
        sidecar = SubprocessSidecar()
        calc_handle = sidecar.register_target(pickle.dumps(_Calculator()))
        mult_handle = sidecar.register_target(pickle.dumps(_Multiplier(factor=7)))
        loop_handle = sidecar.register_target(pickle.dumps(_LoopAwareTarget()))
        errors: list[Exception] = []

        def _calc_worker() -> None:
            try:
                for i in range(50):
                    assert calc_handle.submit_rpc(_AddRPC(a=i, b=i)).result(timeout=10) == i + i
            except Exception as e:
                errors.append(e)

        def _mult_worker() -> None:
            try:
                for i in range(50):
                    assert mult_handle.submit_rpc(_ScaleRPC(x=i)).result(timeout=10) == i * 7
            except Exception as e:
                errors.append(e)

        def _loop_worker() -> None:
            try:
                for _ in range(50):
                    info = loop_handle.submit_rpc(_GetInfoRPC()).result(timeout=10)
                    assert info["on_loop"] is True
            except Exception as e:
                errors.append(e)

        threads = (
            [threading.Thread(target=_calc_worker) for _ in range(3)]
            + [threading.Thread(target=_mult_worker) for _ in range(3)]
            + [threading.Thread(target=_loop_worker) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Errors: {errors}"

    def test_multi_target_concurrent_from_threads(self):
        """Multiple handles used concurrently from different threads."""
        sidecar = SubprocessSidecar()
        calc_handle = sidecar.register_target(pickle.dumps(_Calculator()))
        mult_handle = sidecar.register_target(pickle.dumps(_Multiplier(factor=3)))
        results: dict[str, list[int | None]] = {"calc": [None] * 10, "mult": [None] * 10}
        errors: list[Exception] = []

        def _calc_worker(idx: int) -> None:
            try:
                results["calc"][idx] = calc_handle.submit_rpc(_AddRPC(a=idx, b=idx)).result(
                    timeout=10
                )
            except Exception as e:
                errors.append(e)

        def _mult_worker(idx: int) -> None:
            try:
                results["mult"][idx] = mult_handle.submit_rpc(_ScaleRPC(x=idx)).result(timeout=10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_calc_worker, args=(i,)) for i in range(10)]
        threads += [threading.Thread(target=_mult_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised: {errors}"
        for i in range(10):
            assert results["calc"][i] == i + i
            assert results["mult"][i] == i * 3
