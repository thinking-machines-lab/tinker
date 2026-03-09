"""Subprocess sidecar for GIL-isolated RPC execution.

Runs one or more picklable target objects in a dedicated subprocess and routes
typed RPC calls to them via multiprocessing queues.  The subprocess runs a
shared asyncio event loop and self-terminates when the parent process dies.

Usage — multiple RPCs on one target::

    handle = create_sidecar_handle(Calculator())

    @dataclasses.dataclass
    class MultiplyRPC(SidecarRPC):
        a: int
        b: int

        async def execute(self, target: Calculator) -> int:
            return target.multiply(self.a, self.b)

    # Methods that return Futures work too — the sidecar automatically awaits
    # them after execute() returns.
    @dataclasses.dataclass
    class SampleRPC(SidecarRPC):
        prompt: str

        async def execute(self, target: SamplingClient) -> SampleResponse:
            return target.sample(self.prompt)

    assert handle.submit_rpc(MultiplyRPC(a=3, b=4)).result() == 12

Multiple targets on the same sidecar::

    calc_handle = create_sidecar_handle(Calculator())
    db_handle   = create_sidecar_handle(DatabaseClient(uri))

    # Each handle routes RPCs to its own target
    calc_handle.submit_rpc(MultiplyRPC(a=1, b=2))
    db_handle.submit_rpc(QueryRPC(sql="SELECT ..."))
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import logging
import multiprocessing
import multiprocessing.connection
import multiprocessing.process
import pickle
import queue
import threading
import time
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

from tinker._exceptions import SidecarDiedError, SidecarIPCError, SidecarStartupError

logger = logging.getLogger(__name__)

# Use "spawn" context to avoid fork issues with background event loop threads.
# Fork would duplicate threads in a broken state; spawn creates a fresh interpreter.
_mp_context = multiprocessing.get_context("spawn")

# Startup protocol messages
_STARTUP_OK = "__startup_ok__"

# Timeouts (seconds)
_STARTUP_TIMEOUT_SECONDS = 30
_COLLECTOR_POLL_INTERVAL_SECONDS = 1.0

# Maximum number of requests to dequeue per event loop tick. This provides
# natural backpressure: when the event loop is busy (many in-flight requests,
# slow networking), ticks slow down and the worker pulls less work.
_MAX_REQUESTS_PER_TICK = 16


def _close_queue(q: multiprocessing.Queue[Any] | None) -> None:
    """Close a queue without blocking. cancel_join_thread() prevents hangs."""
    if q is None:
        return
    with contextlib.suppress(Exception):
        q.cancel_join_thread()
        q.close()


# ---------------------------------------------------------------------------
# Target registry (subprocess-only, single-threaded event loop)
# ---------------------------------------------------------------------------

_targets: dict[int, Any] = {}
_next_target_id: int = 0


# ---------------------------------------------------------------------------
# RPC protocol
# ---------------------------------------------------------------------------


class SidecarRPC:
    """Base class for sidecar RPC requests.

    Subclasses must implement ``async execute(target)`` which runs directly on
    the subprocess event loop. ``target`` is the object registered via
    ``create_sidecar_handle()``.

    If the return value is a ``ConcurrentFuture`` or awaitable, the sidecar
    automatically awaits it before sending the result back.

    RPC objects must be picklable since they are sent through a
    ``multiprocessing.Queue``.
    """

    async def execute(self, target: Any) -> Any:
        """Execute this RPC on the subprocess event loop.

        If the return value is a ``ConcurrentFuture`` or awaitable, it is
        automatically awaited.

        Args:
            target: The registered target object for this handle.
        """
        raise NotImplementedError


@dataclasses.dataclass
class _RegisterTargetRPC(SidecarRPC):
    """Built-in RPC: register a new target in the subprocess."""

    pickled_target: bytes

    async def execute(self, target: Any) -> int:
        global _next_target_id
        unpickled = pickle.loads(self.pickled_target)
        target_id = _next_target_id
        _next_target_id += 1
        _targets[target_id] = unpickled
        return target_id


@dataclasses.dataclass
class _UnregisterTargetRPC(SidecarRPC):
    """Built-in RPC: remove a target from the subprocess registry."""

    target_id: int

    async def execute(self, target: Any) -> None:
        _targets.pop(self.target_id, None)


# ---------------------------------------------------------------------------
# SidecarHandle
# ---------------------------------------------------------------------------


class SidecarHandle:
    """Handle for submitting RPCs to a specific target in the sidecar.

    Wraps a ``SubprocessSidecar`` and a ``target_id``. Provides the same
    ``submit_rpc()`` API — the handle pairs each RPC with its target ID in the
    wire protocol so the subprocess can resolve the target before calling
    ``execute()``. The RPC object itself is never mutated.

    When the handle is garbage-collected or explicitly deleted,
    the target is automatically unregistered from the subprocess.

    Thread-safe: multiple threads may call ``submit_rpc()`` on the same handle
    and may safely reuse the same RPC instance across calls.

    Not picklable — obtain handles via ``create_sidecar_handle()``
    in the process that owns the sidecar.
    """

    def __init__(self, sidecar: SubprocessSidecar, target_id: int):
        self._sidecar = sidecar
        self._target_id = target_id

    def submit_rpc(self, rpc: SidecarRPC) -> ConcurrentFuture[Any]:
        """Submit an RPC to the handle's target."""
        return self._sidecar._submit_rpc(rpc, target_id=self._target_id)

    def __del__(self) -> None:
        # Auto-unregister target from subprocess (fire-and-forget).
        # Uses bare try/except — not contextlib.suppress — because during
        # interpreter shutdown module globals (including contextlib) can be None.
        try:
            self._sidecar._submit_rpc(_UnregisterTargetRPC(target_id=self._target_id))
        except Exception:
            pass

    def __reduce__(self) -> None:  # type: ignore[override]
        raise TypeError(
            "SidecarHandle cannot be pickled. It holds a reference to a local "
            "SubprocessSidecar. Call create_sidecar_handle() in the target process instead."
        )


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------


# Set to True inside the sidecar subprocess. Checked by callers to avoid
# nesting (daemon processes cannot spawn children).
_inside_sidecar: bool = False


def _subprocess_worker(
    request_queue: multiprocessing.Queue[Any],
    response_queue: multiprocessing.Queue[Any],
    parent_conn: multiprocessing.connection.Connection,
) -> None:
    """Entry point for the sidecar subprocess.

    Creates the event loop manually and injects it into the SDK's
    ``InternalClientHolderThreadSingleton`` **before** any target is unpickled.
    This way heartbeats, rate limiting, and RPCs all share one loop with no
    extra background thread.

    Starts empty (no initial target). Targets are registered via
    ``_RegisterTargetRPC``. Monitors ``parent_conn`` for parent death
    (EOF on pipe -> self-terminate).
    """
    global _inside_sidecar
    _inside_sidecar = True

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from tinker.lib.internal_client_holder import _internal_client_holder_thread_singleton

    _internal_client_holder_thread_singleton._set_loop(loop)

    response_queue.put((_STARTUP_OK, None, None))
    try:
        loop.run_until_complete(_async_worker_main(request_queue, response_queue, parent_conn))
    finally:
        loop.close()


async def _async_worker_main(
    request_queue: multiprocessing.Queue[Any],
    response_queue: multiprocessing.Queue[Any],
    parent_conn: multiprocessing.connection.Connection,
) -> None:
    """Async event loop: dequeue RPCs, dispatch as tasks, monitor parent pipe."""
    loop = asyncio.get_running_loop()
    pending_tasks: set[asyncio.Task[None]] = set()
    shutting_down = False

    # Monitor parent death via a detached daemon thread. Uses blocking poll(None)
    # — when the parent dies, the OS closes the pipe and poll raises EOFError.
    # Sets a threading.Event (not asyncio.Event) so it's safe from any thread.
    parent_gone = threading.Event()

    def _watch_parent() -> None:
        try:
            parent_conn.poll(None)
        except (EOFError, OSError):
            pass
        parent_gone.set()

    watcher = threading.Thread(target=_watch_parent, daemon=True, name="parent-death-watcher")
    watcher.start()

    while not shutting_down:
        # Wait for next request, checking parent liveness between polls.
        raw = None
        while raw is None and not parent_gone.is_set():
            try:
                raw = await loop.run_in_executor(
                    None, lambda: request_queue.get(timeout=_COLLECTOR_POLL_INTERVAL_SECONDS)
                )
            except queue.Empty:
                continue
            except Exception:
                break

        if raw is None or parent_gone.is_set():
            break

        # Wire format: (request_id, target_id | None, rpc).
        # pickle.loads failures are not caught — corrupted payloads crash
        # the worker, and the collector fails all pending futures. This is
        # the correct response: we serialize payloads ourselves, so
        # deserialization failure means the IPC channel is broken.
        batch: list[tuple[int, int | None, SidecarRPC]] = [pickle.loads(raw)]

        # Non-blocking drain of up to N-1 more requests
        for _ in range(_MAX_REQUESTS_PER_TICK - 1):
            try:
                raw = request_queue.get_nowait()
            except Exception:
                break
            if raw is None:
                shutting_down = True
                break
            batch.append(pickle.loads(raw))

        for request_id, target_id, rpc in batch:
            task = asyncio.create_task(_handle_request(request_id, target_id, rpc, response_queue))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        # Yield to let in-flight tasks make progress before pulling more work
        await asyncio.sleep(0)

    for task in pending_tasks:
        task.cancel()


async def _handle_request(
    request_id: int,
    target_id: int | None,
    rpc: SidecarRPC,
    response_queue: multiprocessing.Queue[Any],
) -> None:
    """Resolve the target, run execute() on the event loop, bridge the result."""
    try:
        target = None
        if target_id is not None:
            target = _targets.get(target_id)
            if target is None:
                raise RuntimeError(
                    f"{type(rpc).__name__}: target_id={target_id} is not registered "
                    "(target was never registered or was already unregistered)"
                )

        result = await rpc.execute(target)

        # If execute() returned a Future or awaitable, await it on the event
        # loop.
        if isinstance(result, ConcurrentFuture):
            result = await asyncio.wrap_future(result)
        elif inspect.isawaitable(result):
            result = await result

        _put_response(response_queue, request_id, result, None)
    except Exception as e:
        _put_response(response_queue, request_id, None, e)


def _put_response(
    response_queue: multiprocessing.Queue[Any],
    request_id: int,
    result: Any,
    exception: BaseException | None,
) -> None:
    """Pre-serialize and enqueue a response. Wraps unpicklable exceptions."""
    if exception is not None:
        try:
            pickle.dumps(exception)
        except Exception:
            exception = SidecarIPCError(f"{type(exception).__name__}: {exception}")
    try:
        payload = pickle.dumps((request_id, result, exception))
        response_queue.put(payload)
    except Exception:
        try:
            payload = pickle.dumps(
                (request_id, None, SidecarIPCError("Failed to serialize response"))
            )
            response_queue.put(payload)
        except Exception:
            logger.error(f"Failed to send response for request {request_id} — queue is broken")


# ---------------------------------------------------------------------------
# Response collector thread
# ---------------------------------------------------------------------------


class _ResponseCollector(threading.Thread):
    """Daemon thread that owns all reads from the response queue.

    Handles two phases:

    1. **Startup handshake** — waits for ``_STARTUP_OK`` from the subprocess.
       The parent thread blocks on :meth:`wait_ready` until this completes.
    2. **Response loop** — reads RPC responses and resolves the corresponding
       ``ConcurrentFuture`` objects.

    When the subprocess dies or the queue breaks, all pending futures are
    failed with ``SidecarDiedError``.
    """

    def __init__(
        self,
        response_queue: multiprocessing.Queue[Any],
        pending_futures: dict[int, ConcurrentFuture[Any]],
        pending_lock: threading.Lock,
        process: multiprocessing.process.BaseProcess,
    ):
        super().__init__(daemon=True, name="SubprocessSidecar-ResponseCollector")
        self._response_queue = response_queue
        self._pending_futures = pending_futures
        self._pending_lock = pending_lock
        self._process = process
        self._ready = threading.Event()
        self._startup_error: SidecarStartupError | None = None

    def wait_ready(self) -> None:
        """Block until startup completes or fails.

        Raises:
            SidecarStartupError: If the subprocess dies, sends an unexpected
                message, or fails to start within the timeout.
        """
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def run(self) -> None:
        # Phase 1: startup handshake
        if not self._wait_for_startup():
            self._fail_all_pending(f"Sidecar subprocess failed to start: {self._startup_error}")
            return

        # Phase 2: response loop
        while self._process.is_alive():
            try:
                raw = self._response_queue.get(timeout=_COLLECTOR_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                continue
            except Exception:
                break

            if raw is None:
                break

            try:
                self._resolve(pickle.loads(raw))
            except Exception:
                logger.debug("Failed to deserialize response — skipping")
                continue

        self._drain_queue()

        exitcode = self._process.exitcode
        self._fail_all_pending(f"Sidecar subprocess exited unexpectedly (exit code: {exitcode})")

    def _wait_for_startup(self) -> bool:
        """Wait for ``_STARTUP_OK``. Returns True on success, False on failure.

        Always sets ``_ready`` before returning so ``wait_ready()`` never hangs.
        """
        try:
            return self._wait_for_startup_inner()
        except Exception as e:
            # Queue broken (OSError/ValueError from closed queue) or other
            # unexpected error.  Must still unblock wait_ready().
            self._startup_error = SidecarStartupError(
                f"Startup handshake failed: {type(e).__name__}: {e}"
            )
            return False
        finally:
            self._ready.set()

    def _wait_for_startup_inner(self) -> bool:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not self._process.is_alive():
                self._startup_error = SidecarStartupError(
                    f"Sidecar subprocess died before startup (exit code: {self._process.exitcode})"
                )
                return False
            try:
                tag, _, _ = self._response_queue.get(timeout=_COLLECTOR_POLL_INTERVAL_SECONDS)
            except queue.Empty:
                continue
            if tag == _STARTUP_OK:
                return True
            self._startup_error = SidecarStartupError(f"Unexpected startup message: {tag}")
            return False
        self._startup_error = SidecarStartupError(
            f"Sidecar subprocess failed to start within {_STARTUP_TIMEOUT_SECONDS}s"
        )
        return False

    def _resolve(self, msg: tuple[int, Any, BaseException | None]) -> None:
        request_id, result, exception = msg

        with self._pending_lock:
            future = self._pending_futures.pop(request_id, None)

        if future is None:
            logger.debug(f"Received response for unknown request {request_id}")
            return

        try:
            if exception is not None:
                future.set_exception(exception)
            else:
                future.set_result(result)
        except Exception:
            logger.debug(f"Could not resolve future for request {request_id} (likely cancelled)")

    def _drain_queue(self) -> None:
        """Drain responses that arrived between the last poll and process exit.

        Without this, responses enqueued just before the process dies would
        never be resolved, leaving their futures hanging until timeout.
        """
        while True:
            try:
                raw = self._response_queue.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            if raw is None:
                break
            try:
                self._resolve(pickle.loads(raw))
            except Exception:
                continue

    def _fail_all_pending(self, message: str) -> None:
        with self._pending_lock:
            futures = list(self._pending_futures.values())
            self._pending_futures.clear()
        for future in futures:
            if not future.done():
                # set_exception raises InvalidStateError on a TOCTOU race
                # with concurrent cancellation or resolution.
                with contextlib.suppress(Exception):
                    future.set_exception(SidecarDiedError(message))


# ---------------------------------------------------------------------------
# SubprocessSidecar
# ---------------------------------------------------------------------------


class SubprocessSidecar:
    """Runs picklable objects in a dedicated subprocess and routes RPC calls to them.

    Use ``create_sidecar_handle()`` to get a ``SidecarHandle`` — it manages
    the singleton sidecar internally.

    Cleanup is automatic: ``__del__`` calls ``_shutdown()`` on GC, and the
    parent-death pipe ensures the child self-terminates if the parent is killed.

    If the subprocess dies, pending futures fail with ``SidecarDiedError`` and
    subsequent ``submit_rpc()`` calls raise immediately. Not picklable.
    """

    def __init__(self) -> None:
        # Concurrency state — protected by _pending_lock
        self._request_id_counter: int = 0
        self._pending_futures: dict[int, ConcurrentFuture[Any]] = {}
        self._pending_lock: threading.Lock = threading.Lock()

        # Subprocess state
        self._process: multiprocessing.process.BaseProcess | None = None
        self._request_queue: multiprocessing.Queue[Any] | None = None
        self._response_queue: multiprocessing.Queue[Any] | None = None
        self._parent_conn: multiprocessing.connection.Connection | None = None
        self._collector: _ResponseCollector | None = None

        self._start_subprocess()

    def _start_subprocess(self) -> None:
        """Start the subprocess worker and response collector."""
        self._request_queue = _mp_context.Queue()
        self._response_queue = _mp_context.Queue()
        self._parent_conn, child_conn = _mp_context.Pipe()

        self._process = _mp_context.Process(
            target=_subprocess_worker,
            args=(
                self._request_queue,
                self._response_queue,
                child_conn,
            ),
            daemon=True,
        )
        self._process.start()

        # The collector owns all response_queue reads — including the startup
        # handshake. On failure, clean up so repeated failures don't leak FDs.
        self._collector = _ResponseCollector(
            self._response_queue,
            self._pending_futures,
            self._pending_lock,
            self._process,
        )
        self._collector.start()

        try:
            self._collector.wait_ready()
        except Exception:
            self._shutdown()
            raise

        logger.debug("SubprocessSidecar started (pid=%s)", self._process.pid)

    def register_target(self, pickled_target: bytes) -> SidecarHandle:
        """Register a target in the subprocess. Returns a handle for RPCs.

        Blocks the calling thread until the subprocess confirms registration.
        Do not call from an asyncio event loop — use ``loop.run_in_executor()``
        to avoid blocking the loop.

        Raises:
            SidecarDiedError: If the sidecar subprocess is not running.
            Exception: If the target cannot be unpickled in the subprocess.
        """
        future = self._submit_rpc(_RegisterTargetRPC(pickled_target=pickled_target))
        target_id = future.result()
        logger.debug(
            "Registered target_id=%d (pickled_target %d bytes)", target_id, len(pickled_target)
        )
        return SidecarHandle(self, target_id)

    def _submit_rpc(
        self, rpc: SidecarRPC, *, target_id: int | None = None
    ) -> ConcurrentFuture[Any]:
        """Submit an RPC to the subprocess and return a Future.

        Args:
            rpc: The RPC to execute in the subprocess.
            target_id: The target to resolve before calling ``rpc.execute(target)``.
                ``None`` for built-in RPCs that don't need a target.
        """
        future: ConcurrentFuture[Any] = ConcurrentFuture()

        with self._pending_lock:
            process = self._process
            request_queue = self._request_queue

            if process is None or not process.is_alive():
                exitcode = process.exitcode if process is not None else None
                raise SidecarDiedError(f"Sidecar subprocess is not running (exit code: {exitcode})")

            if request_queue is None:
                raise SidecarDiedError("Sidecar subprocess is not running (exit code: None)")

            request_id = self._request_id_counter
            self._request_id_counter += 1
            self._pending_futures[request_id] = future

        try:
            payload = pickle.dumps((request_id, target_id, rpc))
            request_queue.put(payload)
        except Exception as e:
            with self._pending_lock:
                self._pending_futures.pop(request_id, None)
            # Guard: a concurrent shutdown() may have already resolved this future.
            if not future.done():
                future.set_exception(SidecarIPCError(f"Failed to serialize RPC: {e}"))

        return future

    def _shutdown(self) -> None:
        """Kill the subprocess and release all resources.

        All pending futures are failed with ``SidecarDiedError`` before this
        method returns. Safe to call multiple times. Called by ``__del__``
        on GC and by ``_get_sidecar()`` when replacing a dead sidecar.
        """
        # 1. Kill and reap
        process = self._process
        with contextlib.suppress(Exception):
            if process is not None and process.is_alive():
                process.kill()
            if process is not None:
                process.join(timeout=5)

        # 2. Null references before failing futures (prevents re-entrant submit)
        request_queue = self._request_queue
        response_queue = self._response_queue
        parent_conn = self._parent_conn
        self._process = None
        self._request_queue = None
        self._response_queue = None
        self._parent_conn = None

        # 3. Collect under lock, fail outside (prevents deadlock from callbacks)
        with self._pending_lock:
            futures = list(self._pending_futures.values())
            self._pending_futures.clear()
        for future in futures:
            if not future.done():
                with contextlib.suppress(Exception):
                    future.set_exception(SidecarDiedError("Sidecar subprocess was shut down"))

        # 4. Release OS resources
        _close_queue(request_queue)
        _close_queue(response_queue)
        with contextlib.suppress(Exception):
            if parent_conn is not None:
                parent_conn.close()

    def __reduce__(self) -> None:  # type: ignore[override]
        raise TypeError(
            "SubprocessSidecar cannot be pickled. It manages OS-level resources "
            "(subprocesses, threads, queues) that cannot be transferred across processes."
        )

    def __del__(self) -> None:
        self._shutdown()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_sidecar: SubprocessSidecar | None = None
_global_sidecar_lock = threading.Lock()


def _get_sidecar() -> SubprocessSidecar:
    """Return the module-level shared sidecar, creating it if needed.

    If the previous sidecar's subprocess has died, a new one is created.
    Thread-safe via ``_global_sidecar_lock``.

    Note: existing ``SidecarHandle`` instances from the old sidecar will
    fail with ``SidecarDiedError`` on next use — callers must re-register.
    """
    global _global_sidecar
    with _global_sidecar_lock:
        if (
            _global_sidecar is None
            or _global_sidecar._process is None
            or not _global_sidecar._process.is_alive()
        ):
            old = _global_sidecar
            if old is not None:
                logger.debug("Sidecar subprocess died, cleaning up before replacement")
                old._shutdown()
            _global_sidecar = SubprocessSidecar()
    return _global_sidecar


def create_sidecar_handle(target: Any) -> SidecarHandle:
    """Register a picklable target in the shared sidecar subprocess.

    Pickles ``target``, sends it to the sidecar subprocess, and returns a
    ``SidecarHandle`` for submitting RPCs.  The sidecar singleton is created
    on first call and reused thereafter.

    Blocks the calling thread until registration completes.  Do not call
    from an asyncio event loop — use ``loop.run_in_executor()`` instead.

    Args:
        target: Any picklable object to run in the subprocess.

    Raises:
        RuntimeError: If called from inside the sidecar subprocess.
        SidecarStartupError: If the sidecar subprocess fails to start.
        SidecarDiedError: If the sidecar subprocess is not running.
        Exception: If the target cannot be pickled or unpickled.
    """
    if _inside_sidecar:
        raise RuntimeError(
            "create_sidecar_handle() cannot be called from inside the sidecar subprocess. "
            "Daemon processes cannot spawn children."
        )
    return _get_sidecar().register_target(pickle.dumps(target))
