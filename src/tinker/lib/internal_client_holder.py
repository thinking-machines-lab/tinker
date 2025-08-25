"""Internal client holder for managing AsyncTinker clients."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Generator
from contextlib import AbstractContextManager
from typing import Any, TypeVar

import tinker
from tinker.lib.async_tinker_provider import AsyncTinkerProvider
from tinker.lib.telemetry import Telemetry, TelemetryProvider, init_telemetry

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_REQUESTS_PER_HTTPX_CLIENT = 100


class InternalClientHolder(AsyncTinkerProvider, TelemetryProvider):
    def __init__(self, **kwargs: Any) -> None:
        self._constructor_kwargs = kwargs
        # So we can use async eventloop for parallel sampling requests
        # in sync code.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started: threading.Event = threading.Event()
        self._lifecycle_lock: threading.Lock = threading.Lock()
        self._clients: list[tinker.AsyncTinker] = []
        self._client_active_refcount: list[int] = []
        self._sample_backoff_until: float | None = None
        self._sample_dispatch_semaphore: asyncio.Semaphore = asyncio.Semaphore(200)
        self._telemetry: Telemetry | None = init_telemetry(self)

    def aclient(self) -> AbstractContextManager[tinker.AsyncTinker]:
        from contextlib import contextmanager

        @contextmanager
        def _aclient() -> Generator[tinker.AsyncTinker, None, None]:
            assert _current_loop() is self.get_loop(), (
                "AsyncTinker client called from incorrect event loop"
            )
            client_idx = -1
            for i, ref_count in enumerate(self._client_active_refcount):
                if ref_count < MAX_REQUESTS_PER_HTTPX_CLIENT:
                    client_idx = i
                    break
            if client_idx == -1:
                self._clients.append(tinker.AsyncTinker(**self._constructor_kwargs))
                client_idx = len(self._clients) - 1
                self._client_active_refcount.append(1)

            self._client_active_refcount[client_idx] += 1
            try:
                yield self._clients[client_idx]
            finally:
                self._client_active_refcount[client_idx] -= 1

        return _aclient()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._lifecycle_lock:
            if self._loop is None:
                self._start_background_thread()
            assert self._loop is not None, "Background thread not started"
            return self._loop

    def get_telemetry(self) -> Telemetry | None:
        return self._telemetry

    def _start_background_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
        self._thread.start()
        self._started.wait()

    def _background_thread_func(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def close(self):
        with self._lifecycle_lock:
            if telemetry := self._telemetry:
                telemetry.stop()
            if self._thread is not None:
                if self._loop and self._loop.is_running():
                    _ = self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    logger.warning("Background thread did not join")
                self._thread = None
            if self._loop is not None:
                self._loop.close()
                self._loop = None

    def __del__(self):
        self.close()


def _current_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
