"""Internal client holder for managing AsyncTinker clients."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import tinker

from .public_interfaces.api_future import AwaitableConcurrentFuture

logger = logging.getLogger(__name__)

MAX_REQUESTS_PER_HTTPX_CLIENT = 100


class InternalClientHolder:
    def __init__(self, **kwargs: Any) -> None:
        self._constructor_kwargs = kwargs
        # So we can use async eventloop for parallel sampling requests
        # in sync code.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._clients: list[tinker.AsyncTinker] = []
        self._client_active_refcount: list[int] = []
        self._sample_backoff_until: float | None = None
        self._sample_dispatch_semaphore: asyncio.Semaphore = asyncio.Semaphore(50)

    def aclient(self):
        from contextlib import contextmanager

        @contextmanager
        def _aclient():
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

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._start_background_thread()
        assert self._loop is not None, "Background thread not started"
        return self._loop

    def run_coroutine_threadsafe(self, coro: Any) -> AwaitableConcurrentFuture:
        return AwaitableConcurrentFuture(asyncio.run_coroutine_threadsafe(coro, self._get_loop()))

    def _start_background_thread(self):
        assert self._thread is None, "Background thread already started"
        self._thread = threading.Thread(target=self._background_thread_func, daemon=True)
        self._thread.start()
        self._started.wait()

    def _background_thread_func(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def close(self):
        if self._thread is not None:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Background thread did not join")
            self._thread = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def __del__(self):
        self.close()
