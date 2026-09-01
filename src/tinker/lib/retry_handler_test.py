"""Tests for RetryHandler concurrency limiting."""

from __future__ import annotations

import asyncio

import pytest

from tinker.lib.retry_handler import RetryConfig, RetryHandler


@pytest.mark.asyncio
async def test_max_connections_none_does_not_gate() -> None:
    """With max_connections=None, arbitrarily many tasks run at once."""
    handler = RetryHandler(RetryConfig(max_connections=None, enable_stuck_detection=False))
    all_three_running = asyncio.Event()
    release = asyncio.Event()
    running = 0

    async def work() -> None:
        nonlocal running
        running += 1
        if running == 3:
            all_three_running.set()
        await release.wait()

    tasks = [asyncio.create_task(handler.execute(work)) for _ in range(3)]
    await asyncio.wait_for(all_three_running.wait(), timeout=5)
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_max_connections_gates_concurrency() -> None:
    """An integer max_connections gates concurrency to that value."""
    handler = RetryHandler(RetryConfig(max_connections=1, enable_stuck_detection=False))
    release = asyncio.Event()
    running = 0
    peak = 0

    async def work() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1

    tasks = [asyncio.create_task(handler.execute(work)) for _ in range(3)]
    await asyncio.sleep(0.2)
    assert peak == 1
    release.set()
    await asyncio.gather(*tasks)
