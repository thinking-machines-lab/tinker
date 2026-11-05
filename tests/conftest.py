from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from pytest_asyncio import is_async_test

pytest.register_assert_rewrite("tests.utils")

logging.getLogger("tinker").setLevel(logging.DEBUG)


# automatically add `pytest.mark.asyncio()` to all of our async tests
# so we don't have to add that boilerplate everywhere
def pytest_collection_modifyitems(items: list[pytest.Function]) -> None:
    pytest_asyncio_tests = (item for item in items if is_async_test(item))
    session_scope_marker = pytest.mark.asyncio(loop_scope="session")
    for async_test in pytest_asyncio_tests:
        async_test.add_marker(session_scope_marker, append=False)


@pytest.fixture(params=[True, False], ids=["result", "future-as-client"])
def resolve_created_client(request: pytest.FixtureRequest) -> Iterator[Callable[[Any], Any]]:
    """How a test turns the future a client-creation method returns into the client.

    `True`: `.result()`, the contract. `False`: the future stands in for the client, which
    the SDK keeps working for backwards compatibility; its deprecation warning is silenced
    for the test.
    """
    if request.param:
        yield lambda future: future.result()
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield lambda future: future
