from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from ._client import AsyncTinker


class AsyncAPIResource:
    _client: AsyncTinker

    def __init__(self, client: AsyncTinker) -> None:
        self._client = client
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete
        self._get_api_list = client.get_api_list

    async def _sleep(self, seconds: float) -> None:
        await anyio.sleep(seconds)
