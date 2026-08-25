from __future__ import annotations

from typing import cast

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Headers, NotGiven
from ..types.sample_request import SampleRequest
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncSamplingResource"]


class AsyncSamplingResource(AsyncAPIResource):
    async def asample(
        self,
        *,
        request: SampleRequest,
        extra_headers: Headers | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Generates samples from the model using the specified sampling parameters

        Args:
          request: The sample request containing prompt, sampling params, and options

          extra_headers: Send extra headers
        """
        options = make_request_options(extra_headers=extra_headers)
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = cast(int, max_retries)

        return await self._post(
            "/api/v1/asample",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )
