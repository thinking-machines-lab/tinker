from __future__ import annotations

import asyncio

import httpx
import zstandard as zstd

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from ..lib._pydantic_conv import to_pydantic_request
from ..proto.request_conv import forward_backward_request_to_proto
from ..types.forward_backward_request import ForwardBackwardRequest
from ..types.forward_request import ForwardRequest
from ..types.optim_step_request import OptimStepRequest
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncTrainingResource"]

_PROTO_CONTENT_TYPE = "application/x-protobuf"


class AsyncTrainingResource(AsyncAPIResource):
    async def forward(
        self,
        *,
        request: ForwardRequest,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Performs a forward pass through the model

        Args:
          request: The forward request containing input data, model_id, and seq_id

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        options = make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            idempotency_key=idempotency_key,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/forward",
            body=model_dump(
                to_pydantic_request(request),
                exclude_unset=False,
                exclude_none=True,
                mode="json",
            ),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def forward_backward(
        self,
        *,
        request: ForwardBackwardRequest,
        forward_only: bool = False,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Performs a forward and backward pass through the model

        Args:
          request: The forward backward request containing input data, model_id, and seq_id

          forward_only: When true, only the forward pass runs (no backward / gradient
            accumulation). Honored only on the proto path (proto_write_fwdbwd=true);
            silently ignored on the JSON path because the server's Pydantic surface
            hardcodes forward_only=False for /api/v1/forward_backward.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        cfg = self._client._client_config
        headers = dict(extra_headers or {})
        if cfg.proto_write_fwdbwd:
            proto_msg = forward_backward_request_to_proto(request)
            proto_msg.forward_only = forward_only
            body = proto_msg.SerializeToString()
            headers["Content-Type"] = _PROTO_CONTENT_TYPE
            if cfg.proto_compress_fwdbwd:
                body = await asyncio.to_thread(zstd.ZstdCompressor().compress, body)
                headers["Content-Encoding"] = "zstd"
        else:
            body = model_dump(
                to_pydantic_request(request),
                exclude_unset=False,
                exclude_none=True,
                mode="json",
            )

        options = make_request_options(
            extra_headers=headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            idempotency_key=idempotency_key,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries
        return await self._post(
            "/api/v1/forward_backward",
            body=body,
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def optim_step(
        self,
        *,
        request: OptimStepRequest,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Performs an optimization step to update model parameters

        Args:
          request: The optimization step request containing adam_params, model_id, and seq_id

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds

          idempotency_key: Specify a custom idempotency key for this request
        """
        options = make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            idempotency_key=idempotency_key,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/optim_step",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )
