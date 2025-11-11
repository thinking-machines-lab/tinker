from __future__ import annotations

import httpx

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from ..types.create_model_request import CreateModelRequest
from ..types.get_info_request import GetInfoRequest
from ..types.get_info_response import GetInfoResponse
from ..types.shared.untyped_api_future import UntypedAPIFuture
from ..types.unload_model_request import UnloadModelRequest

__all__ = ["AsyncModelsResource"]


class AsyncModelsResource(AsyncAPIResource):
    async def create(
        self,
        *,
        request: CreateModelRequest,
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
        Creates a new model.

        Pass a LoRA config to create a new LoRA adapter for the
        base model.

        Args:
          request: The create model request containing base_model, user_metadata, and lora_config

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
            "/api/v1/create_model",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def get_info(
        self,
        *,
        request: GetInfoRequest,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> GetInfoResponse:
        """
        Retrieves information about the current model

        Args:
          request: The get info request containing model_id

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
            "/api/v1/get_info",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=GetInfoResponse,
        )

    async def unload(
        self,
        *,
        request: UnloadModelRequest,
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
        Unload the model weights and ends the user's session.

        Args:
          request: The unload model request containing model_id

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
            "/api/v1/unload_model",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )
