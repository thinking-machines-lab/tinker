from __future__ import annotations

import datetime

import httpx

from .._base_client import make_request_options
from .._compat import model_dump
from .._exceptions import APIStatusError
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Body, Headers, NoneType, NotGiven, Query
from ..types import CheckpointArchiveUrlResponse, CheckpointsListResponse, ModelID
from ..types.load_weights_request import LoadWeightsRequest
from ..types.save_weights_for_sampler_request import SaveWeightsForSamplerRequest
from ..types.save_weights_request import SaveWeightsRequest
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncWeightsResource"]


class AsyncWeightsResource(AsyncAPIResource):
    async def load(
        self,
        *,
        request: LoadWeightsRequest,
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
        Loads model weights from disk

        Args:
          request: The load weights request containing model_id, path, and seq_id

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
            "/api/v1/load_weights",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def save(
        self,
        *,
        request: SaveWeightsRequest,
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
        Saves model weights to disk

        Args:
          request: The save weights request containing model_id, path, and seq_id

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
            "/api/v1/save_weights",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def save_for_sampler(
        self,
        *,
        request: SaveWeightsForSamplerRequest,
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
        Saves model weights for sampler

        Args:
          request: The save weights for sampler request containing model_id, path, and seq_id

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
            "/api/v1/save_weights_for_sampler",
            body=model_dump(request, exclude_unset=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def list(
        self,
        model_id: ModelID,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CheckpointsListResponse:
        """
        Lists available model checkpoints (both training and sampler)

        Args:
          model_id: The model ID to list checkpoints for

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        return await self._get(
            f"/api/v1/training_runs/{model_id}/checkpoints",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=CheckpointsListResponse,
        )

    async def delete_checkpoint(
        self,
        *,
        model_id: ModelID,
        checkpoint_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> None:
        """Delete a checkpoint for the given training run."""
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        if not checkpoint_id:
            raise ValueError(
                f"Expected a non-empty value for `checkpoint_id` but received {checkpoint_id!r}"
            )

        await self._delete(
            f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=NoneType,
        )

        return None

    async def get_checkpoint_archive_url(
        self,
        *,
        model_id: ModelID,
        checkpoint_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> CheckpointArchiveUrlResponse:
        """
        Get signed URL to download checkpoint archive.

        Args:
          model_id: The training run ID to download weights for
          checkpoint_id: The checkpoint ID to download

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        if not checkpoint_id:
            raise ValueError(
                f"Expected a non-empty value for `checkpoint_id` but received {checkpoint_id!r}"
            )

        from .._response import APIResponse

        # Merge the accept header
        merged_headers: Headers = {"accept": "application/gzip"}
        if extra_headers is not None:
            merged_headers = {**merged_headers, **extra_headers}

        options = make_request_options(
            extra_headers=merged_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        options["follow_redirects"] = False

        try:
            response = await self._get(
                f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive",
                cast_to=APIResponse,
                options=options,
            )
        except APIStatusError as e:
            # On success, this API responds with a 302
            if e.status_code != 302:
                raise e

            location = e.response.headers.get("Location")
            if location is None:
                raise e

            expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=15)
            try:
                if expires_header := e.response.headers.get("Expires"):
                    expires = datetime.datetime.strptime(
                        expires_header, "%a, %d %b %Y %H:%M:%S GMT"
                    )
            except ValueError:
                pass

            return CheckpointArchiveUrlResponse(
                url=location,
                expires=expires,
            )

        # If we did not get an exception we should have gotten a redirect...
        raise RuntimeError("Expected a redirect response, got: " + str(response))
