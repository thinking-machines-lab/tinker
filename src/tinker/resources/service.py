from __future__ import annotations

import httpx

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from ..types.auth_token_response import AuthTokenResponse
from ..types.client_config_request import ClientConfigRequest
from ..types.client_config_response import ClientConfigResponse
from ..types.create_sampling_session_request import CreateSamplingSessionRequest
from ..types.create_sampling_session_response import CreateSamplingSessionResponse
from ..types.create_session_request import CreateSessionRequest
from ..types.create_session_response import CreateSessionResponse
from ..types.get_server_capabilities_response import GetServerCapabilitiesResponse
from ..types.health_response import HealthResponse
from ..types.session_heartbeat_request import SessionHeartbeatRequest
from ..types.session_heartbeat_response import SessionHeartbeatResponse

__all__ = ["AsyncServiceResource"]


class AsyncServiceResource(AsyncAPIResource):
    async def get_server_capabilities(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> GetServerCapabilitiesResponse:
        """Retrieves information about supported models and server capabilities"""
        return await self._get(
            "/api/v1/get_server_capabilities",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=GetServerCapabilitiesResponse,
        )

    async def health_check(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> HealthResponse:
        """Checks if the API server is ready"""
        return await self._get(
            "/api/v1/healthz",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=HealthResponse,
        )

    async def auth_token(
        self,
        *,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> AuthTokenResponse:
        """Exchange the current credential for a short-lived JWT."""
        options = make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/auth/token",
            body={},
            options=options,
            cast_to=AuthTokenResponse,
        )

    async def client_config(
        self,
        *,
        request: ClientConfigRequest,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ClientConfigResponse:
        """Fetch server-side feature flags for this client."""
        return await self._post(
            "/api/v1/client/config",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            ),
            cast_to=ClientConfigResponse,
        )

    async def create_session(
        self,
        *,
        request: CreateSessionRequest,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        idempotency_key: str | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> CreateSessionResponse:
        """
        Creates a new session

        Args:
          request: The create session request containing tags

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
            "/api/v1/create_session",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=CreateSessionResponse,
        )

    async def session_heartbeat(
        self,
        *,
        session_id: str,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> SessionHeartbeatResponse:
        """
        Send a heartbeat for an active session to keep it alive

        Args:
          session_id: The ID of the session to heartbeat

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """

        options = make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        request = SessionHeartbeatRequest(session_id=session_id)
        return await self._post(
            "/api/v1/session_heartbeat",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=SessionHeartbeatResponse,
        )

    async def create_sampling_session(
        self,
        *,
        request: CreateSamplingSessionRequest,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> CreateSamplingSessionResponse:
        """
        Creates a new sampling session

        Args:
          request: The create sampling session request containing session_id, sampling_session_seq_id, model_path/base_model

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        options = make_request_options(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/create_sampling_session",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=CreateSamplingSessionResponse,
        )
