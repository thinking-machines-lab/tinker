"""gRPC client interceptors that inject auth metadata on every RPC.

grpc.aio's channel dispatches each interceptor instance to exactly one of
{unary,stream}{Unary,Stream} via an elif isinstance chain, so we expose one
class per RPC kind and a helper that returns the per-kind instance list.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar, cast

import grpc
import grpc.aio

from tinker.lib._auth_token_provider import AuthTokenProvider

_RequestT = TypeVar("_RequestT")
_ResponseT = TypeVar("_ResponseT")


class _AuthBase:
    """Shared state and helper for injecting `x-api-key` into call metadata."""

    def __init__(self, auth: AuthTokenProvider) -> None:
        self._auth = auth

    async def _add_auth(
        self, client_call_details: grpc.aio.ClientCallDetails
    ) -> grpc.aio.ClientCallDetails:
        token = await self._auth.get_token()
        if not token:
            return client_call_details
        # Copy the existing metadata into a fresh Metadata so the caller's
        # instance is never mutated. Prevents duplicate `x-api-key` entries
        # if a future grpc retry policy ever re-invokes the interceptor
        # chain on the same ClientCallDetails.
        md = grpc.aio.Metadata()
        existing = client_call_details.metadata
        if existing is not None:
            # typeshed types Metadata.__iter__ as Iterator[str] (yielding
            # keys), but at runtime each entry is a (key, value) tuple —
            # cast to match runtime.
            for key, value in cast(Iterable[tuple[str, str | bytes]], existing):
                md.add(key, value)
        md.add("x-api-key", token)
        return grpc.aio.ClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=md,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
        )


class _AuthUnaryUnary(_AuthBase, grpc.aio.UnaryUnaryClientInterceptor, grpc.aio.ClientInterceptor):
    async def intercept_unary_unary(
        self,
        continuation: Callable[
            [grpc.aio.ClientCallDetails, _RequestT],
            grpc.aio.UnaryUnaryCall[_RequestT, _ResponseT],
        ],
        client_call_details: grpc.aio.ClientCallDetails,
        request: _RequestT,
    ) -> _ResponseT:
        return await continuation(await self._add_auth(client_call_details), request)


def auth_interceptors(auth: AuthTokenProvider) -> list[grpc.aio.ClientInterceptor]:
    """Return one interceptor per RPC kind, all sharing `auth`.

    Only unary-unary is implemented. Add per-kind classes here when streaming
    RPCs are introduced.
    """
    return [_AuthUnaryUnary(auth)]
