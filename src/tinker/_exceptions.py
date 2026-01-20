from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

__all__ = [
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "RequestFailedError",
]

if TYPE_CHECKING:
    from tinker.types import RequestErrorCategory


class TinkerError(Exception):
    """Base exception for all Tinker-related errors."""

    pass


class APIError(TinkerError):
    """Base class for all API-related errors."""

    message: str
    request: httpx.Request

    body: object | None
    """The API response body.

    If the API responded with a valid JSON structure then this property will be the
    decoded result.

    If it isn't a valid JSON structure then this will be the raw response.

    If there was no response associated with this error then it will be `None`.
    """

    def __init__(self, message: str, request: httpx.Request, *, body: object | None) -> None:  # noqa: ARG002
        super().__init__(message)
        self.request = request
        self.message = message
        self.body = body


class APIResponseValidationError(APIError):
    """Raised when API response doesn't match expected schema."""

    response: httpx.Response
    status_code: int

    def __init__(
        self, response: httpx.Response, body: object | None, message: str | None = None
    ) -> None:
        super().__init__(
            message or "Data returned by API invalid for expected schema.",
            response.request,
            body=body,
        )
        self.response = response
        self.status_code = response.status_code

    def __reduce__(self):
        # Return a tuple of (callable, args) to recreate the exception
        return (
            self.__class__,
            (self.response, self.body, self.message),  # positional args
            None,
        )


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: httpx.Response
    status_code: int

    def __init__(self, message: str, response: httpx.Response, body: object | None) -> None:
        super().__init__(message, response.request, body=body)
        self.response = response
        self.status_code = response.status_code

    def __reduce__(self):
        # Return a tuple of (callable, args) to recreate the exception
        return (
            self.__class__,
            (self.message, self.response, self.body),  # positional args
            None,
        )


class APIConnectionError(APIError):
    """Raised when a connection error occurs while making an API request."""

    def __init__(self, request: httpx.Request, message: str = "Connection error.") -> None:
        super().__init__(message, request, body=None)

    def __reduce__(self):
        # Return a tuple of (callable, args) to recreate the exception
        return (
            self.__class__,
            (self.request, self.message),  # positional args
            None,
        )


class APITimeoutError(APIConnectionError):
    """Raised when an API request times out."""

    def __init__(self, request: httpx.Request) -> None:
        super().__init__(request=request, message="Request timed out.")


class BadRequestError(APIStatusError):
    """HTTP 400: The request was invalid or malformed."""

    status_code: int = 400


class AuthenticationError(APIStatusError):
    """HTTP 401: Authentication credentials are missing or invalid."""

    status_code: int = 401


class PermissionDeniedError(APIStatusError):
    """HTTP 403: Insufficient permissions to access the resource."""

    status_code: int = 403


class NotFoundError(APIStatusError):
    """HTTP 404: The requested resource was not found."""

    status_code: int = 404


class ConflictError(APIStatusError):
    """HTTP 409: The request conflicts with the current state of the resource."""

    status_code: int = 409


class UnprocessableEntityError(APIStatusError):
    """HTTP 422: The request was well-formed but contains semantic errors."""

    status_code: int = 422


class RateLimitError(APIStatusError):
    """HTTP 429: Too many requests, rate limit exceeded."""

    status_code: int = 429


class InternalServerError(APIStatusError):
    """HTTP 500+: An error occurred on the server."""

    pass


class RequestFailedError(TinkerError):
    """Raised when an asynchronous request completes in a failed state."""

    def __init__(
        self,
        message: str,
        request_id: str,
        category: "RequestErrorCategory",
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.request_id: str = request_id
        self.category: RequestErrorCategory = category

    def __reduce__(self):
        # Return a tuple of (callable, args) to recreate the exception
        return (
            self.__class__,
            (self.message, self.request_id, self.category),  # positional args
            None,
        )
