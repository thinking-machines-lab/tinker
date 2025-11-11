from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping, Union

import httpx
from typing_extensions import Self, override

from . import _exceptions
from ._base_client import (
    DEFAULT_MAX_RETRIES,
    AsyncAPIClient,
)
from ._compat import cached_property
from ._exceptions import APIStatusError, TinkerError
from ._qs import Querystring
from ._streaming import AsyncStream as AsyncStream
from ._streaming import Stream as Stream
from ._types import (
    NOT_GIVEN,
    NotGiven,
    Omit,
    ProxiesTypes,
    RequestOptions,
    Timeout,
    Transport,
)
from ._utils import get_async_library, is_given
from ._version import __version__

if TYPE_CHECKING:
    from .resources import futures, models, sampling, service, telemetry, training, weights
    from .resources.futures import AsyncFuturesResource
    from .resources.models import AsyncModelsResource
    from .resources.sampling import AsyncSamplingResource
    from .resources.service import AsyncServiceResource
    from .resources.telemetry import AsyncTelemetryResource
    from .resources.training import AsyncTrainingResource
    from .resources.weights import AsyncWeightsResource

__all__ = [
    "Timeout",
    "Transport",
    "ProxiesTypes",
    "RequestOptions",
    "AsyncTinker",
]


class AsyncTinker(AsyncAPIClient):
    # client options
    api_key: str

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultAsyncHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#asyncclient) for more details.
        http_client: httpx.AsyncClient | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:
        """Construct a new async AsyncTinker client instance.

        This automatically infers the `api_key` argument from the `TINKER_API_KEY` environment variable if it is not provided.
        """
        if api_key is None:
            api_key = os.environ.get("TINKER_API_KEY")
        if api_key is None:
            raise TinkerError(
                "The api_key client option must be set either by passing api_key to the client or by setting the TINKER_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("TINKER_BASE_URL")
        if base_url is None:
            base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod"

        super().__init__(
            version=__version__,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

        self._idempotency_header = "X-Idempotency-Key"

    @cached_property
    def service(self) -> AsyncServiceResource:
        from .resources.service import AsyncServiceResource

        return AsyncServiceResource(self)

    @cached_property
    def training(self) -> AsyncTrainingResource:
        from .resources.training import AsyncTrainingResource

        return AsyncTrainingResource(self)

    @cached_property
    def models(self) -> AsyncModelsResource:
        from .resources.models import AsyncModelsResource

        return AsyncModelsResource(self)

    @cached_property
    def weights(self) -> AsyncWeightsResource:
        from .resources.weights import AsyncWeightsResource

        return AsyncWeightsResource(self)

    @cached_property
    def sampling(self) -> AsyncSamplingResource:
        from .resources.sampling import AsyncSamplingResource

        return AsyncSamplingResource(self)

    @cached_property
    def futures(self) -> AsyncFuturesResource:
        from .resources.futures import AsyncFuturesResource

        return AsyncFuturesResource(self)

    @cached_property
    def telemetry(self) -> AsyncTelemetryResource:
        from .resources.telemetry import AsyncTelemetryResource

        return AsyncTelemetryResource(self)

    @cached_property
    def with_raw_response(self) -> AsyncTinkerWithRawResponse:
        return AsyncTinkerWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncTinkerWithStreamedResponse:
        return AsyncTinkerWithStreamedResponse(self)

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        return {"X-API-Key": api_key}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            "X-Stainless-Async": f"async:{get_async_library()}",
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError(
                "The `default_headers` and `set_default_headers` arguments are mutually exclusive"
            )

        if default_query is not None and set_default_query is not None:
            raise ValueError(
                "The `default_query` and `set_default_query` arguments are mutually exclusive"
            )

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=body)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=body)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=body)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=body)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=body)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=body)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=body)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=body)
        return APIStatusError(err_msg, response=response, body=body)


class AsyncTinkerWithRawResponse:
    _client: AsyncTinker

    def __init__(self, client: AsyncTinker) -> None:
        self._client = client

    @cached_property
    def service(self) -> service.AsyncServiceResourceWithRawResponse:
        from .resources.service import AsyncServiceResourceWithRawResponse

        return AsyncServiceResourceWithRawResponse(self._client.service)

    @cached_property
    def training(self) -> training.AsyncTrainingResourceWithRawResponse:
        from .resources.training import AsyncTrainingResourceWithRawResponse

        return AsyncTrainingResourceWithRawResponse(self._client.training)

    @cached_property
    def models(self) -> models.AsyncModelsResourceWithRawResponse:
        from .resources.models import AsyncModelsResourceWithRawResponse

        return AsyncModelsResourceWithRawResponse(self._client.models)

    @cached_property
    def weights(self) -> weights.AsyncWeightsResourceWithRawResponse:
        from .resources.weights import AsyncWeightsResourceWithRawResponse

        return AsyncWeightsResourceWithRawResponse(self._client.weights)

    @cached_property
    def sampling(self) -> sampling.AsyncSamplingResourceWithRawResponse:
        from .resources.sampling import AsyncSamplingResourceWithRawResponse

        return AsyncSamplingResourceWithRawResponse(self._client.sampling)

    @cached_property
    def futures(self) -> futures.AsyncFuturesResourceWithRawResponse:
        from .resources.futures import AsyncFuturesResourceWithRawResponse

        return AsyncFuturesResourceWithRawResponse(self._client.futures)

    @cached_property
    def telemetry(self) -> telemetry.AsyncTelemetryResourceWithRawResponse:
        from .resources.telemetry import AsyncTelemetryResourceWithRawResponse

        return AsyncTelemetryResourceWithRawResponse(self._client.telemetry)


class AsyncTinkerWithStreamedResponse:
    _client: AsyncTinker

    def __init__(self, client: AsyncTinker) -> None:
        self._client = client

    @cached_property
    def service(self) -> service.AsyncServiceResourceWithStreamingResponse:
        from .resources.service import AsyncServiceResourceWithStreamingResponse

        return AsyncServiceResourceWithStreamingResponse(self._client.service)

    @cached_property
    def training(self) -> training.AsyncTrainingResourceWithStreamingResponse:
        from .resources.training import AsyncTrainingResourceWithStreamingResponse

        return AsyncTrainingResourceWithStreamingResponse(self._client.training)

    @cached_property
    def models(self) -> models.AsyncModelsResourceWithStreamingResponse:
        from .resources.models import AsyncModelsResourceWithStreamingResponse

        return AsyncModelsResourceWithStreamingResponse(self._client.models)

    @cached_property
    def weights(self) -> weights.AsyncWeightsResourceWithStreamingResponse:
        from .resources.weights import AsyncWeightsResourceWithStreamingResponse

        return AsyncWeightsResourceWithStreamingResponse(self._client.weights)

    @cached_property
    def sampling(self) -> sampling.AsyncSamplingResourceWithStreamingResponse:
        from .resources.sampling import AsyncSamplingResourceWithStreamingResponse

        return AsyncSamplingResourceWithStreamingResponse(self._client.sampling)

    @cached_property
    def futures(self) -> futures.AsyncFuturesResourceWithStreamingResponse:
        from .resources.futures import AsyncFuturesResourceWithStreamingResponse

        return AsyncFuturesResourceWithStreamingResponse(self._client.futures)

    @cached_property
    def telemetry(self) -> telemetry.AsyncTelemetryResourceWithStreamingResponse:
        from .resources.telemetry import AsyncTelemetryResourceWithStreamingResponse

        return AsyncTelemetryResourceWithStreamingResponse(self._client.telemetry)
