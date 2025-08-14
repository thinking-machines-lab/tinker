# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Union, Mapping
from typing_extensions import Self, override

import httpx

from . import _exceptions
from ._qs import Querystring
from ._types import (
    NOT_GIVEN,
    Omit,
    Timeout,
    NotGiven,
    Transport,
    ProxiesTypes,
    RequestOptions,
)
from ._utils import is_given, get_async_library
from ._compat import cached_property
from ._version import __version__
from ._streaming import Stream as Stream, AsyncStream as AsyncStream
from ._exceptions import TinkerError, APIStatusError
from ._base_client import (
    DEFAULT_MAX_RETRIES,
    SyncAPIClient,
    AsyncAPIClient,
)

if TYPE_CHECKING:
    from .resources import models, futures, service, weights, sampling, training, telemetry
    from .resources.models import ModelsResource, AsyncModelsResource
    from .resources.futures import FuturesResource, AsyncFuturesResource
    from .resources.service import ServiceResource, AsyncServiceResource
    from .resources.weights import WeightsResource, AsyncWeightsResource
    from .resources.sampling import SamplingResource, AsyncSamplingResource
    from .resources.training import TrainingResource, AsyncTrainingResource
    from .resources.telemetry import TelemetryResource, AsyncTelemetryResource

__all__ = ["Timeout", "Transport", "ProxiesTypes", "RequestOptions", "Tinker", "AsyncTinker", "Client", "AsyncClient"]


class Tinker(SyncAPIClient):
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
        # We provide a `DefaultHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
        http_client: httpx.Client | None = None,
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
        """Construct a new synchronous Tinker client instance.

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
            base_url = f"https://tinker.thinkingmachines.dev/services/tinker-prod"

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
    def service(self) -> ServiceResource:
        from .resources.service import ServiceResource

        return ServiceResource(self)

    @cached_property
    def training(self) -> TrainingResource:
        from .resources.training import TrainingResource

        return TrainingResource(self)

    @cached_property
    def models(self) -> ModelsResource:
        from .resources.models import ModelsResource

        return ModelsResource(self)

    @cached_property
    def weights(self) -> WeightsResource:
        from .resources.weights import WeightsResource

        return WeightsResource(self)

    @cached_property
    def sampling(self) -> SamplingResource:
        from .resources.sampling import SamplingResource

        return SamplingResource(self)

    @cached_property
    def futures(self) -> FuturesResource:
        from .resources.futures import FuturesResource

        return FuturesResource(self)

    @cached_property
    def telemetry(self) -> TelemetryResource:
        from .resources.telemetry import TelemetryResource

        return TelemetryResource(self)

    @cached_property
    def with_raw_response(self) -> TinkerWithRawResponse:
        return TinkerWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> TinkerWithStreamedResponse:
        return TinkerWithStreamedResponse(self)

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
            "X-Stainless-Async": "false",
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        http_client: httpx.Client | None = None,
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
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

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
            base_url = f"https://tinker.thinkingmachines.dev/services/tinker-prod"

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
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

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


class TinkerWithRawResponse:
    _client: Tinker

    def __init__(self, client: Tinker) -> None:
        self._client = client

    @cached_property
    def service(self) -> service.ServiceResourceWithRawResponse:
        from .resources.service import ServiceResourceWithRawResponse

        return ServiceResourceWithRawResponse(self._client.service)

    @cached_property
    def training(self) -> training.TrainingResourceWithRawResponse:
        from .resources.training import TrainingResourceWithRawResponse

        return TrainingResourceWithRawResponse(self._client.training)

    @cached_property
    def models(self) -> models.ModelsResourceWithRawResponse:
        from .resources.models import ModelsResourceWithRawResponse

        return ModelsResourceWithRawResponse(self._client.models)

    @cached_property
    def weights(self) -> weights.WeightsResourceWithRawResponse:
        from .resources.weights import WeightsResourceWithRawResponse

        return WeightsResourceWithRawResponse(self._client.weights)

    @cached_property
    def sampling(self) -> sampling.SamplingResourceWithRawResponse:
        from .resources.sampling import SamplingResourceWithRawResponse

        return SamplingResourceWithRawResponse(self._client.sampling)

    @cached_property
    def futures(self) -> futures.FuturesResourceWithRawResponse:
        from .resources.futures import FuturesResourceWithRawResponse

        return FuturesResourceWithRawResponse(self._client.futures)

    @cached_property
    def telemetry(self) -> telemetry.TelemetryResourceWithRawResponse:
        from .resources.telemetry import TelemetryResourceWithRawResponse

        return TelemetryResourceWithRawResponse(self._client.telemetry)


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


class TinkerWithStreamedResponse:
    _client: Tinker

    def __init__(self, client: Tinker) -> None:
        self._client = client

    @cached_property
    def service(self) -> service.ServiceResourceWithStreamingResponse:
        from .resources.service import ServiceResourceWithStreamingResponse

        return ServiceResourceWithStreamingResponse(self._client.service)

    @cached_property
    def training(self) -> training.TrainingResourceWithStreamingResponse:
        from .resources.training import TrainingResourceWithStreamingResponse

        return TrainingResourceWithStreamingResponse(self._client.training)

    @cached_property
    def models(self) -> models.ModelsResourceWithStreamingResponse:
        from .resources.models import ModelsResourceWithStreamingResponse

        return ModelsResourceWithStreamingResponse(self._client.models)

    @cached_property
    def weights(self) -> weights.WeightsResourceWithStreamingResponse:
        from .resources.weights import WeightsResourceWithStreamingResponse

        return WeightsResourceWithStreamingResponse(self._client.weights)

    @cached_property
    def sampling(self) -> sampling.SamplingResourceWithStreamingResponse:
        from .resources.sampling import SamplingResourceWithStreamingResponse

        return SamplingResourceWithStreamingResponse(self._client.sampling)

    @cached_property
    def futures(self) -> futures.FuturesResourceWithStreamingResponse:
        from .resources.futures import FuturesResourceWithStreamingResponse

        return FuturesResourceWithStreamingResponse(self._client.futures)

    @cached_property
    def telemetry(self) -> telemetry.TelemetryResourceWithStreamingResponse:
        from .resources.telemetry import TelemetryResourceWithStreamingResponse

        return TelemetryResourceWithStreamingResponse(self._client.telemetry)


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


Client = Tinker

AsyncClient = AsyncTinker
