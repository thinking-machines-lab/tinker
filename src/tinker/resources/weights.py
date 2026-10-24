from __future__ import annotations

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Headers, NotGiven
from ..types import (
    CheckpointsListResponse,
    ExternalWeightsUrlsResponse,
    ModelID,
)
from ..types.load_weights_request import LoadWeightsRequest
from ..types.save_weights_external_request import SaveWeightsExternalRequest
from ..types.save_weights_for_sampler_request import SaveWeightsForSamplerRequest
from ..types.save_weights_request import SaveWeightsRequest
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncWeightsResource"]


class AsyncWeightsResource(AsyncAPIResource):
    async def load(
        self,
        *,
        request: LoadWeightsRequest,
    ) -> UntypedAPIFuture:
        """
        Loads model weights from disk

        Args:
          request: The load weights request containing model_id, path, and seq_id
        """
        return await self._post(
            "/api/v1/load_weights",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            cast_to=UntypedAPIFuture,
        )

    async def save(
        self,
        *,
        request: SaveWeightsRequest,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Saves model weights to disk

        Args:
          request: The save weights request containing model_id, path, and seq_id
        """
        options = make_request_options()
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/save_weights",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def save_for_sampler(
        self,
        *,
        request: SaveWeightsForSamplerRequest,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Saves model weights for sampler

        Args:
          request: The save weights for sampler request containing model_id, path, and seq_id
        """
        options = make_request_options()
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/save_weights_for_sampler",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def save_external(
        self,
        *,
        request: SaveWeightsExternalRequest,
        max_retries: int | NotGiven = NOT_GIVEN,
    ) -> UntypedAPIFuture:
        """
        Saves model weights in an external (e.g. HuggingFace) format.

        Args:
          request: The save weights external request containing model_id, path, and seq_id
        """
        options = make_request_options()
        if max_retries is not NOT_GIVEN:
            options["max_retries"] = max_retries

        return await self._post(
            "/api/v1/save_weights_external",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            options=options,
            cast_to=UntypedAPIFuture,
        )

    async def list(
        self,
        model_id: ModelID,
        *,
        extra_headers: Headers | None = None,
    ) -> CheckpointsListResponse:
        """
        Lists available model checkpoints (both training and sampler)

        Args:
          model_id: The model ID to list checkpoints for
        """
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        return await self._get(
            f"/api/v1/training_runs/{model_id}/checkpoints",
            options=make_request_options(extra_headers=extra_headers),
            cast_to=CheckpointsListResponse,
        )

    async def get_external_weights_urls(
        self, *, model_id: ModelID, checkpoint_id: str, extra_headers: Headers | None = None
    ) -> ExternalWeightsUrlsResponse:
        """Signed download URLs, one per file, for an external weights checkpoint."""
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        if not checkpoint_id:
            raise ValueError(
                f"Expected a non-empty value for `checkpoint_id` but received {checkpoint_id!r}"
            )
        return await self._get(
            f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/external_weights_urls",
            options=make_request_options(extra_headers=extra_headers),
            cast_to=ExternalWeightsUrlsResponse,
        )
