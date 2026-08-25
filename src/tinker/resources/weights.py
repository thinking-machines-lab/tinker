from __future__ import annotations

import asyncio
import datetime
import email.utils

from .._base_client import make_request_options
from .._compat import model_dump, parse_obj
from .._exceptions import APIStatusError
from .._resource import AsyncAPIResource
from .._types import NOT_GIVEN, Headers, NotGiven
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

    async def list(
        self,
        model_id: ModelID,
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
            cast_to=CheckpointsListResponse,
        )

    async def get_checkpoint_archive_url(
        self,
        *,
        model_id: ModelID,
        checkpoint_id: str,
    ) -> CheckpointArchiveUrlResponse:
        """
        Get signed URL to download checkpoint archive.

        Args:
          model_id: The training run ID to download weights for
          checkpoint_id: The checkpoint ID to download
        """
        if not model_id:
            raise ValueError(f"Expected a non-empty value for `model_id` but received {model_id!r}")
        if not checkpoint_id:
            raise ValueError(
                f"Expected a non-empty value for `checkpoint_id` but received {checkpoint_id!r}"
            )

        merged_headers: Headers = {"accept": "application/json"}
        options = make_request_options(extra_headers=merged_headers)
        options["follow_redirects"] = False

        max_retries = 6
        for retry in range(max_retries):
            try:
                # Accept both the current 302 redirect contract and the future 200 JSON contract.
                response = await self._get(
                    f"/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive",
                    cast_to=object,
                    options=options,
                )
                # If 200 JSON response, parse it into a CheckpointArchiveUrlResponse.
                return parse_obj(CheckpointArchiveUrlResponse, response)
            # If 302 redirect, handle the redirect.
            except APIStatusError as e:
                if e.status_code == 503 and retry < max_retries - 1:
                    await asyncio.sleep(30)
                    continue

                if e.status_code != 302:
                    raise e

                location = e.response.headers.get("Location")
                if location is None:
                    raise e

                expires = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=15)
                try:
                    if expires_header := e.response.headers.get("Expires"):
                        expires = email.utils.parsedate_to_datetime(expires_header)
                        if expires.tzinfo is None:
                            expires = expires.replace(tzinfo=datetime.UTC)
                        expires = expires.astimezone(datetime.UTC)
                except (TypeError, ValueError):
                    pass

                return CheckpointArchiveUrlResponse(
                    url=location,
                    expires=expires,
                )
