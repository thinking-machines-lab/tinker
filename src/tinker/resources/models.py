from __future__ import annotations

from collections.abc import Mapping

from .._base_client import make_request_options
from .._compat import model_copy, model_dump
from .._resource import AsyncAPIResource
from .._types import Headers
from ..types.copy_weights_request import CopyWeightsRequest
from ..types.copy_weights_response import CopyWeightsResponse
from ..types.create_model_request import CreateModelRequest
from ..types.get_info_request import GetInfoRequest
from ..types.get_info_response import GetInfoResponse
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncModelsResource"]


class AsyncModelsResource(AsyncAPIResource):
    async def create(
        self,
        *,
        request: CreateModelRequest,
    ) -> UntypedAPIFuture:
        """
        Creates a new model.

        Pass a LoRA config to create a new LoRA adapter for the
        base model.

        Args:
          request: The create model request containing base_model, user_metadata, and lora_config
        """
        return await self._post(
            "/api/v1/create_model",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            cast_to=UntypedAPIFuture,
        )

    async def copy_weights(
        self,
        *,
        request: CopyWeightsRequest,
        extra_headers: Headers | None = None,
        extra_headers_secret: Mapping[str, str] | None = None,
    ) -> CopyWeightsResponse:
        """
        Copies weights into the caller's project

        Args:
          request: The copy request containing source_path and either destination_project_id
            or session_id and model_seq_id
          extra_headers: Additional headers, including the idempotency key
          extra_headers_secret: Headers excluded from request-options logging
        """
        return await self._post(
            "/api/v1/copy_weights",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            cast_to=CopyWeightsResponse,
            options=make_request_options(extra_headers=extra_headers),
            extra_headers_secret=extra_headers_secret,
        )

    async def get_info(
        self,
        *,
        request: GetInfoRequest,
    ) -> GetInfoResponse:
        """
        Retrieves information about the current model

        Args:
          request: The get info request containing model_id
        """
        result = await self._post(
            "/api/v1/get_info",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            cast_to=GetInfoResponse,
        )
        if result.model_data.tokenizer_id:
            tokenizer_id = result.model_data.tokenizer_id.split(":")[0]
            updated_model_data = model_copy(
                result.model_data, update={"tokenizer_id": tokenizer_id}
            )
            result = model_copy(result, update={"model_data": updated_model_data})
        return result
