from __future__ import annotations

from .._compat import model_copy, model_dump
from .._resource import AsyncAPIResource
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
