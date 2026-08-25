from __future__ import annotations

import asyncio

import zstandard as zstd

from .._base_client import make_request_options
from .._compat import model_dump
from .._resource import AsyncAPIResource
from ..proto.request_conv import forward_backward_request_to_proto
from ..types.forward_backward_request import ForwardBackwardRequest
from ..types.optim_step_request import OptimStepRequest
from ..types.shared.untyped_api_future import UntypedAPIFuture

__all__ = ["AsyncTrainingResource"]

_PROTO_CONTENT_TYPE = "application/x-protobuf"


class AsyncTrainingResource(AsyncAPIResource):
    async def forward_backward(
        self,
        *,
        request: ForwardBackwardRequest,
        forward_only: bool = False,
    ) -> UntypedAPIFuture:
        """
        Performs a forward and backward pass through the model

        Args:
          request: The forward backward request containing input data, model_id, and seq_id

          forward_only: When true, only the forward pass runs (no backward / gradient
            accumulation).
        """
        proto_msg = forward_backward_request_to_proto(request)
        proto_msg.forward_only = forward_only
        body: bytes = proto_msg.SerializeToString()
        headers: dict[str, str] = {"Content-Type": _PROTO_CONTENT_TYPE}
        if self._client._client_config.proto_compress_fwdbwd:
            body = await asyncio.to_thread(zstd.ZstdCompressor().compress, body)
            headers["Content-Encoding"] = "zstd"

        return await self._post(
            "/api/v1/forward_backward",
            body=body,
            options=make_request_options(extra_headers=headers),
            cast_to=UntypedAPIFuture,
        )

    async def optim_step(
        self,
        *,
        request: OptimStepRequest,
    ) -> UntypedAPIFuture:
        """
        Performs an optimization step to update model parameters

        Args:
          request: The optimization step request containing adam_params, model_id, and seq_id
        """
        return await self._post(
            "/api/v1/optim_step",
            body=model_dump(request, exclude_unset=False, exclude_none=True, mode="json"),
            cast_to=UntypedAPIFuture,
        )
