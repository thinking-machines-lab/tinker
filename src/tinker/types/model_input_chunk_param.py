# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias

from .encoded_text_chunk_param import EncodedTextChunkParam
from .image_asset_pointer_chunk_param import ImageAssetPointerChunkParam

__all__ = ["ModelInputChunkParam"]

ModelInputChunkParam: TypeAlias = Union[EncodedTextChunkParam, ImageAssetPointerChunkParam]
