from typing import Union
from typing_extensions import Annotated, TypeAlias

from .._utils import PropertyInfo
from .encoded_text_chunk import EncodedTextChunk
from .image_asset_pointer_chunk import ImageAssetPointerChunk
from .image_chunk import ImageChunk

__all__ = ["ModelInputChunk"]

ModelInputChunk: TypeAlias = Annotated[
    Union[EncodedTextChunk, ImageAssetPointerChunk, ImageChunk], PropertyInfo(discriminator="type")
]
