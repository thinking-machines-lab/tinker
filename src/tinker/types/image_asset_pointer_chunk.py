from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["ImageAssetPointerChunk"]


class ImageAssetPointerChunk(StrictBase):
    format: Literal["png", "jpeg"]
    """Image format"""

    height: int
    """Image height in pixels"""

    location: str
    """Path or URL to the image asset"""

    tokens: int
    """Number of tokens this image represents"""

    width: int
    """Image width in pixels"""

    type: Literal["image_asset_pointer"] = "image_asset_pointer"

    @property
    def length(self) -> int:
        return self.tokens
