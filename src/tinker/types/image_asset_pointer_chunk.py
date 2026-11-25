from typing_extensions import Literal

from .._models import StrictBase
from .image_dimensions import ImageDimensions

__all__ = ["ImageAssetPointerChunk"]


class ImageAssetPointerChunk(StrictBase):
    """This input type is not currently accepted by the public API.

    Use ImageChunk to provide image bytes instead.
    """

    format: Literal["png", "jpeg"]
    """Image format"""

    location: str
    """Path or URL to the image asset"""

    width: int
    """Physical image width in pixels."""

    height: int
    """Physical image height in pixels."""

    override_dimensions: ImageDimensions | None = None
    """Optional dimensions used for token counting and preprocessing."""

    expected_tokens: int | None = None
    """Expected number of tokens this image represents.
    This is only advisory: the tinker backend will compute the number of tokens
    from the image, and we can fail requests quickly if the tokens does not
    match expected_tokens."""

    type: Literal["image_asset_pointer"] = "image_asset_pointer"

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError(
                "ImageAssetPointerChunk expected_tokens needs to be set in order to compute the length"
            )
        return self.expected_tokens
