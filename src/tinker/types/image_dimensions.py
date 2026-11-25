from pydantic import Field

from .._models import StrictBase

__all__ = ["ImageDimensions"]


class ImageDimensions(StrictBase):
    """Image dimensions used in model input requests."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
