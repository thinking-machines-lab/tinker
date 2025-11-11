import base64
from typing import Union

from pydantic import field_serializer, field_validator
from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["ImageChunk"]


class ImageChunk(StrictBase):
    data: bytes
    """Image data as bytes"""

    format: Literal["png", "jpeg"]
    """Image format"""

    height: int
    """Image height in pixels"""

    tokens: int
    """Number of tokens this image represents"""

    width: int
    """Image width in pixels"""

    type: Literal["image"] = "image"

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: Union[bytes, str]) -> bytes:
        """Deserialize base64 string to bytes if needed."""
        if isinstance(value, str):
            return base64.b64decode(value)
        return value

    @field_serializer("data")
    def serialize_data(self, value: bytes) -> str:
        """Serialize bytes to base64 string for JSON."""
        return base64.b64encode(value).decode("utf-8")

    @property
    def length(self) -> int:
        return self.tokens
