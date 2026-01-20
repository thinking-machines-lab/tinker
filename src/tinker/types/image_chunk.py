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

    expected_tokens: int | None = None
    """Expected number of tokens this image represents.
    This is only advisory: the tinker backend will compute the number of tokens
    from the image, and we can fail requests quickly if the tokens does not
    match expected_tokens."""

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
        if self.expected_tokens is None:
            raise ValueError(
                "ImageChunk expected_tokens needs to be set in order to compute the length"
            )
        return self.expected_tokens
