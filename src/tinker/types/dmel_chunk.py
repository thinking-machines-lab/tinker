import base64
from typing import Union

from pydantic import field_serializer, field_validator
from typing_extensions import Literal

from .._models import StrictBase
from ._tensor_container import tensor_container_dim

__all__ = ["DmelChunk"]


class DmelChunk(StrictBase):
    dmel: bytes
    """Serialized TensorContainer bytes holding the DMel model token tensor."""

    type: Literal["dmel"] = "dmel"

    @field_validator("dmel", mode="before")
    @classmethod
    def validate_dmel(cls, value: Union[bytes, str]) -> bytes:
        """Deserialize base64 string to bytes if needed."""
        if isinstance(value, str):
            return base64.b64decode(value)
        return value

    @field_serializer("dmel")
    def serialize_dmel(self, value: bytes) -> str:
        """Serialize bytes to base64 string for JSON."""
        return base64.b64encode(value).decode("utf-8")

    @property
    def length(self) -> int:
        return tensor_container_dim(self.dmel, 0)
