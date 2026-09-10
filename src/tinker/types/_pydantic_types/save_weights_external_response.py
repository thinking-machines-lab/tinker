from typing import Optional

from typing_extensions import Literal

from ..._models import BaseModel

__all__: list[str] = ["SaveWeightsExternalResponse"]


class SaveWeightsExternalResponseInternal(BaseModel):
    path: str | None = None
    """A tinker URI for the external-format model weights"""
    size_bytes: int | None = None
    """Size of the saved external weights in bytes"""

    type: Optional[Literal["save_weights_external"]] = None


class SaveWeightsExternalResponse(BaseModel):
    path: str
    """A tinker URI for the external-format model weights"""

    type: Optional[Literal["save_weights_external"]] = None
