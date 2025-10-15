from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["TelemetryResponse"]


class TelemetryResponse(BaseModel):
    status: Literal["accepted"]
