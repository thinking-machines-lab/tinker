from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["HealthResponse"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
