from __future__ import annotations

from .._models import BaseModel
from .request_error_category import RequestErrorCategory

__all__ = ["RequestFailedResponse"]


class RequestFailedResponse(BaseModel):
    error: str
    category: RequestErrorCategory
