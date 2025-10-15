from typing import Optional

from ..._compat import PYDANTIC_V2, ConfigDict
from ..._models import BaseModel
from ..model_id import ModelID
from ..request_id import RequestID

__all__ = ["UntypedAPIFuture"]


class UntypedAPIFuture(BaseModel):
    request_id: RequestID

    model_id: Optional[ModelID] = None
